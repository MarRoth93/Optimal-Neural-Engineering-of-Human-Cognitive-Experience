#!/usr/bin/env python3
import sys
sys.path.append('/home/rothermm/brain-diffuser/vdvae')

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import json

# Import VDVAE modules
from hps import Hyperparams
from vae import VAE
from model_utils import set_up_data, load_vaes

# --- Parse arguments ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Reconstruct images from hierarchical theta')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
parser.add_argument("--n_images", help="Number of images to reconstruct (default: all)", default=None, type=int)
args = parser.parse_args()

sub = args.sub
assert sub in [1, 2, 5, 7], f"Subject must be in [1, 2, 5, 7], got {sub}"
batch_size = args.bs
n_images = args.n_images

print(f'=== Hierarchical Theta Reconstruction ===')
print(f'Subject: {sub}')
print(f'Batch size: {batch_size}')
print(f'N images: {n_images if n_images else "all"}')
print('Libs imported')

# --- Define paths ------------------------------------------------------------
BASE_DIR = Path('/home/rothermm/brain-diffuser')
THETA_DIR = BASE_DIR / 'results' / 'hybrid_theta'  # Updated to use hybrid_theta directory
PRED_FEATURE_DIR = BASE_DIR / 'data' / 'predicted_features'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed_data'
MODEL_DIR = BASE_DIR / 'vdvae' / 'model'
OUTPUT_DIR = BASE_DIR / 'results' / 'hybrid_theta_reconstructions'  # Updated output directory

# --- Model & hyperparameters setup --------------------------------------------
H = {
    'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500,
    'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
    'hparam_sets': 'imagenet64',
    'restore_path': str(MODEL_DIR / 'imagenet64-iter-1600000-model.th'),
    'restore_ema_path': str(MODEL_DIR / 'imagenet64-iter-1600000-model-ema.th'),
    'restore_log_path': str(MODEL_DIR / 'imagenet64-iter-1600000-log.jsonl'),
    'restore_optimizer_path': str(MODEL_DIR / 'imagenet64-iter-1600000-opt.th'),
    'dataset': 'imagenet64', 'ema_rate': 0.999,
    'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
    'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
    'zdim': 16, 'width': 512, 'custom_width_str': '',
    'bottleneck_multiple': 0.25, 'no_bias_above': 64,
    'scale_encblock': False, 'test_eval': True,
    'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0,
    'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015,
    'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
    'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
    'iters_per_ckpt': 25000, 'iters_per_print': 1000,
    'iters_per_save': 10000, 'iters_per_images': 10000,
    'epochs_per_eval': 1, 'epochs_per_probe': None,
    'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
    'mpi_size': 1, 'local_rank': 0, 'rank': 0,
    'logdir': './saved_models/test/log'
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

H = dotdict(H)

# This returns (possibly updated H, preprocess_fn)
H, preprocess_fn = set_up_data(H)

print('Model is Loading...')
ema_vae = load_vaes(H)
print('✓ VDVAE model loaded successfully')

# --- Dataset class -----------------------------------------------------------
class BatchGeneratorExternalImages(Dataset):
    """Dataset class for loading external images."""
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self, idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img, (64, 64))
        img = torch.tensor(np.array(img)).float()
        return img

    def __len__(self):
        return len(self.im)

# --- Load test stimuli & collect reference stats -----------------------------
print(f'Loading test stimuli for subject {sub:02d}...')
image_path = PROCESSED_DATA_DIR / f'subj{sub:02d}' / f'nsd_test_stim_sub{sub}.npy'
test_images = BatchGeneratorExternalImages(data_path=str(image_path))
testloader = DataLoader(test_images, batch_size, shuffle=False)

# Get reference stats from first batch
print('Getting reference stats structure...')
x = next(iter(testloader))
data_input, target = preprocess_fn(x)
with torch.no_grad():
    activations = ema_vae.encoder(data_input)
    px_z, ref_stats = ema_vae.decoder(activations, get_latents=True)

print(f'✓ Reference stats obtained (31 layers)')

# --- Load predicted latents --------------------------------------------------
print(f'Loading predicted latents for subject {sub:02d}...')
pred_latents_path = PRED_FEATURE_DIR / f'subj{sub:02d}' / f'nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy'
pred_latents = np.load(pred_latents_path)
print(f'✓ Loaded predicted latents: {pred_latents.shape}')

# Subset if n_images specified
if n_images is not None:
    pred_latents = pred_latents[:n_images]
    print(f'✓ Using first {n_images} images')

# --- Latent transformation function ------------------------------------------
def latent_transformation(latents, ref):
    """
    Splits flat latent vectors into 31 hierarchical layers using hard-coded dims.
    ref is the `stats` list from one decoder forward, where each ref[i]['z'].shape = [B, Ci, Hi, Wi].
    """
    layer_dims = np.array([
        2**4,  2**4,
        2**8,  2**8,  2**8,  2**8,
        2**10, 2**10, 2**10, 2**10,
        2**10, 2**10, 2**10, 2**10,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**14
    ])
    transformed_latents = []
    for i in range(31):
        start = layer_dims[:i].sum()
        end = layer_dims[:i+1].sum()
        t_lat = latents[:, start:end]  # shape [N, Ci*Hi*Wi]
        c, h, w = ref[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(latents), c, h, w))
    return transformed_latents

# --- Wrapper to pick out minibatch layers & send to GPU ---------------------
def sample_from_hier_latents(latents, sample_ids):
    """
    Given:
      - latents: list of 31 numpy arrays, each [N, Ci, Hi, Wi]
      - sample_ids: a list of indices to pick from dimension 0
    Returns a list of 31 GPU tensors, each [len(ids), Ci, Hi, Wi].
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_ids = [i for i in sample_ids if i < latents[0].shape[0]]
    layers_num = len(latents)
    sample_latents = []
    for i in range(layers_num):
        subset = latents[i][sample_ids]  # numpy slice
        sample_latents.append(torch.tensor(subset, device=device).float())
    return sample_latents

# --- Load hierarchical theta vectors for both assessors ----------------------
print(f'Loading hybrid theta variants for subject {sub:02d}...')
ASSESSORS = ['emonet', 'memnet']
VARIANTS = ['original', 'semantic_heavy', 'semantic_only', 'balanced', 'structural_heavy', 'structural_only']

# Dictionary structure: thetas[assessor][variant] = theta_array
thetas = {}

for assessor in ASSESSORS:
    thetas[assessor] = {}
    for variant in VARIANTS:
        theta_path = THETA_DIR / f'subj{sub:02d}' / f'{assessor}_theta_{variant}.npy'
        if not theta_path.exists():
            print(f'⚠️  Warning: Theta not found: {theta_path}')
            continue
        thetas[assessor][variant] = np.load(theta_path)
        print(f'✓ Loaded {assessor}/{variant}: {thetas[assessor][variant].shape}')

# Check if we have any thetas
if not any(thetas.values()):
    print('❌ ERROR: No theta vectors found. Exiting.')
    sys.exit(1)

print(f'\n✓ Loaded {sum(len(v) for v in thetas.values())} theta variants total')

# --- Define alpha values -----------------------------------------------------
# For hybrid theta comparison, we use a fine-grained alpha range from -2 to +2
ALPHAS = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]

print(f'\nAlpha values: {ALPHAS}')
print(f'Reconstructing image index 2 for all theta variants...')
print(f'Starting reconstruction loop...\n')

# Build the base hierarchical latent for image 2
img_idx = 2
pred_latents_img2 = pred_latents[img_idx:img_idx+1]  # Shape: [1, D_flat]
base_hier = latent_transformation(pred_latents_img2, ref_stats)

# --- Helper function to ablate (zero-out) specific layer groups -------------
def create_ablated_theta(theta, active_layers):
    """
    Create a theta vector with only specified layers active (others zeroed).
    
    Args:
        theta: Full flat theta vector
        active_layers: List of layer indices to keep active
    
    Returns:
        ablated_theta: Theta with non-active layers set to zero
    """
    # Layer dimensions for determining which indices to zero
    layer_dims = np.array([
        2**4,  2**4,
        2**8,  2**8,  2**8,  2**8,
        2**10, 2**10, 2**10, 2**10,
        2**10, 2**10, 2**10, 2**10,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**14
    ])
    
    ablated_theta = theta.copy()
    
    # Zero out inactive layers
    for layer_idx in range(31):
        if layer_idx not in active_layers:
            start = layer_dims[:layer_idx].sum()
            end = layer_dims[:layer_idx+1].sum()
            ablated_theta[start:end] = 0
    
    return ablated_theta

# --- Analytics functions -----------------------------------------------------
def analyze_theta_statistics(theta, assessor_name):
    """
    Comprehensive statistical analysis of theta vector.
    
    Returns:
        dict: Statistics by layer group
    """
    layer_dims = np.array([
        2**4,  2**4,
        2**8,  2**8,  2**8,  2**8,
        2**10, 2**10, 2**10, 2**10,
        2**10, 2**10, 2**10, 2**10,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**12, 2**12, 2**12, 2**12,
        2**14
    ])
    
    stats = {}
    
    for group_name, active_layers in LAYER_GROUPS.items():
        if group_name == 'all':
            continue
            
        # Get indices for this group
        start_indices = []
        end_indices = []
        for layer_idx in active_layers:
            start = layer_dims[:layer_idx].sum()
            end = layer_dims[:layer_idx+1].sum()
            start_indices.append(start)
            end_indices.append(end)
        
        # Extract theta values for this group
        group_theta = np.concatenate([theta[s:e] for s, e in zip(start_indices, end_indices)])
        
        stats[group_name] = {
            'n_layers': len(active_layers),
            'n_params': len(group_theta),
            'l2_norm': np.linalg.norm(group_theta),
            'l1_norm': np.sum(np.abs(group_theta)),
            'mean': np.mean(group_theta),
            'std': np.std(group_theta),
            'mean_abs': np.mean(np.abs(group_theta)),
            'max_abs': np.max(np.abs(group_theta)),
            'sparsity': np.sum(np.abs(group_theta) < 0.0001) / len(group_theta),
            'param_fraction': len(group_theta) / len(theta)
        }
    
    return stats


def compute_image_differences(img1, img2):
    """
    Compute multiple difference metrics between two images.
    
    Args:
        img1, img2: PIL Images or numpy arrays
        
    Returns:
        dict: Various difference metrics
    """
    # Convert to numpy if needed
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    if isinstance(img2, Image.Image):
        img2 = np.array(img2)
    
    # Ensure float
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Normalize to [0, 1]
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Compute metrics
    mse = np.mean((img1 - img2) ** 2)
    mae = np.mean(np.abs(img1 - img2))
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Structural similarity (simplified SSIM)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    
    # Cosine similarity (flatten images)
    flat1 = img1.flatten()
    flat2 = img2.flatten()
    cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'cosine_similarity': float(cos_sim),
        'l2_distance': float(np.linalg.norm(img1 - img2)),
        'max_pixel_diff': float(np.max(np.abs(img1 - img2)))
    }


def analyze_ablation_effects(assessor, sub, img_idx=2):
    """
    Quantify the effect of each layer group ablation.
    
    Returns:
        DataFrame with ablation effect metrics
    """
    results = []
    
    for group_name in ['low_idx', 'mid_idx', 'high_idx']:
        # Load alpha=-2, 0, +2 images
        alpha_neg_path = OUTPUT_DIR / f'subj{sub:02d}' / assessor / group_name / 'alpha_-2' / f'img_{img_idx:03d}.png'
        alpha_zero_path = OUTPUT_DIR / f'subj{sub:02d}' / assessor / group_name / 'alpha_0' / f'img_{img_idx:03d}.png'
        alpha_pos_path = OUTPUT_DIR / f'subj{sub:02d}' / assessor / group_name / 'alpha_2' / f'img_{img_idx:03d}.png'
        
        if not all([p.exists() for p in [alpha_neg_path, alpha_zero_path, alpha_pos_path]]):
            continue
        
        img_neg = np.array(Image.open(alpha_neg_path))
        img_zero = np.array(Image.open(alpha_zero_path))
        img_pos = np.array(Image.open(alpha_pos_path))
        
        # Compute differences
        diff_neg_zero = compute_image_differences(img_neg, img_zero)
        diff_pos_zero = compute_image_differences(img_pos, img_zero)
        diff_neg_pos = compute_image_differences(img_neg, img_pos)
        
        results.append({
            'group': group_name,
            'neg_to_zero_mse': diff_neg_zero['mse'],
            'pos_to_zero_mse': diff_pos_zero['mse'],
            'neg_to_pos_mse': diff_neg_pos['mse'],
            'neg_to_zero_mae': diff_neg_zero['mae'],
            'pos_to_zero_mae': diff_pos_zero['mae'],
            'neg_to_pos_mae': diff_neg_pos['mae'],
            'neg_to_zero_ssim': diff_neg_zero['ssim'],
            'pos_to_zero_ssim': diff_pos_zero['ssim'],
            'neg_to_pos_ssim': diff_neg_pos['ssim'],
            'total_effect': (diff_neg_zero['mse'] + diff_pos_zero['mse']) / 2
        })
    
    return pd.DataFrame(results)


# --- Main reconstruction loop: all theta variants ----------------------------
for assessor, variants_dict in thetas.items():
    print(f'\n{"="*70}')
    print(f'Processing {assessor.upper()}')
    print(f'{"="*70}')
    
    for variant_name, theta in variants_dict.items():
        print(f'\n  Variant: {variant_name.upper()}')
        
        for alpha in ALPHAS:
            print(f'    Alpha = {alpha}')
            
            # If alpha = 0, use base hierarchy; otherwise apply theta shift
            if alpha == 0:
                hier = base_hier
            else:
                mod_flat = pred_latents_img2 + alpha * theta  # shape [1, D_flat]
                hier = latent_transformation(mod_flat, ref_stats)
            
            # Create output dir
            out_dir = OUTPUT_DIR / f'subj{sub:02d}' / assessor / variant_name / f'alpha_{alpha}'
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Decode the single image
            samp = sample_from_hier_latents(hier, [0])  # Single image
            with torch.no_grad():
                px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
                sample_imgs = ema_vae.decoder.out_net.sample(px_z)
            
            # Save image
            pil = Image.fromarray(sample_imgs[0])
            pil = pil.resize((512, 512), resample=Image.BILINEAR)
            fname = f'img_{img_idx:03d}.png'
            pil.save(out_dir / fname)
            
            print(f'      ✓ Saved to {out_dir / fname}')
    
    print(f'\n✓ Completed {assessor}')

print(f'\n{"="*70}')
print(f'ALL RECONSTRUCTIONS COMPLETED FOR SUBJECT {sub:02d}')
print(f'{"="*70}')

# --- Create comparison visualization for theta variants ---------------------
print(f'\n{"="*70}')
print('Creating hybrid theta variant comparison visualization...')
print(f'{"="*70}')

for assessor in thetas.keys():
    print(f'\nGenerating plot for {assessor}, subject {sub:02d}...')
    
    # Get all variants for this assessor
    variants = list(thetas[assessor].keys())
    n_variants = len(variants)
    n_alphas = len(ALPHAS)
    
    # Create figure: rows=variants, cols=alphas
    # With 9 alphas, use smaller per-image size to keep figure manageable
    fig = plt.figure(figsize=(3*n_alphas, 4*n_variants))
    gs = gridspec.GridSpec(n_variants, n_alphas, figure=fig, hspace=0.3, wspace=0.05)
    
    # Color mapping for variant types
    variant_colors = {
        'original': 'lightgray',
        'semantic_heavy': 'lightcoral',
        'semantic_only': 'lightsalmon',
        'balanced': 'lightgreen',
        'structural_heavy': 'lightblue',
        'structural_only': 'lightskyblue'
    }
    
    for row_idx, variant_name in enumerate(variants):
        for col_idx, alpha in enumerate(ALPHAS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Load image
            img_path = OUTPUT_DIR / f'subj{sub:02d}' / assessor / variant_name / f'alpha_{alpha}' / f'img_{img_idx:03d}.png'
            
            if img_path.exists():
                img = plt.imread(img_path)
                ax.imshow(img)
            else:
                # Show placeholder if image doesn't exist
                ax.text(0.5, 0.5, 'Image\nNot Found', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
            
            ax.axis('off')
            
            # Add title on top row
            if row_idx == 0:
                ax.set_title(f'α = {alpha}', fontsize=14, fontweight='bold')
            
            # Add variant label on left column
            if col_idx == 0:
                # Create descriptive label
                if variant_name == 'original':
                    label_text = 'ORIGINAL\n(1.0, 1.0, 1.0)\nNo weighting'
                elif variant_name == 'semantic_heavy':
                    label_text = 'SEMANTIC HEAVY\n(1.0, 0.3, 0.05)\nStrong semantic focus'
                elif variant_name == 'semantic_only':
                    label_text = 'SEMANTIC ONLY\n(1.0, 0.2, 0.0)\nMax structure preservation'
                elif variant_name == 'balanced':
                    label_text = 'BALANCED\n(1.0, 0.6, 0.2)\nModerate all layers'
                elif variant_name == 'structural_heavy':
                    label_text = 'STRUCTURAL HEAVY\n(0.3, 0.6, 1.0)\nStrong structural focus'
                elif variant_name == 'structural_only':
                    label_text = 'STRUCTURAL ONLY\n(0.0, 0.2, 1.0)\nMax semantic preservation'
                else:
                    label_text = variant_name.upper()
                
                ax.text(-0.05, 0.5, label_text, 
                       transform=ax.transAxes,
                       fontsize=11, fontweight='bold',
                       ha='right', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=variant_colors.get(variant_name, 'white'), 
                                edgecolor='black', 
                                linewidth=1.5,
                                alpha=0.9))
    
    # Overall title
    assessor_name = 'EmoNet (Valence)' if assessor == 'emonet' else 'MemNet (Memorability)'
    fig.suptitle(f'{assessor_name} - Hybrid Theta Variants Comparison\n'
                 f'Subject {sub:02d}, Image {img_idx}', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    # Add legend/explanation at bottom
    legend_text = (
        'Theta Variants: Different layer-group weightings (LOW, MID, HIGH)\n'
        'LOW (0-10) = Semantic layers | MID (11-20) = Mid-level | HIGH (21-30) = Structural layers\n'
        'Alpha range: -2 to +2 | Negative values decrease attribute | 0 = Baseline | Positive values increase attribute\n'
        'Weights format: (LOW, MID, HIGH) - Higher weight = stronger manipulation in that layer group'
    )
    fig.text(0.5, 0.02, legend_text, 
             ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.8', 
                      facecolor='wheat', 
                      edgecolor='black',
                      linewidth=1.5,
                      alpha=0.9),
             wrap=True)
    
    # Save figure
    fig_path = OUTPUT_DIR / f'subj{sub:02d}' / f'hybrid_theta_comparison_{assessor}_img{img_idx}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f'  ✓ Saved: {fig_path}')
    plt.close()


print(f'\n{"="*70}')
print('✓ All visualizations completed!')
print(f'{"="*70}')
print(f'\nOutput directory: {OUTPUT_DIR / f"subj{sub:02d}"}')
print(f'Comparison plots saved for each assessor.')
print(f"\n✓ Script completed successfully!")
