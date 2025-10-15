#!/usr/bin/env python3
"""
Layer-Specific Theta Manipulation: Visual Validation

Test the hypothesis that only high-layer theta manipulation produces consistent 
semantic changes across subjects, while low/mid-layer manipulations produce 
subject-specific structural changes.

This script reconstructs test images with different layer groups ze    # ========================================================================
    # Load Subject Data
    # ========================================================================
    print("\n[3/5] Loading subject data...")
    
    subjects = [1, 2, 5, 7]
    subject_data = {}
    
    for subject in subjects:
        print(f"  Loading Subject {subject:02d}...")
        
        # Load predicted latents
        pred_path = FEATURE_DIR / f'subj{subject:02d}' / 'nsd_vdvae_features_31l.npz'
        pred_data = np.load(pred_path)
        pred_latents = pred_data['test_latents']
        
        # Load hierarchical theta
        theta_path = THETA_DIR / f'subj{subject:02d}' / f'theta_{args.assessor}_hierarchical_subject{subject}.npy'
        theta = np.load(theta_path) zeroed (only mid+high active) - semantic changes
2. Mid layers zeroed (only low+high active) - mixed changes
3. High layers zeroed (only low+mid active) - structural changes
4. Full theta (all layers active) - baseline

Author: Brain Diffuser Analysis
Date: 2025
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Setup paths
BASE_DIR = Path('/home/rothermm/brain-diffuser')
sys.path.append(str(BASE_DIR / 'vdvae'))

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from vae import VAE
from train_helpers import restore_params
from image_utils import *
from model_utils import *

# Define directories
ASSESSOR_DIR = BASE_DIR / 'assessors'
IMG_DIR = BASE_DIR / 'results' / 'vdvae'
FEATURE_DIR = BASE_DIR / 'data' / 'extracted_features'
THETA_DIR = BASE_DIR / 'results' / 'thetas_hierarchical'

# ============================================================================
# Helper Functions
# ============================================================================

def latent_transformation_hierarchical(flat_latents, ref_latent):
    """
    Transform flat latent vectors into hierarchical layer structure.
    Uses reference latent to get correct shapes for each layer.
    
    Args:
        flat_latents: [N, D_total] numpy array of flat latent vectors
        ref_latent: stats dict from decoder with shape information
    
    Returns:
        List of 31 numpy arrays, each [N, Ci, Hi, Wi]
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
        t_lat = flat_latents[:, start:end]  # shape [N, Ci*Hi*Wi]
        c, h, w = ref_latent[i]['z'].shape[1:]
        transformed_latents.append(t_lat.reshape(len(flat_latents), c, h, w))
    
    return transformed_latents


def sample_from_hier_latents(latents, sample_ids):
    """
    Given list of numpy hierarchical latents, extract subset and convert to GPU tensors.
    
    Args:
        latents: list of 31 numpy arrays, each [N, Ci, Hi, Wi]
        sample_ids: list of indices to pick
    
    Returns:
        list of 31 GPU tensors, each [len(ids), Ci, Hi, Wi]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sample_ids = [i for i in sample_ids if i < latents[0].shape[0]]
    layers_num = len(latents)
    sample_latents = []
    for i in range(layers_num):
        subset = latents[i][sample_ids]  # numpy slice
        sample_latents.append(torch.tensor(subset, device=device).float())
    return sample_latents


def create_layer_masked_theta(theta_flat, mask_type='zero_low'):
    """
    Create theta with specific layers zeroed out.
    
    Args:
        theta_flat: [D_total] flat theta vector
        mask_type: 'zero_low' (0-10), 'zero_mid' (11-20), 'zero_high' (21-30)
    
    Returns:
        Modified theta vector
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
    
    theta_masked = theta_flat.copy()
    
    # Define layer groups
    if mask_type == 'zero_low':
        mask_layers = range(0, 11)  # Zero out layers 0-10
    elif mask_type == 'zero_mid':
        mask_layers = range(11, 21)  # Zero out layers 11-20
    elif mask_type == 'zero_high':
        mask_layers = range(21, 31)  # Zero out layers 21-30
    else:
        return theta_masked  # No masking
    
    # Zero out selected layers
    start_idx = 0
    for i, dim in enumerate(layer_dims):
        end_idx = start_idx + dim
        if i in mask_layers:
            theta_masked[start_idx:end_idx] = 0
        start_idx = end_idx
    
    return theta_masked


def reconstruct_from_latents(hierarchical_latents, ema_vae):
    """
    Reconstruct images from hierarchical latents.
    
    Args:
        hierarchical_latents: List of 31 tensors [N, Ci, Hi, Wi]
        ema_vae: VDVAE model
    
    Returns:
        List of PIL Images
    """
    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(
            len(hierarchical_latents[0]), 
            hierarchical_latents, 
            t=None
        )
        # Sample images
        sample_imgs = ema_vae.decoder.out_net.sample(px_z)
    
    # Convert to PIL Images and resize
    pil_images = []
    for img_array in sample_imgs:
        pil_img = Image.fromarray(img_array)
        pil_img = pil_img.resize((256, 256), Image.BILINEAR)
        pil_images.append(pil_img)
    
    return pil_images


class batch_generator_external_images(Dataset):
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


def plot_cross_subject_layer_manipulations(reconstructions, image_idx, conditions_order, 
                                           subjects, alpha):
    """
    Create grid showing all subjects × all conditions for one image.
    
    Layout:
    - Rows: Subjects (1, 2, 5, 7)
    - Columns: Conditions (Full, Low Zeroed, Mid Zeroed, High Zeroed)
    """
    n_subjects = len(subjects)
    n_conditions = len(conditions_order)
    
    fig, axes = plt.subplots(n_subjects, n_conditions, 
                            figsize=(4*n_conditions, 4*n_subjects))
    
    for i, subject in enumerate(subjects):
        for j, condition in enumerate(conditions_order):
            ax = axes[i, j] if n_subjects > 1 else axes[j]
            
            # Get image
            img = reconstructions[subject][condition][image_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # Title on top row
            if i == 0:
                ax.set_title(condition, fontweight='bold', fontsize=12, pad=10)
            
            # Subject label on left column
            if j == 0:
                ax.text(-0.15, 0.5, f'Subject {subject:02d}', 
                       transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontweight='bold', fontsize=12)
    
    plt.suptitle(f'Layer-Specific Theta Manipulation: Image {image_idx} (α={alpha})\n'
                 f'EmoNet Hierarchical Theta', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Layer-Specific Theta Manipulation Visual Validation')
    parser.add_argument('--assessor', type=str, default='emonet', 
                       choices=['emonet', 'memnet'],
                       help='Assessor type (emonet or memnet)')
    parser.add_argument('--alpha', type=float, default=50.0,
                       help='Manipulation strength (default: 50)')
    parser.add_argument('--n_images', type=int, default=3,
                       help='Number of test images to reconstruct (default: 3)')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/rothermm/brain-diffuser/results/hierachical_theta',
                       help='Output directory for saved images')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Layer-Specific Theta Manipulation: Visual Validation")
    print("="*70)
    print(f"Assessor: {args.assessor}")
    print(f"Alpha: {args.alpha}")
    print(f"Number of images: {args.n_images}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    # ========================================================================
    # Setup VDVAE Model
    # ========================================================================
    print("\n[1/5] Loading VDVAE model...")
    
    model_dir = str(BASE_DIR / 'vdvae' / 'model')
    
    H = {
        'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500,
        'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
        'hparam_sets': 'imagenet64',
        'restore_path': f'{model_dir}/imagenet64-iter-1600000-model.th',
        'restore_ema_path': f'{model_dir}/imagenet64-iter-1600000-model-ema.th',
        'restore_log_path': f'{model_dir}/imagenet64-iter-1600000-log.jsonl',
        'restore_optimizer_path': f'{model_dir}/imagenet64-iter-1600000-opt.th',
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
    
    # Setup data and load model
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)
    
    print(f"✓ VDVAE model loaded")
    print(f"✓ Image size: {H.image_size}x{H.image_size}")
    
    # ========================================================================
    # Get Reference Latent Structure
    # ========================================================================
    print("\n[2/5] Loading test images to get reference latent structure...")
    
    image_path = BASE_DIR / 'data' / 'processed_data' / 'subj01' / 'nsd_test_stim_sub1.npy'
    test_images = batch_generator_external_images(data_path=str(image_path))
    testloader = DataLoader(test_images, batch_size=30, shuffle=False)
    
    # Get reference latent by encoding a batch
    print("Encoding batch to get layer structure...")
    ref_latent = None
    for i, x in enumerate(testloader):
        data_input, target = preprocess_fn(x)
        with torch.no_grad():
            activations = ema_vae.encoder(data_input)
            px_z, stats = ema_vae.decoder(activations, get_latents=True)
            ref_latent = stats
            break  # Only need one batch
    
    print(f"✓ Reference latent structure obtained ({len(ref_latent)} layers)")
    
    # ========================================================================
    # Load Subject Data
    # ========================================================================
    print("\n[3/5] Loading subject data...")
    
    subjects = [1, 2, 5, 7]
    subject_data = {}
    
    for subject in subjects:
        print(f"  Loading Subject {subject:02d}...")
        
        # Load predicted latents
        pred_path = FEATURE_DIR / f'subj{subject:02d}' / 'nsd_vdvae_features_31l.npz'
        pred_data = np.load(pred_path)
        pred_latents = pred_data['test_latents']
        
        # Load hierarchical theta
        theta_path = THETA_DIR / f'subj{subject:02d}' / f'theta_{args.assessor}_hierarchical_subject{subject}.npy'
        theta = np.load(theta_path)
        
        subject_data[subject] = {
            'pred_latents': pred_latents,
            'theta': theta
        }
        
        print(f"    Latents: {pred_latents.shape}, Theta: {theta.shape}")
    
    print("✓ All subject data loaded")
    
    # ========================================================================
    # Generate Reconstructions
    # ========================================================================
    print("\n[4/5] Generating layer-masked reconstructions...")
    
    alpha = args.alpha  # Manipulation strength
    n_images = args.n_images  # Number of test images
    image_indices = list(range(n_images))
    
    # Conditions to test
    conditions = {
        'Full Theta': None,
        'Low Zeroed\n(Mid+High active)': 'zero_low',
        'Mid Zeroed\n(Low+High active)': 'zero_mid',
        'High Zeroed\n(Low+Mid active)': 'zero_high'
    }
    
    print(f"  α = {alpha}")
    print(f"  Testing {len(conditions)} conditions")
    print(f"  Processing {n_images} images across {len(subjects)} subjects")
    
    all_reconstructions = {}
    
    for subject in subjects:
        print(f"\n  Processing Subject {subject:02d}...")
        
        all_reconstructions[subject] = {}
        data = subject_data[subject]
        
        for cond_name, mask_type in conditions.items():
            # Create theta for this condition
            if mask_type is None:
                theta_cond = data['theta']
            else:
                theta_cond = create_layer_masked_theta(data['theta'], mask_type)
                n_nonzero = np.count_nonzero(theta_cond)
                print(f"    {cond_name.replace(chr(10), ' ')}: {n_nonzero}/{len(theta_cond)} non-zero")
            
            # Manipulate latents: z_new = z + α * θ
            manipulated_latents = data['pred_latents'][image_indices] + alpha * theta_cond
            
            # Transform to hierarchical structure
            hierarchical = latent_transformation_hierarchical(manipulated_latents, ref_latent)
            
            # Convert to tensors and reconstruct
            hier_tensors = sample_from_hier_latents(hierarchical, list(range(n_images)))
            reconstructed_imgs = reconstruct_from_latents(hier_tensors, ema_vae)
            
            all_reconstructions[subject][cond_name] = reconstructed_imgs
    
    print("\n✓ All reconstructions complete!")
    
    # ========================================================================
    # Save and Visualize Results
    # ========================================================================
    print("\n[5/5] Saving visualizations...")
    
    conditions_order = ['Full Theta', 'Low Zeroed\n(Mid+High active)', 
                       'Mid Zeroed\n(Low+High active)', 'High Zeroed\n(Low+Mid active)']
    
    for img_idx in range(n_images):
        print(f"  Saving Image {img_idx}...")
        
        fig = plot_cross_subject_layer_manipulations(
            all_reconstructions, 
            img_idx, 
            conditions_order,
            subjects,
            alpha
        )
        
        # Save figure
        save_path = output_dir / f'layer_manipulation_{args.assessor}_image{img_idx}_alpha{int(alpha)}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        plt.close(fig)
    
    print("\n" + "="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print(f"\nSaved {n_images} visualizations to: {output_dir}")
    print(f"Assessor: {args.assessor}")
    print(f"Alpha: {alpha}")
    print("\nInterpretation Guide:")
    print("  Column 1 (Full Theta): Baseline with all layers active")
    print("  Column 2 (Low Zeroed): Semantic changes (should be consistent across subjects)")
    print("  Column 3 (Mid Zeroed): Mixed structural and semantic changes")
    print("  Column 4 (High Zeroed): Structural changes (should be individual-specific)")
    print("\nExpected: Rows look SIMILAR in Column 2, DIFFERENT in Column 4")
    print("="*70)


if __name__ == "__main__":
    main()
