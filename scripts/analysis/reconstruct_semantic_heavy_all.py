#!/usr/bin/env python3
"""
Generate reconstructions using semantic_heavy theta for all subjects.
This script reconstructs all test images for all subjects using the semantic_heavy
theta variant across a range of alpha values.
"""
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

# Import VDVAE modules
from hps import Hyperparams
from vae import VAE
from model_utils import set_up_data, load_vaes

# --- Parse arguments ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Reconstruct all images using semantic_heavy theta')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
args = parser.parse_args()

sub = args.sub
assert sub in [1, 2, 5, 7], f"Subject must be in [1, 2, 5, 7], got {sub}"
batch_size = args.bs

print(f'=== Semantic Heavy Theta Reconstruction ===')
print(f'Subject: {sub}')
print(f'Batch size: {batch_size}')
print(f'Variant: semantic_heavy')
print('Libs imported')

# --- Define paths ------------------------------------------------------------
BASE_DIR = Path('/home/rothermm/brain-diffuser')
THETA_DIR = BASE_DIR / 'results' / 'hybrid_theta'
PRED_FEATURE_DIR = BASE_DIR / 'data' / 'predicted_features'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed_data'
MODEL_DIR = BASE_DIR / 'vdvae' / 'model'
OUTPUT_DIR = BASE_DIR / 'results' / 'hybrid_theta_reconstructions'

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
n_images = len(pred_latents)

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

# --- Load semantic_heavy theta for both assessors ----------------------------
print(f'Loading semantic_heavy theta for subject {sub:02d}...')
ASSESSORS = ['emonet', 'memnet']
VARIANT = 'semantic_heavy'

thetas = {}
for assessor in ASSESSORS:
    theta_path = THETA_DIR / f'subj{sub:02d}' / f'{assessor}_theta_{VARIANT}.npy'
    if not theta_path.exists():
        print(f'❌ ERROR: Theta not found: {theta_path}')
        sys.exit(1)
    thetas[assessor] = np.load(theta_path)
    print(f'✓ Loaded {assessor}/{VARIANT}: {thetas[assessor].shape}')

# --- Define alpha values -----------------------------------------------------
ALPHAS = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
print(f'\nAlpha values: {ALPHAS}')
print(f'Reconstructing all {n_images} images for subject {sub:02d}...\n')

# --- Main reconstruction loop ------------------------------------------------
total_images = len(ASSESSORS) * len(ALPHAS) * n_images
progress_counter = 0

for assessor in ASSESSORS:
    print(f'\n{"="*70}')
    print(f'Processing {assessor.upper()}')
    print(f'{"="*70}')
    
    theta = thetas[assessor]
    
    for alpha in ALPHAS:
        print(f'\n  Alpha = {alpha}')
        
        # Create output directory
        out_dir = OUTPUT_DIR / f'subj{sub:02d}' / assessor / VARIANT / f'alpha_{alpha}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        num_batches = (n_images + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_images)
            batch_latents = pred_latents[start_idx:end_idx]
            
            # Apply theta manipulation
            if alpha == 0:
                mod_latents = batch_latents
            else:
                mod_latents = batch_latents + alpha * theta
            
            # Transform to hierarchical structure
            hier = latent_transformation(mod_latents, ref_stats)
            
            # Sample from hierarchical latents
            sample_ids = list(range(len(batch_latents)))
            samp = sample_from_hier_latents(hier, sample_ids)
            
            # Decode
            with torch.no_grad():
                px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
                sample_imgs = ema_vae.decoder.out_net.sample(px_z)
            
            # Save images
            for i, img_array in enumerate(sample_imgs):
                img_idx = start_idx + i
                pil = Image.fromarray(img_array)
                pil = pil.resize((512, 512), resample=Image.BILINEAR)
                fname = f'img_{img_idx:03d}.png'
                pil.save(out_dir / fname)
                
                progress_counter += 1
                if progress_counter % 100 == 0:
                    print(f'    Progress: {progress_counter}/{total_images} images ({100*progress_counter/total_images:.1f}%)')
        
        print(f'    ✓ Completed alpha={alpha}: {n_images} images saved to {out_dir}')
    
    print(f'\n✓ Completed {assessor}')

print(f'\n{"="*70}')
print(f'ALL RECONSTRUCTIONS COMPLETED FOR SUBJECT {sub:02d}')
print(f'{"="*70}')
print(f'Total images generated: {progress_counter}')
print(f'Output directory: {OUTPUT_DIR / f"subj{sub:02d}"}')
print(f'\n✓ Script completed successfully!')
