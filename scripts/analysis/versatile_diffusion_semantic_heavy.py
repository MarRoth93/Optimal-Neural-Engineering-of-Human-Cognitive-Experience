#!/usr/bin/env python3
import sys
sys.path.append('/home/psycontrol/01_Marco_ssd/01_fmri_recon/data/model/versatile_diffusion')

# Absolute path to project root
BASE_DIR = "/home/psycontrol/01_Marco_ssd/01_fmri_recon"
import os
os.chdir(BASE_DIR)

import PIL
from PIL import Image
from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as tvtrans

# Workaround for CUDA compatibility issues
# The VD model uses torch.cuda.device_count() which may return 0 if CUDA has issues
# We'll monkey-patch it to return at least 1 to avoid division by zero
_original_device_count = torch.cuda.device_count

def _patched_device_count():
    try:
        count = _original_device_count()
        return max(count, 1)  # Ensure at least 1 is returned
    except:
        return 1  # Default to 1 if any error occurs

torch.cuda.device_count = _patched_device_count

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist

# --- Configuration (directly defined - no argument parsing) ----------------
# Subject and processing settings
sub = 1  # Only subject 1

# Versatile Diffusion parameters
strength = 0.7  # Diffusion strength (0.0-1.0). Lower = more structure preservation
mixing = 0.5    # Mixing ratio between vision/text conditioning (0.5 = balanced)
ddim_steps = 50  # Number of DDIM sampling steps
scale = 7.5      # Unconditional guidance scale
ddim_eta = 0.0   # DDIM eta parameter (0.0 = deterministic)

# Processing scope
assessors = ['emonet', 'memnet']  # Which assessors to process
alphas = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]  # Alpha values to process

# Fixed variant for this script
VARIANT = 'semantic_heavy'

# Additional processing parameters (from notebook)
batch_size = 1  # Process one image at a time
n_samples = 1   # Generate one sample per image

# Validation
assert 0.0 <= strength <= 1.0, "diff_str must be in [0.0, 1.0]"
assert 0.0 <= mixing <= 1.0, "mix_str must be in [0.0, 1.0]"

# Warn if strength is too high for structure preservation
if strength > 0.5:
    print(f"\n{'='*70}")
    print(f"⚠️  WARNING: High diffusion strength ({strength:.2f})")
    print(f"{'='*70}")
    print(f"   High strength may significantly alter structural properties!")
    print(f"   Recommended range: 0.2-0.4 for structure preservation")
    print(f"   while allowing semantic changes from theta manipulation.")
    print(f"{'='*70}\n")

# --- Versatile Diffusion model loading ----------------------------------------
def regularize_image(x):
    """
    Regularize PIL image to torch tensor [3, 512, 512] in range [0, 1].
    """
    BICUBIC = PIL.Image.Resampling.BICUBIC
    if isinstance(x, str):
        x = Image.open(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, PIL.Image.Image):
        x = x.resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, np.ndarray):
        x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
        x = tvtrans.ToTensor()(x)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        assert False, 'Unknown image type'
    
    assert (x.shape[1]==512) & (x.shape[2]==512), \
        'Wrong image size'
    return x

print("Loading Versatile Diffusion model...")
cfgm_name = 'vd_noema'
pth = '/home/psycontrol/01_Marco_ssd/01_fmri_recon/data/model/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

print(f"Number of available GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")

# --- Device setup -------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear CUDA cache
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Move model components to device
net.clip.to(device)
net.autokl.to(device)
net.autokl.half()

sampler = DDIMSampler_VD(net)
print(f"✓ Model loaded successfully\n")

# --- Load predicted CLIP latents once ----------------------------------------
print(f"Loading predicted CLIP features for subject {sub:02d}...")

# Load the predicted features from files
# pred_text = np.load(f'/media/data/01_Marco_hdd/fMRI_recon/data/generated_data/predicted_features/subj{sub:02d}/nsd_cliptext_predtest_nsdgeneral.npy')
pred_vision = np.load(f'/media/data/01_Marco_hdd/fMRI_recon/data/generated_data/predicted_features/subj{sub:02d}/nsd_clipvision_predtest_nsdgeneral.npy')

pred_text = np.load(f'/home/psycontrol/01_Marco_ssd/01_fmri_recon/data/generated_data/extracted_features/subj{sub:02d}/nsd_cliptext_test.npy')
# pred_vision = np.load(f'/home/psycontrol/01_Marco_ssd/01_fmri_recon/data/generated_data/extracted_features/subj{sub:02d}/nsd_clipvision_test.npy')


# Try moving tensors to the GPU, and handle any errors that occur
try:
    pred_text_all = torch.tensor(pred_text).half().to(device)
    pred_vision_all = torch.tensor(pred_vision).half().to(device)
    print(f"✓ Loaded vision features: {pred_vision_all.shape}")
    print(f"✓ Loaded text features: {pred_text_all.shape}")
except RuntimeError as e:
    print(f"Error moving tensors to GPU: {e}")
    print("Attempting to use CPU instead...")
    device = torch.device("cpu")
    pred_text_all = torch.tensor(pred_text).half()
    pred_vision_all = torch.tensor(pred_vision).half()
    print(f"✓ Loaded vision features (CPU): {pred_vision_all.shape}")
    print(f"✓ Loaded text features (CPU): {pred_text_all.shape}")
print()

# --- Define paths ------------------------------------------------------------
base_input_dir = Path('/media/data/01_Marco_hdd/fMRI_recon/generated images/hierachical_theta/hybrid_theta_reconstructions')
base_output_dir = Path('/media/data/01_Marco_hdd/fMRI_recon/generated images/versatile_diffusion_semantic_heavy')

print(f"{'='*70}")
print(f"PROCESSING CONFIGURATION")
print(f"{'='*70}")
print(f"Subject: {sub:02d}")
print(f"Variant: {VARIANT} (semantic manipulation with structure preservation)")
print(f"Diffusion strength: {strength:.3f} (structure preservation)")
print(f"Mixing ratio: {mixing:.3f}")
print(f"  - Vision weight: {1-mixing:.3f}")
print(f"  - Text weight: {mixing:.3f}")
print(f"DDIM steps: {ddim_steps}")
print(f"Guidance scale: {scale}")
print(f"Assessors: {assessors}")
print(f"Alphas: {alphas}")
print(f"{'='*70}\n")

# --- Main processing loop ----------------------------------------------------
# Clear CUDA cache and set random seed
if device.type == 'cuda':
    torch.cuda.empty_cache()
torch.manual_seed(0)  # For reproducibility

total_processed = 0
total_skipped = 0

for assessor_name in assessors:
    print(f"\n{'='*70}")
    print(f"PROCESSING ASSESSOR: {assessor_name.upper()}")
    print(f"{'='*70}\n")
    
    for alpha in alphas:
        print(f"\n  Alpha = {alpha}")
        
        # Input directory: subj{XX}/{assessor}/semantic_heavy/alpha_{alpha}/
        in_dir = base_input_dir / f'subj{sub:02d}' / assessor_name / VARIANT / f'alpha_{alpha}'
        
        if not in_dir.exists():
            print(f"  ⚠️  Input directory does not exist: {in_dir}")
            print(f"     Skipping...")
            total_skipped += 1
            continue
        
        # Output directory: mirror the input structure
        out_dir = base_output_dir / f'subj{sub:02d}' / assessor_name / VARIANT / f'alpha_{alpha}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Input:  {in_dir}")
        print(f"  Output: {out_dir}")
            
        # Get all PNG files
        png_files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith('.png')])
        
        if not png_files:
            print(f"  ⚠️  No PNG files found in {in_dir}")
            total_skipped += 1
            continue
        
        print(f"  Found {len(png_files)} images to process")
        
        # Process each image
        for idx, fn in enumerate(png_files):
            img_path = in_dir / fn
            
            # Extract image index from filename (e.g., img_002.png -> 2)
            try:
                img_num = int(fn.split('_')[1].split('.')[0])
            except:
                print(f"  ⚠️  Warning: Could not parse image index from {fn}, skipping")
                continue
            
            # Check if output already exists
            out_path = out_dir / fn
            if out_path.exists():
                if (idx + 1) % 50 == 0:
                    print(f"    Progress: {idx+1}/{len(png_files)} (skipping existing)")
                continue
                
            # 1) Load & preprocess the image
            try:
                zim = Image.open(img_path).convert('RGB')
                zim = regularize_image(zim)  # [3, 512, 512], float in [0, 1]
                zin = (zim * 2.0 - 1.0).unsqueeze(0).to(device).half()
            except Exception as e:
                print(f"  ❌ Error loading image {fn}: {e}")
                continue
                
            # 2) Encode to autoKL latent
            with torch.no_grad():
                init_latent = net.autokl_encode(zin)
            
            # 3) Prepare DDIM schedule & stochastic encode
            sampler.make_schedule(
                ddim_num_steps=ddim_steps,
                ddim_eta=ddim_eta,
                verbose=False
            )
            t_enc = int(strength * ddim_steps)
            z_enc = sampler.stochastic_encode(
                init_latent,
                torch.tensor([t_enc]).to(device)
            )
            
            # 4) Prepare empty (unconditional) embeddings
            dummy_text = ""
            utx = net.clip_encode_text(dummy_text).to(device).half()
            
            dummy_image = torch.zeros((1, 3, 224, 224), device=device)
            uim = net.clip_encode_vision(dummy_image).to(device).half()
            
            # 5) Get predicted CLIP latents for this image
            if img_num >= len(pred_vision_all):
                print(f"  ⚠️  Warning: Image index {img_num} out of range, skipping")
                continue
            
            cim = pred_vision_all[img_num].unsqueeze(0).to(device)  # [1, D_vision]
            ctx = pred_text_all[img_num].unsqueeze(0).to(device)  # [1, D_text]
            
            # 6) Set device for diffusion model
            sampler.model.model.diffusion_model.device = device
            sampler.model.model.diffusion_model.half().to(device)
            
            # 7) Perform double-conditioning decode with structure preservation
            # Key: Lower strength + CLIP guidance changes semantics while preserving structure
            with torch.no_grad():
                try:
                    z = sampler.decode_dc(
                        x_latent=z_enc,
                        first_conditioning=[uim, cim],      # Vision conditioning
                        second_conditioning=[utx, ctx],      # Text conditioning
                        t_start=t_enc,
                        unconditional_guidance_scale=scale,
                        xtype='image',
                        first_ctype='vision',
                        second_ctype='prompt',
                        mixed_ratio=(1 - mixing),  # Balance between vision/text
                    )
                except Exception as e:
                    print(f"  ❌ Error during diffusion decode for {fn}: {e}")
                    continue
            
            # 8) Decode back to pixel space
            z = z.to(device).half()
            x = net.autokl_decode(z)
            
            # 9) Clamp and convert to PIL
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            pil_image = tvtrans.ToPILImage()(x[0])
            
            # 10) Save
            pil_image.save(out_path)
            total_processed += 1
            
            # Progress logging
            if (idx + 1) % 50 == 0 or (idx + 1) == len(png_files):
                print(f"    Progress: {idx+1}/{len(png_files)} images")
        
        print(f"  ✓ Completed alpha={alpha} ({len(png_files)} images)")
    
    print(f"\n✓ Completed assessor: {assessor_name.upper()}")

# --- Final summary ------------------------------------------------------------
print(f"\n{'='*70}")
print(f"PROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Subject: {sub:02d}")
print(f"Total images processed: {total_processed}")
print(f"Total images skipped: {total_skipped}")
print(f"\nOutputs saved to:")
print(f"  {base_output_dir / f'subj{sub:02d}'}")
print(f"\nStructure preservation settings:")
print(f"  Diffusion strength: {strength:.3f}")
print(f"  Mixing ratio: {mixing:.3f}")
print(f"  (Lower strength = better structure preservation)")
print(f"{'='*70}\n")

# --- Create comparison visualization (optional) ------------------------------
print(f"{'='*70}")
print("CREATING COMPARISON VISUALIZATION")
print(f"{'='*70}\n")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

for assessor in assessors:
    print(f"Generating comparison for {assessor}...")
    
    # Compare original semantic_heavy reconstruction vs VD output for image 2
    img_idx = 2
    n_alphas = len(alphas)
    
    # Create figure: 2 rows (original + VD) × alphas columns
    fig = plt.figure(figsize=(3*n_alphas, 8))
    gs = gridspec.GridSpec(2, n_alphas, figure=fig, hspace=0.2, wspace=0.05)
    
    for col_idx, alpha in enumerate(alphas):
        # Original VDVAE reconstruction (top row)
        ax_orig = fig.add_subplot(gs[0, col_idx])
        orig_path = (base_input_dir / f'subj{sub:02d}' / assessor / 
                    VARIANT / f'alpha_{alpha}' / f'img_{img_idx:03d}.png')
        
        if orig_path.exists():
            img_orig = plt.imread(orig_path)
            ax_orig.imshow(img_orig)
        else:
            ax_orig.text(0.5, 0.5, 'Not Found', ha='center', va='center')
        
        ax_orig.axis('off')
        if col_idx == 0:
            ax_orig.set_ylabel('VDVAE\nSemantic Heavy', fontsize=11, fontweight='bold', 
                              rotation=0, labelpad=70, va='center')
        ax_orig.set_title(f'α = {alpha}', fontsize=12, fontweight='bold')
        
        # Versatile Diffusion output (bottom row)
        ax_vd = fig.add_subplot(gs[1, col_idx])
        vd_path = (base_output_dir / f'subj{sub:02d}' / assessor / 
                  VARIANT / f'alpha_{alpha}' / f'img_{img_idx:03d}.png')
        
        if vd_path.exists():
            img_vd = plt.imread(vd_path)
            ax_vd.imshow(img_vd)
        else:
            ax_vd.text(0.5, 0.5, 'Not Found', ha='center', va='center')
        
        ax_vd.axis('off')
        if col_idx == 0:
            ax_vd.set_ylabel('Versatile\nDiffusion', fontsize=11, fontweight='bold',
                            rotation=0, labelpad=70, va='center')
    
    # Overall title
    assessor_name = 'EmoNet (Valence)' if assessor == 'emonet' else 'MemNet (Memorability)'
    title_text = (f'{assessor_name} - Semantic Heavy + VD Quality Enhancement\n'
                 f'Subject {sub:02d}, Image {img_idx}, Variant: {VARIANT}\n'
                 f'Diffusion Strength: {strength:.2f} (Structure Preservation)')
    fig.suptitle(title_text, fontsize=13, fontweight='bold', y=0.98)
    
    # Add info box
    info_text = (f'Settings: strength={strength:.2f}, mixing={mixing:.2f}, steps={ddim_steps}\n'
                f'Semantic heavy theta (1.0, 0.3, 0.05) + VD refinement for best quality\n'
                f'Alpha range: {min(alphas)} to {max(alphas)} | Lower strength preserves structure')
    fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                     edgecolor='black', linewidth=1, alpha=0.8))
    
    # Save
    fig_path = (base_output_dir / f'subj{sub:02d}' / assessor / 
               f'comparison_{VARIANT}_img{img_idx}_str{strength:.2f}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    print(f"  ✓ Saved comparison: {fig_path.name}")
    plt.close()

print(f"\n{'='*70}")
print("✓ All comparisons generated!")
print(f"{'='*70}\n")
