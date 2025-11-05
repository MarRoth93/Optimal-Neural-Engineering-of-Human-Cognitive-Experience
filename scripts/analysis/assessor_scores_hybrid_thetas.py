#!/usr/bin/env python3
"""
Compute assessor scores for all hybrid theta reconstructions.

This script processes all reconstructed images from reconstruct_semantic_heavy_all.py
and computes assessor scores using the MATCHING assessor (Emonet scores for Emonet-theta
images, MemNet scores for MemNet-theta images).

Organized by:
  - Subject (1, 2, 5, 7)
  - Assessor used for theta (emonet, memnet)
  - Variant (original, semantic_heavy, semantic_only, balanced, structural_heavy, structural_only)
  - Alpha value (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5)

The scores are stored in pickle files with the structure:
{
  'variant_name': {
    'alpha_value': [scores...]
  }
}

Output: results/assessor_scores/hybrid_theta/subj{XX}/{assessor}_scores.pkl
        (e.g., emonet_scores.pkl, memnet_scores.pkl)
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import re
from pathlib import Path

# ---------- Utilities ----------
def natural_sort_key(s):
    """Natural sorting: splits strings into digit/non-digit chunks."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def gather_pngs_from_dir(dir_path):
    """
    Collect .png files directly in dir_path (non-recursive).
    Returns a naturally sorted list of absolute paths.
    """
    if not os.path.isdir(dir_path):
        return []
    pngs = [os.path.join(dir_path, fn)
            for fn in os.listdir(dir_path)
            if fn.lower().endswith('.png')]
    pngs.sort(key=natural_sort_key)
    return pngs

def load_scores_for_images(image_paths, assessor, mean=None, is_memnet=False):
    """
    Load images, preprocess, and compute assessor scores.
    Returns a list of scores (one per image).
    """
    transform_emo = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    transform_mem = T.Compose([
        T.Resize((256, 256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x)),
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean)),  # RGB→BGR, mean-sub
        T.Lambda(lambda x: x[15:242, 15:242]),                       # center-crop 227x227
        T.ToTensor()
    ])
    
    scores = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        transform = transform_mem if is_memnet else transform_emo
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            s = assessor(tensor).detach().cpu().numpy()[0][0]
        scores.append(float(s))
    return scores

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Compute Emonet/MemNet scores for all hybrid theta reconstructions"
    )
    parser.add_argument("-sub", "--sub", type=int, choices=[1, 2, 5, 7], required=True,
                        help="Subject number")
    args = parser.parse_args()
    sub = args.sub

    BASE_DIR = Path("/home/rothermm/brain-diffuser")
    ASSESSOR_DIR = BASE_DIR / "assessors"
    RECON_DIR = BASE_DIR / "results" / "hybrid_theta_reconstructions"
    OUT_DIR = BASE_DIR / "results" / "assessor_scores" / "hybrid_theta"

    # Input directory for this subject
    RECON_SUBJ_DIR = RECON_DIR / f"subj{sub:02d}"
    # Output directory for this subject
    OUT_SUBJ_DIR = OUT_DIR / f"subj{sub:02d}"
    OUT_SUBJ_DIR.mkdir(parents=True, exist_ok=True)

    # Check if reconstruction directory exists
    if not RECON_SUBJ_DIR.exists():
        print(f"❌ ERROR: Reconstruction directory not found: {RECON_SUBJ_DIR}")
        print(f"   Please run reconstruct_semantic_heavy_all.py first.")
        sys.exit(1)

    # Load assessors
    sys.path.append(str(ASSESSOR_DIR))
    import emonet
    from memnet import MemNet

    print("="*70)
    print(f"Computing assessor scores for hybrid theta reconstructions")
    print(f"Subject: {sub:02d}")
    print("="*70)
    print("\nLoading assessors...")
    
    # Load Emonet
    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to("cpu")
    print("✓ Emonet loaded")

    # Load MemNet
    mean = np.load(ASSESSOR_DIR / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")
    print("✓ MemNet loaded")

    # Define theta assessors and variants
    THETA_ASSESSORS = ['emonet', 'memnet']
    VARIANTS = ['original', 'semantic_heavy', 'semantic_only', 'balanced', 'structural_heavy', 'structural_only']
    ALPHAS = ['-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5']
    
    # Map theta assessor to the corresponding scoring assessor
    ASSESSOR_MAP = {
        'emonet': (assessor_emo, False, None),
        'memnet': (assessor_mem, True, mean)
    }

    print(f"\nConfiguration:")
    print(f"  Theta assessors: {THETA_ASSESSORS}")
    print(f"  Variants: {len(VARIANTS)}")
    print(f"  Alpha values: {len(ALPHAS)}")
    print(f"  Scoring: Each theta with its matching assessor")
    print(f"  Input dir: {RECON_SUBJ_DIR}")
    print(f"  Output dir: {OUT_SUBJ_DIR}")
    print()

    total_conditions = len(THETA_ASSESSORS) * len(VARIANTS) * len(ALPHAS)
    condition_count = 0

    # Process each theta assessor
    for theta_assessor in THETA_ASSESSORS:
        print(f"\n{'='*70}")
        print(f"Processing theta from: {theta_assessor.upper()}")
        print(f"Scoring with: {theta_assessor.upper()}")
        print(f"{'='*70}")
        
        theta_dir = RECON_SUBJ_DIR / theta_assessor
        if not theta_dir.exists():
            print(f"⚠️  Warning: Directory not found, skipping: {theta_dir}")
            continue

        # Get the matching scoring assessor
        scoring_assessor, is_memnet, mean_arr = ASSESSOR_MAP[theta_assessor]
        
        # Dictionary to hold all scores for this theta assessor
        all_scores = {}
        
        # Process each variant
        for variant in VARIANTS:
            variant_dir = theta_dir / variant
            if not variant_dir.exists():
                print(f"  ⚠️  Variant not found, skipping: {variant}")
                continue
            
            print(f"  Variant: {variant}")
            variant_scores = {}
            
            # Process each alpha
            for alpha in ALPHAS:
                alpha_dir = variant_dir / f"alpha_{alpha}"
                if not alpha_dir.exists():
                    print(f"    ⚠️  Alpha dir not found, skipping: alpha_{alpha}")
                    continue
                
                # Gather all images
                image_paths = gather_pngs_from_dir(alpha_dir)
                if not image_paths:
                    print(f"    ⚠️  No images found in: alpha_{alpha}")
                    continue
                
                # Compute scores
                scores = load_scores_for_images(image_paths, scoring_assessor, mean_arr, is_memnet)
                variant_scores[f"alpha_{alpha}"] = scores
                
                condition_count += 1
                print(f"    ✓ alpha_{alpha}: {len(scores)} scores computed ({condition_count}/{total_conditions})")
            
            # Store variant scores
            if variant_scores:
                all_scores[variant] = variant_scores
        
        # Save scores to pickle file
        if all_scores:
            out_path = OUT_SUBJ_DIR / f"{theta_assessor}_scores.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(all_scores, f)
            print(f"\n✓ Saved: {out_path}")
            print(f"  Variants: {len(all_scores)}")
            print(f"  Total alpha conditions: {sum(len(v) for v in all_scores.values())}")
        else:
            print(f"\n⚠️  No scores computed for {theta_assessor}")

    print(f"\n{'='*70}")
    print(f"✅ ALL DONE for subject {sub:02d}")
    print(f"{'='*70}")
    print(f"Output directory: {OUT_SUBJ_DIR}")
    print(f"Files saved:")
    for theta_assessor in THETA_ASSESSORS:
        pkl_file = OUT_SUBJ_DIR / f"{theta_assessor}_scores.pkl"
        if pkl_file.exists():
            print(f"  ✓ {pkl_file.name}")
    print()

if __name__ == "__main__":
    main()
