#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import re

# ---------- Utilities ----------
def natural_sort_key(s):
    """Natural sorting: splits strings into digit/non-digit chunks."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def gather_all_pngs(root_dir):
    """
    Recursively collect all .png files under root_dir (used for alpha_* layouts).
    Returns a naturally sorted list of absolute paths (sorted by relpath to root).
    """
    pngs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.png'):
                pngs.append(os.path.join(dirpath, fn))
    if not pngs:
        return []
    rels = [os.path.relpath(p, root_dir) for p in pngs]
    sorted_pairs = sorted(zip(rels, pngs), key=lambda kv: natural_sort_key(kv[0]))
    return [p for _, p in sorted_pairs]

def gather_pngs_flat(dir_path):
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

def load_scores_grouped_by_alpha(image_paths, assessor, mean=None, is_memnet=False):
    """
    For alpha_* tree:
      - Load and preprocess each image, run assessor
      - Group scores by the parent folder name (alpha_-4, alpha_0, etc.)
    Returns dict: {alpha_name: [scores...]}
    """
    transform_emo = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    transform_mem = T.Compose([
        T.Resize((256, 256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x)),
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean)),  # RGB→BGR, mean-sub
        T.Lambda(lambda x: x[15:242, 15:242]),                       # center-crop 227x227
        T.ToTensor()
    ])
    grouped = {}
    for p in image_paths:
        alpha = os.path.basename(os.path.dirname(p))
        img = Image.open(p).convert('RGB')
        transform = transform_mem if is_memnet else transform_emo
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            score = assessor(tensor).detach().cpu().numpy()[0][0]
        grouped.setdefault(alpha, []).append(float(score))
    return grouped

def load_scores_flat(image_paths, assessor, mean=None, is_memnet=False):
    """
    For flat directories (no alpha folders), returns a single list of scores.
    """
    transform_emo = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    transform_mem = T.Compose([
        T.Resize((256, 256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x)),
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean)),
        T.Lambda(lambda x: x[15:242, 15:242]),
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
        description="Compute + save Emonet/MemNet scores for VDVAE & Versatile Diffusion images (by alpha), and add image_to_image baselines."
    )
    parser.add_argument("-sub", "--sub", type=int, choices=[1, 2, 5, 7], required=True,
                        help="Subject number")
    args = parser.parse_args()
    sub = args.sub

    BASE_DIR = "/home/rothermm/brain-diffuser"
    ASSESSOR_DIR = os.path.join(BASE_DIR, "assessors")
    RESULTS_VDVAE_SUBJ = os.path.join(BASE_DIR, "results", "vdvae", f"subj{sub:02d}")
    RESULTS_VERS_SUBJ = os.path.join(BASE_DIR, "results", "versatile_diffusion", f"subj{sub:02d}")
    OUT_DIR = os.path.join(BASE_DIR, "results", "assessor_scores", f"subj{sub:02d}")

    # New, subject-agnostic i2i sets:
    VERS_I2I_DIR = os.path.join(BASE_DIR, "results", "versatile_diffusion", "image_to_image")
    VDVAE_I2I_DIR = os.path.join(BASE_DIR, "results", "vdvae", "image_to_image")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load assessors
    sys.path.append(ASSESSOR_DIR)
    import emonet
    from memnet import MemNet

    print("Loading assessors...")
    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to("cpu")

    mean = np.load(os.path.join(ASSESSOR_DIR, "image_mean.npy"))
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")

    # ---------------- VDVAE ----------------
    print("Processing VDVAE + Emonet...")
    vd_emonet_dir = os.path.join(RESULTS_VDVAE_SUBJ, "emonet")
    vd_img_paths = gather_all_pngs(vd_emonet_dir)
    emo_scores_vdvae = load_scores_grouped_by_alpha(vd_img_paths, assessor_emo)

    # Add VDVAE image_to_image
    vd_i2i_paths = gather_pngs_flat(VDVAE_I2I_DIR)
    if vd_i2i_paths:
        print(f"Adding VDVAE image_to_image ({len(vd_i2i_paths)} imgs) to emonet_vdvae...")
        emo_scores_vdvae["vdvae_image_to_image"] = load_scores_flat(vd_i2i_paths, assessor_emo)

    out_path = os.path.join(OUT_DIR, f"emonet_vdvae_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(emo_scores_vdvae, f)
    print(f"Saved: {out_path}")

    print("Processing VDVAE + MemNet...")
    vd_memnet_dir = os.path.join(RESULTS_VDVAE_SUBJ, "memnet")
    vd_img_paths = gather_all_pngs(vd_memnet_dir)
    mem_scores_vdvae = load_scores_grouped_by_alpha(vd_img_paths, assessor_mem, mean, is_memnet=True)

    # Add VDVAE image_to_image
    if vd_i2i_paths:
        print(f"Adding VDVAE image_to_image ({len(vd_i2i_paths)} imgs) to memnet_vdvae...")
        mem_scores_vdvae["vdvae_image_to_image"] = load_scores_flat(vd_i2i_paths, assessor_mem, mean, is_memnet=True)

    out_path = os.path.join(OUT_DIR, f"memnet_vdvae_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mem_scores_vdvae, f)
    print(f"Saved: {out_path}")

    # ---------------- Versatile Diffusion ----------------
    print("Processing Versatile Diffusion + Emonet...")
    vs_emonet_dir = os.path.join(RESULTS_VERS_SUBJ, "emonet")
    vs_img_paths = gather_all_pngs(vs_emonet_dir)
    emo_scores_vers = load_scores_grouped_by_alpha(vs_img_paths, assessor_emo)

    # Add Versatile image_to_image
    vers_i2i_paths = gather_pngs_flat(VERS_I2I_DIR)
    if vers_i2i_paths:
        print(f"Adding clip_image_to_image ({len(vers_i2i_paths)} imgs) to emonet_versatile...")
        emo_scores_vers["clip_image_to_image"] = load_scores_flat(vers_i2i_paths, assessor_emo)

    out_path = os.path.join(OUT_DIR, f"emonet_versatile_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(emo_scores_vers, f)
    print(f"Saved: {out_path}")

    print("Processing Versatile Diffusion + MemNet...")
    vs_memnet_dir = os.path.join(RESULTS_VERS_SUBJ, "memnet")
    vs_img_paths = gather_all_pngs(vs_memnet_dir)
    mem_scores_vers = load_scores_grouped_by_alpha(vs_img_paths, assessor_mem, mean, is_memnet=True)

    # Add Versatile image_to_image
    if vers_i2i_paths:
        print(f"Adding clip_image_to_image ({len(vers_i2i_paths)} imgs) to memnet_versatile...")
        mem_scores_vers["clip_image_to_image"] = load_scores_flat(vers_i2i_paths, assessor_mem, mean, is_memnet=True)

    out_path = os.path.join(OUT_DIR, f"memnet_versatile_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mem_scores_vers, f)
    print(f"Saved: {out_path}")

    print("✅ All done for subject", sub)

if __name__ == "__main__":
    main()
