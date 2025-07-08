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


def natural_sort_key(s):
    """
    Natural sorting: splits strings into digit/non-digit chunks.
    E.g., image2.png before image10.png
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def gather_all_pngs(root_dir):
    """
    Recursively collect all .png files under root_dir, including alpha_* subfolders.
    Returns a sorted list of absolute paths.
    """
    pngs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.png'):
                fullp = os.path.join(dirpath, fn)
                pngs.append(fullp)
    # Sort relative to root_dir naturally
    rels = [os.path.relpath(p, root_dir) for p in pngs]
    sorted_pairs = sorted(zip(rels, pngs), key=lambda kv: natural_sort_key(kv[0]))
    return [p for _, p in sorted_pairs]


def load_scores_grouped_by_alpha(image_paths, assessor, mean=None, is_memnet=False):
    """
    For each image:
    - Load and preprocess
    - Run through assessor
    - Group scores by alpha folder
    Returns: dict {alpha_name: [scores]}
    """
    transform_emo = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    transform_mem = T.Compose([
        T.Resize((256, 256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x)),
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean)),  # RGB → BGR and subtract mean
        T.Lambda(lambda x: x[15:242, 15:242]),  # Center crop to 227x227
        T.ToTensor()
    ])

    grouped_scores = {}
    for p in image_paths:
        alpha = os.path.basename(os.path.dirname(p))  # e.g., 'alpha_0'
        img = Image.open(p).convert('RGB')
        transform = transform_mem if is_memnet else transform_emo
        tensor = transform(img).unsqueeze(0)  # shape [1, 3, H, W]
        with torch.no_grad():
            score = assessor(tensor).detach().cpu().numpy()[0][0]
        grouped_scores.setdefault(alpha, []).append(float(score))
    return grouped_scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute + save Emonet/MemNet scores for VDVAE & Versatile Diffusion images grouped by alpha"
    )
    parser.add_argument(
        "-sub", "--sub",
        type=int,
        choices=[1, 2, 5, 7],
        required=True,
        help="Subject number"
    )
    args = parser.parse_args()
    sub = args.sub

    BASE_DIR = "/home/rothermm/brain-diffuser"
    ASSESSOR_DIR = os.path.join(BASE_DIR, "assessors")
    RESULTS_VDVAE = os.path.join(BASE_DIR, "results", "vdvae", f"subj{sub:02d}")
    RESULTS_VERS = os.path.join(BASE_DIR, "results", "versatile_diffusion", f"subj{sub:02d}")
    OUT_DIR = os.path.join(BASE_DIR, "results", "assessor_scores", f"subj{sub:02d}")
    os.makedirs(OUT_DIR, exist_ok=True)

    sys.path.append(ASSESSOR_DIR)
    import emonet
    from memnet import MemNet

    # Load models
    print("Loading assessors...")
    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to("cpu")

    mean = np.load(os.path.join(ASSESSOR_DIR, "image_mean.npy"))
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")

    # ---------------------------------------------------------------
    # 1) VDVAE + Emonet
    print("Processing VDVAE + Emonet...")
    vd_emonet_dir = os.path.join(RESULTS_VDVAE, "emonet")
    img_paths = gather_all_pngs(vd_emonet_dir)
    emo_scores_vdvae = load_scores_grouped_by_alpha(img_paths, assessor_emo)
    out_path = os.path.join(OUT_DIR, f"emonet_vdvae_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(emo_scores_vdvae, f)
    print(f"Saved: {out_path}")

    # 2) VDVAE + MemNet
    print("Processing VDVAE + MemNet...")
    vd_memnet_dir = os.path.join(RESULTS_VDVAE, "memnet")
    img_paths = gather_all_pngs(vd_memnet_dir)
    mem_scores_vdvae = load_scores_grouped_by_alpha(img_paths, assessor_mem, mean, is_memnet=True)
    out_path = os.path.join(OUT_DIR, f"memnet_vdvae_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mem_scores_vdvae, f)
    print(f"Saved: {out_path}")

    # 3) Versatile Diffusion + Emonet
    print("Processing Versatile Diffusion + Emonet...")
    vs_emonet_dir = os.path.join(RESULTS_VERS, "emonet")
    img_paths = gather_all_pngs(vs_emonet_dir)
    emo_scores_vers = load_scores_grouped_by_alpha(img_paths, assessor_emo)
    out_path = os.path.join(OUT_DIR, f"emonet_versatile_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(emo_scores_vers, f)
    print(f"Saved: {out_path}")

    # 4) Versatile Diffusion + MemNet
    print("Processing Versatile Diffusion + MemNet...")
    vs_memnet_dir = os.path.join(RESULTS_VERS, "memnet")
    img_paths = gather_all_pngs(vs_memnet_dir)
    mem_scores_vers = load_scores_grouped_by_alpha(img_paths, assessor_mem, mean, is_memnet=True)
    out_path = os.path.join(OUT_DIR, f"memnet_versatile_sub{sub:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(mem_scores_vers, f)
    print(f"Saved: {out_path}")

    print("✅ All done for subject", sub)


if __name__ == "__main__":
    main()
