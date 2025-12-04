#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# --- Paths ---
BASE_DIR = Path("/home/rothermm/brain-diffuser")
ASSESSOR_DIR = BASE_DIR / "assessors"
IMG_DIR = BASE_DIR / "data" / "original_images"
OUT_DIR = BASE_DIR / "results" / "assessor_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load assessors ---
def load_assessors():
    sys.path.append(str(ASSESSOR_DIR))
    import emonet
    from memnet import MemNet

    print("Loading EmoNet and MemNet…")
    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to("cpu")

    mean = np.load(ASSESSOR_DIR / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")

    return assessor_emo, assessor_mem, mean

# --- Preprocessing ---
def get_transforms(mean):
    transform_emo = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    transform_mem = T.Compose([
        T.Resize((256, 256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x)),                         # HWC
        T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mean)),  # RGB→BGR & mean-sub
        T.Lambda(lambda x: x[15:242, 15:242]),                   # center crop 227×227
        T.ToTensor(),
    ])
    return transform_emo, transform_mem

# --- Scoring ---
@torch.no_grad()
def score_images(img_paths, assessor, transform, is_memnet=False):
    scores = []
    for i, p in enumerate(sorted(img_paths)):
        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        s = assessor(tensor).detach().cpu().numpy()
        scores.append(float(s[0][0]))
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(img_paths)} images")
    return scores

# --- Main ---
def main():
    assessor_emo, assessor_mem, mean = load_assessors()
    transform_emo, transform_mem = get_transforms(mean)

    img_paths = sorted([p for p in IMG_DIR.glob("*.png")])
    if not img_paths:
        raise RuntimeError(f"No PNGs found in {IMG_DIR}")

    print(f"Found {len(img_paths)} images in {IMG_DIR}")

    # EmoNet
    print("Scoring with EmoNet…")
    emo_scores = score_images(img_paths, assessor_emo, transform_emo)
    with open(OUT_DIR / "emonet_original.pkl", "wb") as f:
        pickle.dump({"score": emo_scores}, f)
    print(f"✅ Saved: {OUT_DIR / 'emonet_original.pkl'} (n={len(emo_scores)})")

    # MemNet
    print("Scoring with MemNet…")
    mem_scores = score_images(img_paths, assessor_mem, transform_mem, is_memnet=True)
    with open(OUT_DIR / "memnet_original.pkl", "wb") as f:
        pickle.dump({"score": mem_scores}, f)
    print(f"✅ Saved: {OUT_DIR / 'memnet_original.pkl'} (n={len(mem_scores)})")

    print("All done.")

if __name__ == "__main__":
    main()
