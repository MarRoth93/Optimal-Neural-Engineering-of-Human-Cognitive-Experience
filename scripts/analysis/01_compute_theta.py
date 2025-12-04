#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import numpy as np
import re
import torchvision.transforms as T
from PIL import Image

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def test_images_prep_assessor_CLIP(image_folder, target_size=(256, 256)):
    """Load all images in a folder, resize and to-tensor."""
    files = [f for f in os.listdir(image_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_rp' not in f]
    files.sort(key=natural_sort_key)
    tensors = []
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor()
    ])
    for fn in files:
        img = Image.open(os.path.join(image_folder, fn)).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors)

def test_images_prep_assessor_memnet(data_path, mean):
    """Load an .npy of HxWxC uint8 images, apply memnet preprocessing."""
    arr = np.load(data_path).astype(np.uint8)
    n = arr.shape[0]
    out = torch.empty(n, 3, 227, 227)
    for i in range(n):
        img = Image.fromarray(arr[i])
        transform = T.Compose([
            T.Resize((256,256), Image.BILINEAR),
            T.Lambda(lambda x: np.array(x)),
            T.Lambda(lambda x: x[:,:,[2,1,0]] - mean),
            T.Lambda(lambda x: x[15:242,15:242]),
            T.ToTensor()
        ])
        out[i] = transform(img)
    return out

def get_indexes(arr, per=0.15):
    top_n = int(len(arr) * per)
    sorted_idx = np.argsort(arr)
    top_idx = sorted_idx[-top_n:]
    bottom_idx = sorted_idx[:top_n]
    return top_idx, bottom_idx

def calculate_theta(mask_data, test_latents):
    top = mask_data['top_15_percent']['indexes']
    bot = mask_data['bottom_15_percent']['indexes']
    lat_top = test_latents[top].mean(axis=0)
    lat_bot = test_latents[bot].mean(axis=0)
    return lat_top - lat_bot

def main():
    parser = argparse.ArgumentParser(
        description='Compute Emonet + MemNet thetas for one subject'
    )
    parser.add_argument('-sub','--sub', type=int, choices=[1,2,5,7],
                        help='Subject number', required=True)
    args = parser.parse_args()
    sub = args.sub

    # --- Directories ---------------------------------------------------------
    BASE_DIR      = '/home/rothermm/brain-diffuser'
    ASSESSOR_DIR  = os.path.join(BASE_DIR, 'assessors')
    IMG_DIR       = os.path.join(BASE_DIR, 'results', 'vdvae')
    FEATURE_DIR   = os.path.join(BASE_DIR, 'data', 'extracted_features')
    THETA_DIR     = os.path.join(BASE_DIR, 'results', 'thetas', f'subj{sub:02d}')
    os.makedirs(THETA_DIR, exist_ok=True)
    os.chdir(BASE_DIR)

    # --- Import assessors ----------------------------------------------------
    sys.path.append(ASSESSOR_DIR)
    import emonet
    from memnet import MemNet

    # --- Emonet setup --------------------------------------------------------
    model, emo_input_transform, emo_output_transform = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to('cpu')

    # --- Run Emonet on generated images --------------------------------------
    img_folder = os.path.join(IMG_DIR, f'subj{sub:02d}')
    emonet_tensor = test_images_prep_assessor_CLIP(img_folder)
    emonet_input = emonet_tensor.view(-1, *emonet_tensor.shape[-3:])
    scores_emo = assessor_emo(emonet_input)
    emonet_res = [s.detach().cpu().numpy()[0] for s in scores_emo]

    # --- Load test latents and compute Emonet theta --------------------------
    feat_path = os.path.join(FEATURE_DIR, f'subj{sub:02d}', 'nsd_vdvae_features_31l.npz')
    data = np.load(feat_path)
    test_latents = data['test_latents']

    top_i, bot_i = get_indexes(emonet_res)
    indexed_emo = {
        'scores': emonet_res,
        'top_15_percent': {
            'indexes': top_i.tolist(),
            'values': [emonet_res[i] for i in top_i]
        },
        'bottom_15_percent': {
            'indexes': bot_i.tolist(),
            'values': [emonet_res[i] for i in bot_i]
        }
    }
    theta_emo = calculate_theta(indexed_emo, test_latents)
    out_emo = os.path.join(THETA_DIR, f'theta_emonet_nsdgeneral_subject{sub}.npy')
    np.save(out_emo, theta_emo)
    print(f"Saved Emonet theta to {out_emo}")

    # --- MemNet setup --------------------------------------------------------
    mean_path    = os.path.join(ASSESSOR_DIR, 'image_mean.npy')
    mean         = np.load(mean_path)
    assessor_mem = MemNet().eval().requires_grad_(False).to('cpu')

    # --- Prepare images for MemNet (like your original) ---------------------
    def test_images_prep_assessor_memnet(image_folder):
        data_tensor = []
        # list all png/jpg/jpeg
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg')) and '_rp' not in f
        ]
        image_files.sort(key=natural_sort_key)
        for fn in image_files:
            img = Image.open(os.path.join(image_folder, fn)).convert('RGB')
            transform = T.Compose([
                T.Resize((256,256), Image.BILINEAR),
                T.Lambda(lambda x: np.array(x)),
                T.Lambda(lambda x: np.subtract(x[:,:,[2,1,0]], mean)),
                T.Lambda(lambda x: x[15:242,15:242]),
                T.ToTensor()
            ])
            data_tensor.append(transform(img))
        return torch.stack(data_tensor)

    # --- Run MemNet on the VDVAE reconstructions ----------------------------
    img_folder  = os.path.join(IMG_DIR, f'subj{sub:02d}')
    memnet_tensor = test_images_prep_assessor_memnet(img_folder)
    mem_input     = memnet_tensor.view(-1, *memnet_tensor.shape[-3:])
    scores_mem    = assessor_mem(mem_input)
    memnet_res    = [s.detach().cpu().numpy()[0] for s in scores_mem]

    # --- Compute MemNet theta (top/bottom 15%) -----------------------------
    top_m, bot_m = get_indexes(memnet_res, per=0.15)
    indexed_mem = {
        'scores': memnet_res,
        'top_15_percent': {
            'indexes': top_m.tolist(),
            'values': [memnet_res[i] for i in top_m]
        },
        'bottom_15_percent': {
            'indexes': bot_m.tolist(),
            'values': [memnet_res[i] for i in bot_m]
        }
    }
    theta_mem = calculate_theta(indexed_mem, test_latents)

    # --- Save ----------------------------------------------------------------
    out_mem = os.path.join(
        THETA_DIR,
        f'theta_memnet_nsdgeneral_subject{sub}.npy'
    )
    np.save(out_mem, theta_mem)
    print(f"Saved MemNet theta to {out_mem}")


if __name__ == '__main__':
    main()
