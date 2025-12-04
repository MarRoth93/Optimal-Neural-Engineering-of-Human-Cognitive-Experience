#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/rothermm/brain-diffuser/vdvae')

import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, local_mpi_rank, mpi_size, maybe_download, mpi_rank
from data import mkdir_p
from vae import VAE
from train_helpers import restore_params
from image_utils import *
from model_utils import *

# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(description="VDVAE round-trip on TEST images (universal, image→latent→image)")
    p.add_argument("--images-npy", required=True,
                   help="Path to universal test images .npy (e.g., .../nsd_test_stim.npy)")
    p.add_argument("--bs", "--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--resize", type=int, default=512,
                   help="Output PNG size (square). Use 0 to keep native 64x64.")
    p.add_argument("--save-latents", action="store_true",
                   help="Save per-layer latents for all images")
    p.add_argument("--save-flattened", action="store_true",
                   help="Save flattened latents (layer-concatenated)")
    return p.parse_args()

# ---------- Dataset ----------
class NPYImageDataset(Dataset):
    def __init__(self, npy_path: str):
        self.arr = np.load(npy_path).astype(np.uint8)
        self.resize_to = (64, 64)  # VDVAE expected input size

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        img = Image.fromarray(self.arr[idx])
        img = T.functional.resize(img, self.resize_to)
        img = torch.tensor(np.array(img)).float()  # [H,W,C], float
        return img

# ---------- Model setup ----------
def build_hparams():
    model_dir = '/home/rothermm/brain-diffuser/vdvae/model'
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
        'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25,
        'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100,
        'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015,
        'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
        'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
        'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000,
        'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None,
        'epochs_per_eval_save': 1, 'num_images_visualize': 8,
        'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
        'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'
    }
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    H = dotdict(H)
    H, preprocess_fn = set_up_data(H)
    return H, preprocess_fn

def load_model(H):
    print("[VDVAE] Loading EMA model...")
    ema_vae = load_vaes(H)
    ema_vae.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ema_vae.to(device)
    except Exception:
        pass
    return ema_vae, device

# ---------- Main ----------
@torch.no_grad()
def main():
    args = get_args()
    H, preprocess_fn = build_hparams()
    ema_vae, device = load_model(H)

    batch_size = args.batch_size
    out_resize = (args.resize, args.resize) if args.resize and args.resize > 0 else None

    # Inputs (universal)
    npy_path = args.images_npy
    ds = NPYImageDataset(npy_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Output images (universal)
    out_img_dir = "/home/rothermm/brain-diffuser/results/vdvae/image_to_image"
    Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    # Optional latent outputs (universal, no subject)
    lat_dir = "/home/rothermm/brain-diffuser/data/extracted_features/vdvae_universal"
    Path(lat_dir).mkdir(parents=True, exist_ok=True)
    num_latents = 31

    layers_accum: List[List[np.ndarray]] = [ [] for _ in range(num_latents) ] if args.save_latents else None
    flattened_accum: List[np.ndarray] = [] if args.save_flattened else None

    n_total = len(ds)
    ctr = 0
    print(f"[UNIVERSAL TEST] {n_total} images → encoding+decoding to {out_img_dir}")

    torch.set_grad_enabled(False)
    for i, x in enumerate(loader):
        data_input, _ = preprocess_fn(x)  # [B,C,H,W] normalized as VDVAE expects
        data_input = data_input.to(device, non_blocking=True)

        # Encode → latents
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)

        # Decode → reconstructed images (uint8 HxWxC)
        recons = ema_vae.decoder.out_net.sample(px_z)

        # Save recon PNGs
        for k, img_arr in enumerate(recons):
            img = Image.fromarray(img_arr)
            if out_resize is not None:
                img = img.resize(out_resize, resample=Image.BICUBIC)
            img.save(os.path.join(out_img_dir, f"{ctr + k}.png"))
        ctr += recons.shape[0]

        # Optional: save latents
        if args.save_latents or args.save_flattened:
            if args.save_latents:
                for j in range(num_latents):
                    layers_accum[j].append(stats[j]['z'].detach().cpu().numpy())
            if args.save_flattened:
                flat_layers = []
                for j in range(num_latents):
                    z = stats[j]['z']  # (B,C,H,W)
                    flat_layers.append(z.reshape(z.shape[0], -1).detach().cpu().numpy())
                flattened_accum.append(np.hstack(flat_layers))

    print(f"[UNIVERSAL TEST] Saved reconstructions: {out_img_dir}")

    # Finalize latent saves
    if args.save_latents:
        ref_latent_full: List[Dict[str, np.ndarray]] = []
        for j in range(num_latents):
            z_full = np.concatenate(layers_accum[j], axis=0)  # (N, C, H, W)
            ref_latent_full.append({'z': z_full})
        np.savez(os.path.join(lat_dir, "ref_latents_universal_test.npz"), ref_latent=ref_latent_full)
        print(f"[UNIVERSAL TEST] Saved per-layer latents → {lat_dir}/ref_latents_universal_test.npz "
              f"(N={ref_latent_full[0]['z'].shape[0]})")

    if args.save_flattened:
        flat_all = np.concatenate(flattened_accum, axis=0)  # (N, sum_dims)
        np.savez(os.path.join(lat_dir, "nsd_vdvae_features_31l_universal_test.npz"),
                 test_latents=flat_all)
        print(f"[UNIVERSAL TEST] Saved flattened latents → "
              f"{lat_dir}/nsd_vdvae_features_31l_universal_test.npz (shape={flat_all.shape})")

    print("Done.")

if __name__ == "__main__":
    main()
