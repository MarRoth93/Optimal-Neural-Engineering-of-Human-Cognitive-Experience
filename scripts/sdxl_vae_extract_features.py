#!/usr/bin/env python3
"""Extract SDXL-VAE latents for NSD stimuli.

Mirrors the VDVAE feature extraction pipeline while replacing the encoder with the
Stable Diffusion XL VAE from HuggingFace diffusers.
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SDXL-VAE latents from NSD stimuli")
    parser.add_argument("--subject", "-sub", type=int, default=1, choices=[1, 2, 5, 7],
                        help="NSD subject identifier")
    parser.add_argument("--batch-size", "-bs", type=int, default=16,
                        help="Number of images to encode per batch")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("BRAIN_DIFFUSER_DATA", "/home/rothermm/brain-diffuser/data")),
                        help="Root directory containing processed_data/subjXX arrays")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Base directory where extracted features will be stored. Defaults to data-root")
    parser.add_argument("--vae-repo-id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace repo id providing the SDXL VAE weights")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Image resolution (square) expected by the SDXL VAE")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device to run the VAE on")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32",
                        help="Computation precision for the VAE forward pass")
    return parser.parse_args()


class NpyImageDataset(Dataset):
    def __init__(self, npy_path: Path, image_size: int):
        if not npy_path.exists():
            raise FileNotFoundError(f"Cannot find stimulus file: {npy_path}")
        self.data = np.load(npy_path)
        if self.data.ndim != 4 or self.data.shape[-1] != 3:
            raise ValueError(f"Expected NHWC RGB array in {npy_path}, got {self.data.shape}")
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.data[idx]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        return self.transform(pil)


def load_vae(repo_id: str, device: str, precision: str) -> AutoencoderKL:
    kwargs = {}
    if precision == "fp16" and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", **kwargs)
    vae.to(device)
    vae.eval()
    return vae


def encode_dataset(vae: AutoencoderKL, loader: DataLoader, device: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    latents = []
    latent_shape: Optional[Tuple[int, int, int]] = None
    scaling = vae.config.scaling_factor
    dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pixel_values = batch.to(device=device, dtype=dtype)
            pixel_values = pixel_values * 2.0 - 1.0
            latent_dist = vae.encode(pixel_values).latent_dist
            latent_mean = latent_dist.mean * scaling
            if latent_shape is None:
                latent_shape = tuple(latent_mean.shape[1:])
            latents.append(latent_mean.cpu().numpy().reshape(len(pixel_values), -1))
            print(f"Encoded batch {batch_idx + 1}/{len(loader)}")
    if not latents:
        raise RuntimeError("No latents were produced; check the dataset inputs")
    return np.concatenate(latents, axis=0), latent_shape  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()

    output_root = args.output_root or args.data_root
    subj_dir = Path(output_root) / "extracted_features" / f"subj{args.subject:02d}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.data_root) / "processed_data" / f"subj{args.subject:02d}" / f"nsd_train_stim_sub{args.subject}.npy"
    test_path = Path(args.data_root) / "processed_data" / f"subj{args.subject:02d}" / f"nsd_test_stim_sub{args.subject}.npy"

    print("Loading SDXL VAE...")
    vae = load_vae(args.vae_repo_id, args.device, args.precision)

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=args.device.startswith("cuda"))

    print(f"Encoding training stimuli from {train_path}")
    train_dataset = NpyImageDataset(train_path, args.image_size)
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    train_latents, latent_shape = encode_dataset(vae, train_loader, args.device)

    print(f"Encoding test stimuli from {test_path}")
    test_dataset = NpyImageDataset(test_path, args.image_size)
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    test_latents, _ = encode_dataset(vae, test_loader, args.device)

    feature_path = subj_dir / "nsd_sdxl_vae_features.npz"
    np.savez(feature_path, train_latents=train_latents, test_latents=test_latents)

    ref_path = subj_dir / "sdxl_vae_ref_latents.npz"
    np.savez(
        ref_path,
        latent_shape=np.array(latent_shape, dtype=np.int32),
        scaling_factor=np.float32(vae.config.scaling_factor),
        vae_repo_id=args.vae_repo_id,
        image_size=np.int32(args.image_size),
        dtype=str(next(vae.parameters()).dtype),
    )

    print(f"Saved SDXL VAE features to {feature_path}")
    print(f"Saved reference metadata to {ref_path}")


if __name__ == "__main__":
    main()
