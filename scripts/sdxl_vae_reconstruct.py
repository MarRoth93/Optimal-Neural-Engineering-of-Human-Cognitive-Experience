#!/usr/bin/env python3
"""Reconstruct images from SDXL-VAE latents.

This mirrors the VDVAE reconstruction utility but decodes latents with the
Stable Diffusion XL VAE.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode SDXL-VAE latents into images")
    parser.add_argument("--subject", "-sub", type=int, default=1, choices=[1, 2, 5, 7],
                        help="NSD subject identifier")
    parser.add_argument("--batch-size", "-bs", type=int, default=16,
                        help="Number of latents to decode per batch")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("BRAIN_DIFFUSER_DATA", "/home/rothermm/brain-diffuser/data")),
                        help="Root directory containing predicted_features/subjXX arrays")
    parser.add_argument("--extracted-root", type=Path, default=None,
                        help="Root directory where SDXL VAE reference metadata was stored")
    parser.add_argument("--predicted-latents", type=Path, default=None,
                        help="Path to flattened predicted latents (.npy or .npz)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory where decoded PNGs will be written")
    parser.add_argument("--vae-repo-id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace repo id providing the SDXL VAE weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device to run the VAE on")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32",
                        help="Computation precision for the VAE forward pass")
    parser.add_argument("--save-size", type=int, default=None,
                        help="Optional override for output image size (square)")
    return parser.parse_args()


def load_vae(repo_id: str, device: str, precision: str) -> AutoencoderKL:
    kwargs = {}
    if precision == "fp16" and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", **kwargs)
    vae.to(device)
    vae.eval()
    return vae


def load_predicted_latents(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Predicted latents not found: {path}")
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        for key in ("pred_latents", "latents", "arr_0"):
            if key in data:
                arr = data[key]
                break
        else:
            raise KeyError(f"No compatible array key found in {path}")
    else:
        arr = np.asarray(data)
    return arr


def reshape_latents(flat_latents: np.ndarray, latent_shape: Tuple[int, int, int]) -> np.ndarray:
    expected = int(np.prod(latent_shape))
    if flat_latents.ndim == 2:
        if flat_latents.shape[1] != expected:
            raise ValueError(f"Predicted latent width {flat_latents.shape[1]} does not match expected {expected}")
        return flat_latents.reshape((-1, *latent_shape))
    if flat_latents.ndim == 4:
        if tuple(flat_latents.shape[1:]) != latent_shape:
            raise ValueError(f"Predicted latent shape {tuple(flat_latents.shape[1:])} does not match expected {latent_shape}")
        return flat_latents
    raise ValueError(f"Unsupported latent array shape {flat_latents.shape}; expected (n,{expected}) or (n,{latent_shape})")


def save_batch_images(images: torch.Tensor, output_dir: Path, start_index: int, save_size: int) -> None:
    images = images.clamp(0, 1)
    images = images.mul(255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    for offset, array in enumerate(images):
        img = Image.fromarray(array)
        if save_size:
            img = img.resize((save_size, save_size), Image.BICUBIC)
        img.save(output_dir / f"{start_index + offset:05d}.png")


def main() -> None:
    args = parse_args()

    feature_root = args.extracted_root or args.data_root
    subj_dir = Path(feature_root) / "extracted_features" / f"subj{args.subject:02d}"
    ref_path = subj_dir / "sdxl_vae_ref_latents.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {ref_path}")

    ref = np.load(ref_path)
    ref_files = set(ref.files)
    latent_shape = tuple(ref["latent_shape"].tolist())  # type: ignore[arg-type]
    scaling_factor = float(ref["scaling_factor"]) if "scaling_factor" in ref_files else 1.0
    if "vae_repo_id" in ref_files:
        vae_repo = str(ref["vae_repo_id"]).strip()
    else:
        vae_repo = args.vae_repo_id
    if "image_size" in ref_files:
        default_size = int(ref["image_size"])  # type: ignore[arg-type]
    else:
        default_size = 1024

    pred_path = Path(args.predicted_latents) if args.predicted_latents else Path(args.data_root) / "predicted_features" / f"subj{args.subject:02d}" / f"nsd_sdxl_vae_pred_sub{args.subject}.npy"
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_root) / "results" / "sdxl_vae" / f"subj{args.subject:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SDXL VAE...")
    vae = load_vae(vae_repo, args.device, args.precision)
    dtype = next(vae.parameters()).dtype

    print(f"Reading predicted latents from {pred_path}")
    flat_latents = load_predicted_latents(pred_path)
    latents = reshape_latents(flat_latents, latent_shape)

    total = latents.shape[0]
    batch_size = args.batch_size
    save_size = args.save_size or default_size

    print(f"Decoding {total} samples in batches of {batch_size}")

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = latents[start:end]
            batch_tensor = torch.from_numpy(batch).to(device=args.device, dtype=dtype)
            if scaling_factor:
                batch_tensor = batch_tensor / scaling_factor
            decoded = vae.decode(batch_tensor).sample
            decoded = (decoded / 2.0) + 0.5
            save_batch_images(decoded, output_dir, start, save_size)
            print(f"Decoded samples {start}-{end - 1}")

    print(f"Saved reconstructions to {output_dir}")


if __name__ == "__main__":
    main()
