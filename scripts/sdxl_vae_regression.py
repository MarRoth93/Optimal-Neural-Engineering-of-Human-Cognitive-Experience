#!/usr/bin/env python3
"""Train ridge regression models mapping fMRI responses to SDXL-VAE latents.

This mirrors the original VDVAE ridge regression workflow but adapts the
regularisation strength to the higher-resolution SDXL latent space. The script
expects SDXL features produced by `sdxl_vae_extract_features.py`.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regress fMRI onto SDXL-VAE latents")
    parser.add_argument("--subject", "-sub", type=int, default=1, choices=[1, 2, 5, 7],
                        help="NSD subject identifier")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("BRAIN_DIFFUSER_DATA", "/home/rothermm/brain-diffuser/data")),
                        help="Root directory containing processed/extracted/predicted data")
    parser.add_argument("--alpha-min", type=float, default=1e4,
                        help="Smallest ridge penalty considered during CV")
    parser.add_argument("--alpha-max", type=float, default=1e7,
                        help="Largest ridge penalty considered during CV")
    parser.add_argument("--alpha-count", type=int, default=8,
                        help="Number of alpha values (log-spaced) to evaluate")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--fmri-scale", type=float, default=300.0,
                        help="Divisor applied to raw fMRI values before normalisation")
    parser.add_argument("--output-suffix", type=str, default="nsd_sdxl_vae_pred",
                        help="Filename stem for saved predicted latents")
    return parser.parse_args()


def load_latents(feature_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing SDXL feature file: {feature_path}")
    npz = np.load(feature_path)
    if not {"train_latents", "test_latents"}.issubset(npz.files):
        raise KeyError(f"Feature file {feature_path} lacks expected arrays")
    return npz["train_latents"], npz["test_latents"]


def load_fmri(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    train_path = data_root / "processed_data" / f"subj{subject:02d}" / f"nsd_train_fmriavg_nsdgeneral_sub{subject}.npy"
    test_path = data_root / "processed_data" / f"subj{subject:02d}" / f"nsd_test_fmriavg_nsdgeneral_sub{subject}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing fMRI arrays for subject {subject:02d}")
    return np.load(train_path), np.load(test_path)


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    feature_path = data_root / "extracted_features" / f"subj{args.subject:02d}" / "nsd_sdxl_vae_features.npz"
    train_latents, test_latents = load_latents(feature_path)

    train_fmri, test_fmri = load_fmri(data_root, args.subject)

    # Scale and standardise the fMRI signals to stabilise regression.
    fmri_scaler = StandardScaler()
    train_fmri_scaled = fmri_scaler.fit_transform(train_fmri / args.fmri_scale)
    test_fmri_scaled = fmri_scaler.transform(test_fmri / args.fmri_scale)

    # Standardise the latent targets so ridge regularisation operates per-dimension.
    latent_scaler = StandardScaler()
    train_latents_scaled = latent_scaler.fit_transform(train_latents)
    test_latents_scaled = latent_scaler.transform(test_latents)

    alphas = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), num=args.alpha_count)
    reg = RidgeCV(alphas=alphas, cv=args.cv_folds, fit_intercept=True, scoring="r2", store_cv_values=False)
    reg.fit(train_fmri_scaled, train_latents_scaled)

    best_alpha = float(reg.alpha_)
    print(f"Selected ridge alpha: {best_alpha:.2e}")

    pred_latents_scaled = reg.predict(test_fmri_scaled)
    pred_latents = latent_scaler.inverse_transform(pred_latents_scaled)

    r2 = reg.score(test_fmri_scaled, test_latents_scaled)
    print(f"Held-out R^2 (scaled space): {r2:.4f}")

    pred_dir = data_root / "predicted_features" / f"subj{args.subject:02d}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.output_suffix}_sub{args.subject}.npy"
    np.save(pred_path, pred_latents.astype(np.float32))
    print(f"Saved predicted latents to {pred_path}")

    weights_dir = data_root / "regression_weights" / f"subj{args.subject:02d}"
    weights_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "alpha": best_alpha,
        "coef": reg.coef_.astype(np.float32),
        "intercept": reg.intercept_.astype(np.float32),
        "fmri_scaler_mean": fmri_scaler.mean_.astype(np.float32),
        "fmri_scaler_scale": fmri_scaler.scale_.astype(np.float32),
        "latent_scaler_mean": latent_scaler.mean_.astype(np.float32),
        "latent_scaler_scale": latent_scaler.scale_.astype(np.float32),
    }

    weights_path = weights_dir / "sdxl_vae_regression_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved regression weights to {weights_path}")


if __name__ == "__main__":
    main()
