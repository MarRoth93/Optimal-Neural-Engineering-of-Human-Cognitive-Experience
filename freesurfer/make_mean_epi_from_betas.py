#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from pathlib import Path

ROOT = os.environ.get("ROOT", "/home/rothermm/brain-diffuser")
SUBJECT = os.environ.get("SUBJECT", "subj01")

betas_dir = Path(f"{ROOT}/data/nsddata_betas/ppdata/{SUBJECT}/func1pt8mm/betas_fithrf_GLMdenoise_RR")
out_dir   = Path(f"{ROOT}/data/nsddata/ppdata/{SUBJECT}/func1pt8mm")
out_dir.mkdir(parents=True, exist_ok=True)
out_file  = out_dir / "mean_epi_from_betas.nii.gz"

# collect session files
sess_files = sorted(betas_dir.glob("betas_session*.nii.gz"))
if not sess_files:
    raise FileNotFoundError(f"No beta files found in {betas_dir}")

print(f"Found {len(sess_files)} sessions in {betas_dir}")

# Running sum of per-session means to keep memory down a bit
sum_img = None
ref_img = None

for i, f in enumerate(sess_files, 1):
    img = nib.load(str(f))
    data = np.asarray(img.dataobj, dtype=np.float32)  # shape: X,Y,Z,750
    m = data.mean(axis=3)                             # per-session mean (X,Y,Z)
    if sum_img is None:
        sum_img = m
        ref_img = img
    else:
        sum_img += m
    print(f"Session {i:02d}: {f.name} -> mean computed")

grand_mean = (sum_img / len(sess_files)).astype(np.float32)

# Save with first session's affine/header
out_img = nib.Nifti1Image(grand_mean, ref_img.affine, ref_img.header)
nib.save(out_img, str(out_file))
print(f"Wrote: {out_file}")