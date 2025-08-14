#!/usr/bin/env python3
"""
Volume→Surface projection (mri_vol2surf) + flatmap rendering (PyCortex)
for memnet ±4 maps.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def run(cmd, env=None):
    print("[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")

def require(p: Path, msg: str):
    if not p.exists():
        print(f"[ERROR] {msg}: {p}", file=sys.stderr)
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=os.environ.get("SUBJECT","subj01"))
    parser.add_argument("--root", default=os.environ.get("ROOT","/home/rothermm/brain-diffuser"))
    parser.add_argument("--subjects_dir", default=os.environ.get("SUBJECTS_DIR", None),
                        help="Defaults to <root>/freesurfer if not set")
    parser.add_argument("--register", default=None,
                        help="Path to register.dat; default <root>/results/nifti/<SUBJECT>_register.dat")
    parser.add_argument("--inputs", nargs="*", default=None,
                        help="Input NIfTI(s). Default: memnet ±4 files in brain_mapping/<SUBJECT>")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--symmetric", action="store_true",
                        help="If set and vmin/vmax not provided, use symmetric scaling from data")
    args = parser.parse_args()

    ROOT = Path(args.root)
    SUBJECT = args.subject
    SUBJECTS_DIR = Path(args.subjects_dir) if args.subjects_dir else ROOT/"freesurfer"
    REG = Path(args.register) if args.register else ROOT/f"results/nifti/{SUBJECT}_register.dat"

    in_dir    = ROOT / f"data/brain_mapping/{SUBJECT}"
    surf_dir  = ROOT / f"results/surf_native/{SUBJECT}"
    flat_dir  = ROOT / f"results/flatmaps/{SUBJECT}"
    store_dir = ROOT / "results" / "pycortex_db"   # local PyCortex filestore
    surf_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Default inputs (memnet ±4)
    inputs = args.inputs or [
        str(in_dir/"recon_fmri_alpha-4_idx755_memnet.nii.gz"),
        str(in_dir/"recon_fmri_alpha+4_idx755_memnet.nii.gz"),
    ]
    inputs = [Path(p) for p in inputs]

    # Sanity checks
    fs_license = os.environ.get("FS_LICENSE","")
    if not fs_license or not Path(fs_license).exists():
        print(f"[ERROR] FS_LICENSE not set or unreadable: {fs_license}", file=sys.stderr)
        sys.exit(2)
    if not shutil.which("mri_vol2surf"):
        print("[ERROR] mri_vol2surf not found in PATH (load freesurfer module).", file=sys.stderr)
        sys.exit(2)

    require(SUBJECTS_DIR/SUBJECT, "Missing FreeSurfer subject dir")
    require(SUBJECTS_DIR/SUBJECT/"mri"/"brainmask.mgz", "Missing brainmask")
    for hemi_file in ["lh.white","lh.pial","rh.white","rh.pial"]:
        require(SUBJECTS_DIR/SUBJECT/"surf"/hemi_file, f"Missing surface {hemi_file}")
    require(REG, "Missing register.dat")
    for p in inputs:
        require(p, "Missing input NIfTI")

    # Python deps
    try:
        import nibabel as nib
        import numpy as np
        import cortex
        import matplotlib
        import matplotlib.pyplot as plt
        from cortex import database
    except Exception as e:
        print(f"[ERROR] Python libs missing (pycortex/nibabel/matplotlib): {e}", file=sys.stderr)
        sys.exit(2)

    # Non-interactive Matplotlib
    os.environ.setdefault("MPLBACKEND","Agg")

    # Use LOCAL PyCortex filestore
    print(f"[INFO] Using local PyCortex filestore: {store_dir}")
    database.db = database.Database(str(store_dir))

    # FS tool environment
    child_env = os.environ.copy()
    child_env["SUBJECTS_DIR"] = str(SUBJECTS_DIR)
    child_env["FS_LICENSE"]   = str(fs_license)

    # 1) VOL→SURF for each input, both hemis
    out_pairs = []  # (stem, lh_mgh, rh_mgh)
    for src in inputs:
        base = src.name[:-7] if src.name.endswith(".nii.gz") else src.stem
        lh_out = surf_dir / f"lh.{base}.mgh"
        rh_out = surf_dir / f"rh.{base}.mgh"
        print(f"[INFO] Projecting {src.name} to surface: {lh_out.name}, {rh_out.name}")

        for hemi, out_path in (("lh", lh_out), ("rh", rh_out)):
            cmd = [
                "mri_vol2surf",
                "--mov", str(src),
                "--reg", str(REG),
                "--hemi", hemi,
                "--o", str(out_path),
                "--projfrac-avg", "0", "1", "0.1",  # ribbon avg 0..1 step 0.1
                "--interp", "trilinear",
                "--surf", "pial",
            ]
            run(cmd, env=child_env)
            require(out_path, f"mri_vol2surf failed to create {out_path}")

        out_pairs.append((base, lh_out, rh_out))

    # 2) Ensure PyCortex subject exists in LOCAL filestore (idempotent, non-interactive)
    print("[INFO] Ensuring PyCortex subject exists in local filestore (idempotent)")

    # Make absolutely sure both the global db and the default filestore point to our local store
    database.default_filestore = str(store_dir)
    database.db = database.Database(str(store_dir))

    def purge_subject_dirs():
        # PyCortex keeps per-subject assets in several subfolders. Remove any stale remnants.
        candidates = [
            store_dir / SUBJECT,
            store_dir / "db" / SUBJECT,
            store_dir / "overlays" / SUBJECT,
            store_dir / "surfaces" / SUBJECT,
            store_dir / "rois" / SUBJECT,
            store_dir / "masks" / SUBJECT,
            store_dir / "tf" / SUBJECT,
            store_dir / "blender" / SUBJECT,
        ]
        for p in candidates:
            if p.exists():
                print(f"[INFO] Removing stale dir: {p}")
                shutil.rmtree(p, ignore_errors=True)
        # Catch-all: any directory named SUBJECT anywhere under the filestore
        for p in store_dir.rglob(SUBJECT):
            if p.is_dir():
                print(f"[INFO] Removing stray dir: {p}")
                shutil.rmtree(p, ignore_errors=True)

    try:
        _ = database.db.get_paths(SUBJECT)  # raises if not registered
        print("[INFO] Subject already present in local filestore; skipping import")
    except Exception:
        purge_subject_dirs()

        # Feed "YES" to any interactive prompt to avoid EOFError in non-TTY runs
        import io, sys, contextlib
        @contextlib.contextmanager
        def feed_stdin(text):
            old = sys.stdin
            try:
                sys.stdin = io.StringIO(text)
                yield
            finally:
                sys.stdin = old

        with feed_stdin("YES\n"):
            cortex.freesurfer.import_subj(SUBJECT, str(SUBJECTS_DIR))
        print("[OK] PyCortex import complete")

if __name__ == "__main__":
    main()
