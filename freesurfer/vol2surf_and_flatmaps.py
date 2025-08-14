#!/usr/bin/env python3
"""
Volume → Surface projection (mri_vol2surf) + flatmap rendering (PyCortex).

Default inputs (if none given):
    <ROOT>/data/brain_mapping/<SUBJECT>/recon_fmri_alpha-4_idx755_memnet.nii.gz
    <ROOT>/data/brain_mapping/<SUBJECT>/recon_fmri_alpha+4_idx755_memnet.nii.gz

Outputs:
  - Surface maps (.mgh): <ROOT>/results/surf_native/<SUBJECT>/{lh.,rh.}<stem>.mgh
  - Flatmaps (.png/.svg): <ROOT>/results/flatmaps/<SUBJECT>/<stem>_flatmap.{png,svg}
  - Local PyCortex filestore: <ROOT>/results/pycortex_db

Requirements:
  - FreeSurfer in PATH (mri_vol2surf)
  - FS_LICENSE set and readable
  - SUBJECTS_DIR points to subject with {brainmask.mgz, lh/rh.{white,pial}}
  - Python: nibabel, numpy, matplotlib, pycortex

Example:
  export ROOT=/home/rothermm/brain-diffuser
  export SUBJECT=subj01
  export SUBJECTS_DIR=$ROOT/freesurfer
  export FS_LICENSE=$ROOT/freesurfer/license.txt

  python vol2surf_and_flatmaps.py --symmetric
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Non-interactive rendering
os.environ.setdefault("MPLBACKEND", "Agg")


def run(cmd, env=None):
    print("[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


def require(p: Path, msg: str):
    if not p.exists():
        print(f"[ERROR] {msg}: {p}", file=sys.stderr)
        sys.exit(2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=os.environ.get("SUBJECT", "subj01"),
                    help="Subject label, e.g., subj01")
    ap.add_argument("--root", default=os.environ.get("ROOT", "/home/rothermm/brain-diffuser"),
                    help="Project root")
    ap.add_argument("--subjects_dir", default=os.environ.get("SUBJECTS_DIR", None),
                    help="FreeSurfer SUBJECTS_DIR; defaults to <root>/freesurfer")
    ap.add_argument("--register", default=None,
                    help="Path to register.dat; default <root>/results/nifti/<SUBJECT>_register.dat")
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="Input NIfTI(s); defaults to memnet ±4 in brain_mapping/<SUBJECT>")
    ap.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for flatmaps")
    ap.add_argument("--vmin", type=float, default=None, help="Color scale min")
    ap.add_argument("--vmax", type=float, default=None, help="Color scale max")
    ap.add_argument("--symmetric", action="store_true",
                    help="If set and vmin/vmax not provided, use symmetric scaling from data")
    ap.add_argument("--with-sulci", dest="with_sulci", action="store_true",
                    help="Draw sulcal map on flatmaps")
    ap.add_argument("--no-sulci", dest="with_sulci", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(with_sulci=True)
    ap.add_argument("--with-colorbar", dest="with_colorbar", action="store_true",
                    help="Add colorbar to flatmaps")
    ap.add_argument("--no-colorbar", dest="with_colorbar", action="store_false", help=argparse.SUPPRESS)
    ap.set_defaults(with_colorbar=True)
    ap.add_argument("--projfrac-avg", nargs=3, metavar=("START", "END", "STEP"),
                    default=("0", "1", "0.1"),
                    help="Ribbon projection range and step (default: 0 1 0.1)")
    ap.add_argument("--surf", default="pial", choices=["pial", "white", "orig", "inflated"],
                    help="Surface for sampling (default: pial)")
    return ap.parse_args()


def ensure_pycortex_subject_noninteractive(subject: str, subjects_dir: Path, store_dir: Path):
    """
    Ensure PyCortex subject exists in LOCAL filestore without interactive prompts.
    Uses overwrite=True if supported; otherwise overrides module-level input() used by cortex.database.
    """
    print("[INFO] Ensuring PyCortex subject exists in local filestore (idempotent)")
    try:
        import cortex
        from cortex import database

        # Point database to local filestore
        database.db = database.Database(str(store_dir))

        try:
            _ = database.db.get_paths(subject)  # raises if not registered
            print("[INFO] Subject already present in local filestore; skipping import")
            return
        except Exception:
            pass  # will import

        # Try to import with overwrite=True if available
        import inspect
        sig = None
        try:
            sig = inspect.signature(cortex.freesurfer.import_subj)
        except Exception:
            sig = None

        if sig and ("overwrite" in sig.parameters):
            print("[INFO] Importing PyCortex subject with overwrite=True (non-interactive)")
            cortex.freesurfer.import_subj(subject, str(subjects_dir), overwrite=True)
        else:
            print("[INFO] Importing PyCortex subject with module-level prompt override (non-interactive)")
            # Override the specific input used inside cortex.database
            database.input = lambda *a, **k: "YES"
            cortex.freesurfer.import_subj(subject, str(subjects_dir))

        print("[OK] PyCortex import complete")
    except Exception as e:
        print(f"[ERROR] PyCortex subject import failed: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    args = parse_args()

    ROOT = Path(args.root)
    SUBJECT = args.subject
    SUBJECTS_DIR = Path(args.subjects_dir) if args.subjects_dir else ROOT / "freesurfer"
    REG = Path(args.register) if args.register else ROOT / f"results/nifti/{SUBJECT}_register.dat"

    # Derived dirs
    in_dir = ROOT / f"data/brain_mapping/{SUBJECT}"
    surf_dir = ROOT / f"results/surf_native/{SUBJECT}"
    flat_dir = ROOT / f"results/flatmaps/{SUBJECT}"
    store_dir = ROOT / "results" / "pycortex_db"  # local PyCortex filestore
    surf_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Inputs (defaults to memnet ±4)
    inputs = args.inputs
    if not inputs:
        inputs = [
            str(in_dir / "recon_fmri_alpha-4_idx755_memnet.nii.gz"),
            str(in_dir / "recon_fmri_alpha+4_idx755_memnet.nii.gz"),
        ]
    inputs = [Path(p) for p in inputs]

    # Sanity checks: env & tools
    fs_license = os.environ.get("FS_LICENSE", "")
    if not fs_license or not Path(fs_license).exists():
        print(f"[ERROR] FS_LICENSE not set or unreadable: {fs_license}", file=sys.stderr)
        sys.exit(2)
    if not shutil.which("mri_vol2surf"):
        print("[ERROR] mri_vol2surf not found in PATH (load freesurfer module).", file=sys.stderr)
        sys.exit(2)

    # Sanity checks: FS subject + registration + inputs
    require(SUBJECTS_DIR / SUBJECT, "Missing FreeSurfer subject dir")
    require(SUBJECTS_DIR / SUBJECT / "mri" / "brainmask.mgz", "Missing brainmask")
    for hemi_file in ["lh.white", "lh.pial", "rh.white", "rh.pial"]:
        require(SUBJECTS_DIR / SUBJECT / "surf" / hemi_file, f"Missing surface {hemi_file}")
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

    # Point PyCortex DB to local filestore early (for consistency)
    print(f"[INFO] Using local PyCortex filestore: {store_dir}")
    database.db = database.Database(str(store_dir))

    # FreeSurfer tool environment
    child_env = os.environ.copy()
    child_env["SUBJECTS_DIR"] = str(SUBJECTS_DIR)
    child_env["FS_LICENSE"] = str(fs_license)

    # 1) VOL→SURF for each input, both hemis
    out_pairs = []  # (stem, lh_mgh, rh_mgh)
    pf_start, pf_end, pf_step = args.projfrac_avg
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
                "--projfrac-avg", pf_start, pf_end, pf_step,
                "--interp", "trilinear",
                "--surf", args.surf,
            ]
            run(cmd, env=child_env)
            require(out_path, f"mri_vol2surf failed to create {out_path}")

        out_pairs.append((base, lh_out, rh_out))

    # 2) Ensure PyCortex subject exists (non-interactive)
    ensure_pycortex_subject_noninteractive(SUBJECT, SUBJECTS_DIR, store_dir)

    # 3) Flatmaps for each pair
    for base, lh_mgh, rh_mgh in out_pairs:
        print(f"[INFO] Rendering flatmap for {base}")

        L = nib.load(str(lh_mgh)).get_fdata().squeeze()
        R = nib.load(str(rh_mgh)).get_fdata().squeeze()

        vmin, vmax = args.vmin, args.vmax
        if args.symmetric and (vmin is None and vmax is None):
            m = float(max(abs(L).max(), abs(R).max()))
            if m == 0:
                m = 1.0
            vmin, vmax = -m, m

        vtx = cortex.Vertex({"lh": L, "rh": R}, SUBJECT, vmin=vmin, vmax=vmax, cmap=args.cmap)
        fig = cortex.quickflat.make_figure(
            vtx,
            with_rois=False,
            with_colorbar=args.with_colorbar,
            with_sulci=args.with_sulci,
            recache=False
        )
        fig.set_size_inches(10, 5)

        png = (Path(flat_dir) / f"{base}_flatmap.png").as_posix()
        svg = (Path(flat_dir) / f"{base}_flatmap.svg").as_posix()
        import matplotlib.pyplot as plt
        plt.savefig(png, dpi=300, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {png}")
        print(f"[OK] Wrote: {svg}")

    print("[DONE] All projections + flatmaps complete.")


if __name__ == "__main__":
    main()
