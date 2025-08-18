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

Safety:
  - Hard fail if PyCortex filestore overlaps with FreeSurfer SUBJECTS_DIR.
  - Never auto-confirm prompts; never overwrite an existing PyCortex subject.
  - If the subject already exists in the filestore, skip import.

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


def paths_overlap(a: Path, b: Path) -> bool:
    ra, rb = a.resolve(), b.resolve()
    return ra == rb or ra in rb.parents or rb in ra.parents


def looks_like_freesurfer_dir(p: Path) -> bool:
    p = p.resolve()
    return (p / "mri").exists() and (p / "surf").exists()


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


def ensure_pycortex_subject_safe(subject: str, subjects_dir: Path, store_dir: Path):
    """
    Safely ensure a PyCortex subject exists in the local filestore.

    - Hard-fails if filestore overlaps SUBJECTS_DIR or looks like a FreeSurfer tree.
    - Never overwrites a complete subject.
    - If DB says "not found" but a stale/incomplete folder exists, remove it and retry once.
    - Forces PyCortex to use the project-local filestore/config only.
    """
    # --- safety against catastrophic mis-paths ---
    if paths_overlap(store_dir, subjects_dir):
        print("[FATAL] PyCortex filestore and FreeSurfer SUBJECTS_DIR overlap!", file=sys.stderr)
        print(f"        filestore    = {store_dir.resolve()}", file=sys.stderr)
        print(f"        SUBJECTS_DIR = {subjects_dir.resolve()}", file=sys.stderr)
        sys.exit(3)
    if looks_like_freesurfer_dir(store_dir):
        print("[FATAL] Filestore path looks like a FreeSurfer SUBJECTS_DIR (has mri/ and surf/).", file=sys.stderr)
        print(f"        filestore = {store_dir.resolve()}", file=sys.stderr)
        sys.exit(3)

    # Force PyCortex to our isolated filestore/config
    os.environ["SUBJECTS_DIR"] = str(subjects_dir.resolve())
    os.environ["CORTEX_DB"]   = str(store_dir.resolve())
    xdg_cfg = Path(os.environ.get("XDG_CONFIG_HOME", store_dir.parent / ".pycortex_cfg"))
    cfg_dir = xdg_cfg / "pycortex"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    opt_file = cfg_dir / "options.cfg"
    if not opt_file.exists():
        opt_file.write_text(f"[basic]\nfilestore = {store_dir.resolve()}\n")

    import cortex
    from cortex import database, options
    try:
        options.config.set("basic", "filestore", str(store_dir.resolve()))
    except Exception:
        pass
    database.db = database.Database(str(store_dir))

    subj_dir = store_dir / subject

    def _is_complete(p: Path) -> bool:
        need = [
            p / "transforms.hdf",
            p / "surf" / "lh.fiducial",
            p / "surf" / "rh.fiducial",
            p / "surf" / "lh.inflated",
            p / "surf" / "rh.inflated",
        ]
        return all(q.exists() for q in need)

    # If a complete subject directory is already present, we’re done.
    if subj_dir.exists() and _is_complete(subj_dir):
        print("[INFO] PyCortex subject directory already complete; skipping import")
        return

    # If DB already has valid paths, we’re done.
    try:
        paths = database.db.get_paths(subject)
        if paths and isinstance(paths, dict) and all(Path(p).exists() for p in paths.values() if isinstance(p, str)):
            print("[INFO] PyCortex subject already registered with valid paths; skipping import")
            return
    except Exception:
        pass  # not registered

    # If an incomplete folder is present *before* import, remove it now.
    if subj_dir.exists() and not _is_complete(subj_dir):
        print(f"[INFO] Removing stale/incomplete PyCortex folder before import: {subj_dir}")
        shutil.rmtree(subj_dir, ignore_errors=True)

    # Block interactive prompts; never auto-YES
    import builtins, inspect
    original_input = getattr(builtins, "input", None)
    def non_interactive_input(prompt=""):
        print(f"[WARN] Interactive prompt blocked: {prompt}")
        return "NO"
    builtins.input = non_interactive_input

    def _do_import():
        sig = inspect.signature(cortex.freesurfer.import_subj)
        if "overwrite" in sig.parameters:
            print("[INFO] Importing PyCortex subject (overwrite=False)")
            return cortex.freesurfer.import_subj(subject, str(subjects_dir), overwrite=False)
        else:
            print("[INFO] Importing PyCortex subject (legacy API)")
            return cortex.freesurfer.import_subj(subject, str(subjects_dir))

    try:
        try:
            _do_import()
        except Exception as e:
            msg = str(e).lower()
            # If import refused due to an existing dir, remove it *once* and retry
            if ("overwrite" in msg or "do not overwrite" in msg or "exists" in msg) and subj_dir.exists():
                if _is_complete(subj_dir):
                    print("[INFO] Subject folder exists and appears complete; skipping import")
                    return
                print(f"[INFO] Import refused due to stale folder; removing and retrying: {subj_dir}")
                shutil.rmtree(subj_dir, ignore_errors=True)
                _do_import()
            else:
                raise
        print("[OK] PyCortex import complete")
    finally:
        if original_input is not None:
            builtins.input = original_input



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

    # Echo key paths/env
    print(f"[INFO] ROOT         = {ROOT.resolve()}")
    print(f"[INFO] SUBJECT      = {SUBJECT}")
    print(f"[INFO] SUBJECTS_DIR = {SUBJECTS_DIR.resolve()}")
    print(f"[INFO] register.dat = {REG}")
    print(f"[INFO] filestore    = {store_dir.resolve()}")
    if os.environ.get("CORTEX_DB"):
        print(f"[INFO] CORTEX_DB (env) = {Path(os.environ['CORTEX_DB']).resolve()}")
        if Path(os.environ["CORTEX_DB"]).resolve() != store_dir.resolve():
            print("[WARN] CORTEX_DB env != store_dir; forcing PyCortex to use store_dir at runtime.")

    if os.environ.get("XDG_CONFIG_HOME"):
        print(f"[INFO] XDG_CONFIG_HOME = {Path(os.environ['XDG_CONFIG_HOME']).resolve()}")

    # Create output dirs (safe)
    surf_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    # Inputs (defaults to memnet ±4)
    inputs = args.inputs or [
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
        import matplotlib.pyplot as plt
        from cortex import database
    except Exception as e:
        print(f"[ERROR] Python libs missing (pycortex/nibabel/matplotlib): {e}", file=sys.stderr)
        sys.exit(2)

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

    # 2) Ensure PyCortex subject exists (SAFE, non-overwriting, non-interactive)
    ensure_pycortex_subject_safe(SUBJECT, SUBJECTS_DIR, store_dir)

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
