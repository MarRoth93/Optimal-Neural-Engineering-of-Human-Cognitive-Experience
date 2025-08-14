#!/usr/bin/env python3
"""
Run FreeSurfer bbregister for a subject using the mean EPI you created.

Writes:  $ROOT/results/nifti/<SUBJECT>_register.dat
Creates: $ROOT/results/nifti/<SUBJECT>_register.log  (bbregister's own log)
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

REQUIRED_FS_FILES = [
    "mri/brainmask.mgz",
    "surf/lh.white", "surf/lh.pial",
    "surf/rh.white", "surf/rh.pial",
]

def run(cmd, env=None):
    print("[CMD]", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise RuntimeError("Command failed with exit code {}: {}".format(completed.returncode, " ".join(cmd)))

def check_fs_subject(subjects_dir: Path, subject: str):
    missing = []
    base = subjects_dir / subject
    for rel in REQUIRED_FS_FILES:
        p = base / rel
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[ERROR] FreeSurfer subject is incomplete. Missing:", *missing, sep="\n  ", file=sys.stderr)
        sys.exit(2)
    print("[OK] Found FreeSurfer subject files under:", base)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=os.environ.get("SUBJECT", "subj01"),
                        help="Subject label, e.g., subj01")
    parser.add_argument("--root", default=os.environ.get("ROOT", "/home/rothermm/brain-diffuser"))
    parser.add_argument("--subjects_dir", default=os.environ.get("SUBJECTS_DIR", None),
                        help="FreeSurfer SUBJECTS_DIR; defaults to <root>/freesurfer")
    parser.add_argument("--mean-epi", default=None,
                        help="Path to mean EPI NIfTI; default: <root>/data/nsddata/ppdata/<SUBJECT>/func1pt8mm/mean_epi_from_betas.nii.gz")
    parser.add_argument("--out", default=None,
                        help="Output register.dat; default: <root>/results/nifti/<SUBJECT>_register.dat")
    parser.add_argument("--init", default="fsl", choices=["fsl","coreg"],
                        help="bbregister initialization (default: fsl)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing register.dat")
    parser.add_argument("--retry-coreg", action="store_true", default=True,
                        help="If init=fsl fails, automatically retry with init=coreg")
    args = parser.parse_args()

    ROOT = Path(args.root)
    SUBJECT = args.subject
    SUBJECTS_DIR = Path(args.subjects_dir) if args.subjects_dir else ROOT / "freesurfer"
    MEAN_EPI = Path(args.mean_epi) if args.mean_epi else ROOT / f"data/nsddata/ppdata/{SUBJECT}/func1pt8mm/mean_epi_from_betas.nii.gz"
    OUT = Path(args.out) if args.out else ROOT / f"results/nifti/{SUBJECT}_register.dat"
    LOG = OUT.with_suffix(".log")  # bbregister writes here automatically

    # Echo config
    print(f"[INFO] ROOT={ROOT}")
    print(f"[INFO] SUBJECT={SUBJECT}")
    print(f"[INFO] SUBJECTS_DIR={SUBJECTS_DIR}")
    print(f"[INFO] MEAN_EPI={MEAN_EPI}")
    print(f"[INFO] OUT register.dat={OUT}")
    print(f"[INFO] INIT={args.init}")

    # Basic checks
    fs_license = os.environ.get("FS_LICENSE", "")
    if not fs_license or not Path(fs_license).exists():
        print(f"[ERROR] FS_LICENSE not set or unreadable: {fs_license}", file=sys.stderr)
        sys.exit(2)

    bbregister_bin = shutil.which("bbregister")
    if not bbregister_bin:
        print("[ERROR] bbregister not found in PATH (load freesurfer module).", file=sys.stderr)
        sys.exit(2)

    if not SUBJECTS_DIR.joinpath(SUBJECT).exists():
        print(f"[ERROR] Subject directory not found under SUBJECTS_DIR: {SUBJECTS_DIR}/{SUBJECT}", file=sys.stderr)
        sys.exit(2)
    check_fs_subject(SUBJECTS_DIR, SUBJECT)

    if not MEAN_EPI.exists():
        print(f"[ERROR] Mean EPI file not found: {MEAN_EPI}", file=sys.stderr)
        sys.exit(2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not args.force:
        print(f"[OK] Register file already exists: {OUT} (use --force to overwrite)")
        sys.exit(0)

    # Environment for bbregister
    child_env = os.environ.copy()
    child_env["SUBJECTS_DIR"] = str(SUBJECTS_DIR)
    child_env["FS_LICENSE"] = str(fs_license)

    # Map init opt to flag
    def init_flag(which: str):
        return "--init-fsl" if which == "fsl" else "--init-coreg"

    # Try primary init
    try_init = args.init
    tried = []
    for attempt in (try_init, ("coreg" if args.retry_coreg and try_init == "fsl" else None)):
        if attempt is None:
            continue
        tried.append(attempt)
        print(f"[INFO] Running bbregister with init={attempt}")
        cmd = [
            "bbregister",
            "--s", SUBJECT,
            "--mov", str(MEAN_EPI),
            "--reg", str(OUT),
            init_flag(attempt),
            "--bold",
        ]
        try:
            run(cmd, env=child_env)
            break
        except RuntimeError as e:
            print(f"[WARN] bbregister failed with init={attempt}: {e}", file=sys.stderr)
            if attempt == tried[-1] and (not args.retry_coreg or attempt == "coreg"):
                print("[ERROR] All attempts failed.", file=sys.stderr)
                sys.exit(1)

    # Verify
    if OUT.exists() and OUT.stat().st_size > 0:
        print(f"[OK] Wrote register.dat: {OUT}")
        if LOG.exists():
            print(f"[OK] bbregister log: {LOG}")
        else:
            print("[NOTE] bbregister log file not found alongside register.dat (this is okay on some builds).")
        print("Tip: to visually inspect/finetune, run:\n"
              f"  tkregister2 --mov {MEAN_EPI} --reg {OUT} --surf")
        sys.exit(0)
    else:
        print(f"[ERROR] bbregister finished but {OUT} missing/empty.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
