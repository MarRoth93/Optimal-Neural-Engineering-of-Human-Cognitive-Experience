#!/usr/bin/env python3
"""
Download a full FreeSurfer subject from the NSD public S3 and verify key files.

Examples:
  python download_fs_subject.py --subject subj01 \
      --root /home/rothermm/brain-diffuser --also-fsaverage --mode safe
  python download_fs_subject.py --subject subj01 --mode force
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

NSD_BUCKET = "s3://natural-scenes-dataset"

def run(cmd):
    print("[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def check_tool(tool):
    path = shutil.which(tool)
    if not path:
        print("[ERROR] Required tool not found in PATH:", tool, file=sys.stderr)
        raise SystemExit(2)
    print("[OK] Found {} at {}".format(tool, path))

def aws_cli_major_version():
    # aws-cli/1.38.10 Python/...  OR aws-cli/2.15.10 ...
    try:
        out = subprocess.check_output(["aws", "--version"], stderr=subprocess.STDOUT).decode("utf-8")
        m = re.search(r"aws-cli/(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1  # assume v1 if uncertain

def verify_subject(subj_dir):
    must = [
        subj_dir / "mri" / "brainmask.mgz",
        subj_dir / "surf" / "lh.white",
        subj_dir / "surf" / "lh.pial",
        subj_dir / "surf" / "rh.white",
        subj_dir / "surf" / "rh.pial",
    ]
    missing = [str(p) for p in must if not p.exists()]
    if missing:
        print("[ERROR] Missing required files after download:", *missing, sep="\n  ", file=sys.stderr)
        raise SystemExit(3)
    print("[OK] Verified key files:", *[str(p) for p in must], sep="\n  ")

def dest_nonempty(dest):
    # True if directory contains any files
    if not dest.exists():
        return False
    for _root, _dirs, files in os.walk(dest):
        if files:
            return True
    return False

def sync_subject(subject, root, also_fsaverage, mode):
    """
    mode = "safe"  -> with AWS CLI v2: cp --no-clobber (add-only).
                     with AWS CLI v1: if destination has any files, ABORT; else sync.
    mode = "force" -> full sync (may overwrite).
    """
    check_tool("aws")
    aws_major = aws_cli_major_version()
    print("[INFO] AWS CLI major version detected:", aws_major)

    dest_subj = root / "freesurfer" / subject
    dest_subj.mkdir(parents=True, exist_ok=True)

    if mode == "safe":
        if aws_major >= 2:
            # add-only, no overwrite
            src_subj = "{}/nsddata/freesurfer/{}".format(NSD_BUCKET, subject)
            print("[INFO] SAFE copy (add-only) {} -> {}".format(subject, dest_subj))
            run([
                "aws", "s3", "cp", "--recursive", src_subj, str(dest_subj),
                "--no-sign-request", "--no-clobber"
            ])
        else:
            # v1: Cannot no-clobber. Refuse if destination non-empty.
            if dest_nonempty(dest_subj):
                print("[ABORT] SAFE mode with AWS CLI v1: destination is not empty, refusing to risk overwrite:\n  {}".format(dest_subj))
                print("        Use --mode force if you want to sync/overwrite, or clean/move the folder and retry.")
                raise SystemExit(4)
            src_subj = "{}/nsddata/freesurfer/{}/".format(NSD_BUCKET, subject)
            print("[INFO] SAFE (v1 fallback) destination is empty; syncing {} -> {}".format(src_subj, dest_subj))
            run([
                "aws", "s3", "sync", src_subj, str(dest_subj),
                "--no-sign-request", "--exact-timestamps"
            ])
    else:
        # FORCE
        src_subj = "{}/nsddata/freesurfer/{}/".format(NSD_BUCKET, subject)
        print("[INFO] FORCE sync {} -> {}".format(src_subj, dest_subj))
        run([
            "aws", "s3", "sync", src_subj, str(dest_subj),
            "--no-sign-request", "--exact-timestamps"
        ])

    verify_subject(dest_subj)

    if also_fsaverage:
        dest_fsa = root / "freesurfer" / "fsaverage"
        dest_fsa.mkdir(parents=True, exist_ok=True)
        if mode == "safe" and aws_major >= 2:
            src_fsa = "{}/nsddata/freesurfer/fsaverage".format(NSD_BUCKET)
            print("[INFO] SAFE copy (add-only) fsaverage -> {}".format(dest_fsa))
            run([
                "aws", "s3", "cp", "--recursive", src_fsa, str(dest_fsa),
                "--no-sign-request", "--no-clobber"
            ])
        elif mode == "safe" and aws_major < 2:
            if dest_nonempty(dest_fsa):
                print("[ABORT] SAFE mode with AWS CLI v1: fsaverage destination is not empty:\n  {}".format(dest_fsa))
                raise SystemExit(4)
            src_fsa = "{}/nsddata/freesurfer/fsaverage/".format(NSD_BUCKET)
            print("[INFO] SAFE (v1 fallback) destination empty; syncing fsaverage -> {}".format(dest_fsa))
            run([
                "aws", "s3", "sync", src_fsa, str(dest_fsa),
                "--no-sign-request", "--exact-timestamps"
            ])
        else:
            src_fsa = "{}/nsddata/freesurfer/fsaverage/".format(NSD_BUCKET)
            print("[INFO] FORCE sync fsaverage -> {}".format(dest_fsa))
            run([
                "aws", "s3", "sync", src_fsa, str(dest_fsa),
                "--no-sign-request", "--exact-timestamps"
            ])
        print("[OK] fsaverage retrieval complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--root", default="/home/rothermm/brain-diffuser")
    parser.add_argument("--also-fsaverage", action="store_true")
    parser.add_argument("--mode", choices=["safe", "force"], default="safe",
                        help="safe: add-only (no overwrites). force: full sync (may overwrite).")
    args = parser.parse_args()

    root = Path(args.root)
    print("[INFO] ROOT=", root)
    print("[INFO] SUBJECT=", args.subject)
    print("[INFO] ALSO_FSAVERAGE=", args.also_fsaverage)
    print("[INFO] MODE=", args.mode)

    try:
        sync_subject(args.subject, root, args.also_fsaverage, args.mode)
    except SystemExit as e:
        code = int(str(e) or 1)
        print("[FAIL] Download/verify failed with exit code", code, file=sys.stderr)
        sys.exit(code)

    print("[DONE] Subject ready at:", root / "freesurfer" / args.subject)
    print("       Next step: run registration sbatch for bbregister.")
    sys.exit(0)

if __name__ == "__main__":
    main()
