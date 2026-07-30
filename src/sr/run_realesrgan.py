"""
Precompute Real-ESRGAN x4 outputs for every image in images.csv.

Mirrors the EXACT rel_path from images.csv (relative to data_root) under
processed_root/realesrgan/, e.g.:

    data_root/001.ALI_HD/foo.jpg   -> processed_root/realesrgan/001.ALI_HD/foo.png

This never tries to reconstruct subject folder names -- it reads rel_path
straight from images.csv (written by build_metadata.py, which already
resolved your actual on-disk layout) and mirrors it exactly. Works
regardless of folder naming convention ("01", "001_", "001.ALI_HD", ...)
or whether there's an "images/" subdirectory.

We batch by source folder (all images sharing the same parent directory,
i.e. one subject) so the underlying repo's flat output directory never
has filename collisions across different subjects.

Usage:
    python -m src.sr.run_realesrgan                 # all subjects
    python -m src.sr.run_realesrgan --subjects 1 2 3 # just a few (debug)
    python -m src.sr.run_realesrgan --limit 5        # first 5 images/subject (debug)
    python -m src.sr.run_realesrgan --dry-run        # print the commands, do nothing
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import CFG


def run_one_group(rel_paths: list[str], dry_run: bool = False):
    """rel_paths: images.csv rel_path values that all share the same
    parent folder (i.e. one subject's images)."""
    out_root = CFG.processed_root / "realesrgan"

    todo = [rp for rp in rel_paths if not (out_root / rp).with_suffix(".png").exists()]
    if not todo:
        return  # already done (safe to re-run / resume)

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        tmp_in = Path(tmp_in)
        for rp in todo:
            fname = Path(rp).name
            (tmp_in / fname).symlink_to((CFG.data_root / rp).resolve())

        cmd = [
            str(CFG.realesrgan_python), "inference_realesrgan.py",
            "-n", CFG.sr.realesrgan_model_name,
            "-i", str(tmp_in),
            "-o", str(tmp_out),
            "--model_path", str(CFG.realesrgan_ckpt),
            "--suffix", "",           # no suffix -> output keeps input stem
            "--ext", "png",
        ]
        if CFG.sr.realesrgan_tile:
            cmd += ["--tile", str(CFG.sr.realesrgan_tile)]
        if CFG.sr.realesrgan_fp32:
            cmd += ["--fp32"]

        if dry_run:
            print(" ".join(cmd))
            return

        subprocess.run(cmd, cwd=str(CFG.realesrgan_repo), check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Move outputs into our canonical structure, matched by stem.
        for rp in todo:
            stem = Path(rp).stem
            candidates = list(Path(tmp_out).glob(f"{stem}*.png"))
            if not candidates:
                print(f"  [warn] no Real-ESRGAN output found for {rp}")
                continue
            out_path = (out_root / rp).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidates[0]), str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                     help="only process these subject_ids (debug)")
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N images per subject (debug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CFG.realesrgan_repo.exists():
        raise FileNotFoundError(
            f"{CFG.realesrgan_repo} not found. Run scripts/01_clone_sr_repos.sh first."
        )
    if not CFG.realesrgan_ckpt.exists():
        raise FileNotFoundError(
            f"{CFG.realesrgan_ckpt} not found. Run scripts/02_download_sr_checkpoints.sh first."
        )
    if not CFG.images_csv.exists():
        raise FileNotFoundError(
            f"{CFG.images_csv} not found. Run scripts/03_build_metadata.sh first."
        )

    df = pd.read_csv(CFG.images_csv)
    subjects = sorted(df["subject_id"].unique())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]

    (CFG.processed_root / "realesrgan").mkdir(parents=True, exist_ok=True)

    for sid in tqdm(subjects, desc="Real-ESRGAN"):
        rel_paths = list(df[df["subject_id"] == sid]["rel_path"])
        if args.limit:
            rel_paths = rel_paths[: args.limit]
        run_one_group(rel_paths, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
