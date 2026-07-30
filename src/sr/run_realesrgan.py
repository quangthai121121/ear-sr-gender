"""
Precompute Real-ESRGAN x4 outputs for every image in images.csv.

This calls the *cloned* xinntao/Real-ESRGAN repo's inference_realesrgan.py
as a subprocess (its own env/deps must be installed -- see
scripts/01_clone_sr_repos.sh). We process one subject folder at a time so
that output filenames never collide across subjects (the tool writes to a
flat output folder), then move/rename results into:

    processed_root/realesrgan/<subject_id>/<original_filename>.png

Usage:
    python -m src.sr.run_realesrgan                 # all subjects
    python -m src.sr.run_realesrgan --subjects 1 2 3 # just a few (debug)
    python -m src.sr.run_realesrgan --limit 5        # first 5 images/subject (debug)
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


def run_one_subject(subject_id: int, filenames: list[str], out_root: Path,
                     dry_run: bool = False):
    out_dir = out_root / f"{subject_id:02d}" if False else out_root / str(subject_id)
    # NOTE: rel_path in images.csv already encodes the exact subject folder
    # name (e.g. "images/01/xxx.jpg"); we mirror that folder name here.
    src_subject_dir = CFG.data_root / "images" / _subject_folder_name(subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [f for f in filenames if not (out_dir / Path(f).with_suffix(".png")).exists()]
    if not todo:
        return  # already done (safe to re-run / resume)

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        tmp_in = Path(tmp_in)
        for f in todo:
            (tmp_in / f).symlink_to((src_subject_dir / f).resolve())

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
        for f in todo:
            stem = Path(f).stem
            candidates = list(Path(tmp_out).glob(f"{stem}*.png"))
            if not candidates:
                print(f"  [warn] no Real-ESRGAN output found for subject {subject_id} / {f}")
                continue
            shutil.move(str(candidates[0]), str(out_dir / f"{stem}.png"))


def _subject_folder_name(subject_id: int) -> str:
    """images.csv stores rel_path like 'images/01/xxx.jpg' -- subject folder
    names are NOT necessarily zero-padded the same way everywhere, so we
    resolve the actual on-disk folder name once via images.csv rather than
    assuming zero-padding here."""
    return _SUBJECT_FOLDER_CACHE[subject_id]


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

    df = pd.read_csv(CFG.images_csv)
    global _SUBJECT_FOLDER_CACHE
    _SUBJECT_FOLDER_CACHE = {
        sid: Path(rel).parts[1]  # rel_path = "images/<folder>/<file>"
        for sid, rel in df[["subject_id", "rel_path"]].drop_duplicates("subject_id").values
    }

    subjects = sorted(df["subject_id"].unique())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]

    out_root = CFG.processed_root / "realesrgan"
    out_root.mkdir(parents=True, exist_ok=True)

    for sid in tqdm(subjects, desc="Real-ESRGAN"):
        sub_df = df[df["subject_id"] == sid]
        filenames = [Path(rp).name for rp in sub_df["rel_path"]]
        if args.limit:
            filenames = filenames[: args.limit]
        run_one_subject(sid, filenames, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
