"""
Precompute SwinIR-Large (real_sr, x4) outputs for every image in images.csv.

Mirrors the EXACT rel_path from images.csv (relative to data_root) under
processed_root/swinir_large/ -- same approach as run_realesrgan.py, see
that file's docstring for why (works regardless of your on-disk subject
folder naming/layout).

Calls the *cloned* JingyunLiang/SwinIR repo's main_test_swinir.py as a
subprocess. That script writes outputs to a fixed results/ subfolder with
its own naming convention (commonly "<stem>_SwinIR.png" for --task real_sr,
but this has changed across repo versions) -- we glob-match by stem prefix
so this is robust to the exact suffix. If matching fails for your clone,
open external/SwinIR/main_test_swinir.py, find where it calls
`cv2.imwrite(...)` / `save_path`, and adjust OUTPUT_GLOB_SUFFIXES below.

Usage:
    python -m src.sr.run_swinir
    python -m src.sr.run_swinir --subjects 1 2 3
    python -m src.sr.run_swinir --limit 5
    python -m src.sr.run_swinir --dry-run
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

OUTPUT_GLOB_SUFFIXES = ["_SwinIR.png", "_SwinIR-L.png", ".png"]


def _find_output(results_dir: Path, stem: str) -> Path | None:
    for suf in OUTPUT_GLOB_SUFFIXES:
        cands = list(results_dir.rglob(f"{stem}{suf}"))
        if cands:
            return cands[0]
    cands = list(results_dir.rglob(f"{stem}*"))
    return cands[0] if cands else None


def run_one_group(rel_paths: list[str], dry_run: bool = False):
    """rel_paths: images.csv rel_path values that all share the same
    parent folder (i.e. one subject's images)."""
    out_root = CFG.processed_root / "swinir_large"

    todo = [rp for rp in rel_paths if not (out_root / rp).with_suffix(".png").exists()]
    if not todo:
        return

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_in = Path(tmp_in)
        for rp in todo:
            fname = Path(rp).name
            (tmp_in / fname).symlink_to((CFG.data_root / rp).resolve())

        cmd = [
            str(CFG.swinir_python), "main_test_swinir.py",
            "--task", CFG.sr.swinir_task,
            "--model_path", str(CFG.swinir_large_ckpt),
            "--folder_lq", str(tmp_in),
        ]
        if CFG.sr.swinir_large_model:
            cmd += ["--large_model"]
        if CFG.sr.swinir_tile:
            cmd += ["--tile", str(CFG.sr.swinir_tile)]

        if dry_run:
            print(" ".join(cmd))
            return

        subprocess.run(cmd, cwd=str(CFG.swinir_repo), check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        results_root = CFG.swinir_repo / "results"
        for rp in todo:
            stem = Path(rp).stem
            found = _find_output(results_root, stem)
            if found is None:
                print(f"  [warn] no SwinIR output found for {rp} "
                      f"(looked under {results_root}). Check OUTPUT_GLOB_SUFFIXES.")
                continue
            out_path = (out_root / rp).with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(found), str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CFG.swinir_repo.exists():
        raise FileNotFoundError(
            f"{CFG.swinir_repo} not found. Run scripts/01_clone_sr_repos.sh first."
        )
    if not CFG.swinir_large_ckpt.exists():
        raise FileNotFoundError(
            f"{CFG.swinir_large_ckpt} not found. Run scripts/02_download_sr_checkpoints.sh first."
        )
    if not CFG.images_csv.exists():
        raise FileNotFoundError(
            f"{CFG.images_csv} not found. Run scripts/03_build_metadata.sh first."
        )

    df = pd.read_csv(CFG.images_csv)
    subjects = sorted(df["subject_id"].unique())
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]

    (CFG.processed_root / "swinir_large").mkdir(parents=True, exist_ok=True)

    for sid in tqdm(subjects, desc="SwinIR-Large"):
        rel_paths = list(df[df["subject_id"] == sid]["rel_path"])
        if args.limit:
            rel_paths = rel_paths[: args.limit]
        run_one_group(rel_paths, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
