"""
Precompute SwinIR-Large (real_sr, x4) outputs for every image in images.csv.

Mirrors the EXACT rel_path from images.csv (relative to data_root) under
processed_root/swinir_large/ -- same batching approach as
run_realesrgan.py, see that file's docstring for why (dominant cost is
reloading the checkpoint onto the GPU once per subprocess call, not the
per-image forward pass, for tiny ear crops -- so we batch many images per
call instead of one call per subject).

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
    python -m src.sr.run_swinir --batch-size 500
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


def run_one_batch(rel_paths: list[str], out_root: Path, dry_run: bool = False):
    """rel_paths: images.csv rel_path values (can span many subjects)."""
    todo = [rp for rp in rel_paths if not (out_root / rp).with_suffix(".png").exists()]
    if not todo:
        return

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_in = Path(tmp_in)
        index_to_relpath = {}
        for i, rp in enumerate(todo):
            flat_name = f"{i:06d}{Path(rp).suffix}"
            (tmp_in / flat_name).symlink_to((CFG.data_root / rp).resolve())
            index_to_relpath[i] = rp

        cmd = [
            str(CFG.swinir_python), "main_test_swinir.py",
            "--task", CFG.sr.swinir_task,
            "--scale", str(CFG.sr.swinir_scale),
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

        results_root = CFG.swinir_repo / "results"

        def move_outputs():
            moved = 0
            for i, rp in index_to_relpath.items():
                stem = f"{i:06d}"
                found = _find_output(results_root, stem)
                if found is None:
                    continue
                out_path = (out_root / rp).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(found), str(out_path))
                moved += 1
            return moved

        try:
            subprocess.run(cmd, cwd=str(CFG.swinir_repo), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
        except subprocess.CalledProcessError as e:
            moved = move_outputs()
            print(f"\n[run_swinir] FAILED (exit {e.returncode}) after "
                  f"saving {moved}/{len(todo)} images from this batch. Full output:\n")
            print(e.stdout)
            print(f"\n[run_swinir] cmd: {' '.join(cmd)}")
            print(f"[run_swinir] cwd: {CFG.swinir_repo}")
            raise

        missing = move_outputs()
        if missing < len(todo):
            print(f"  [warn] only {missing}/{len(todo)} outputs found for this batch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                     help="override configs/paths.yaml -> sr.precompute_batch_size")
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
    if args.subjects:
        df = df[df["subject_id"].isin(args.subjects)]
    if args.limit:
        df = df.groupby("subject_id", group_keys=False).head(args.limit)

    # Skip images that src/data/dataset.py will bypass to Original anyway
    # (native long side already >= the skip threshold -- see
    # configs/paths.yaml -> dataset.skip_upsample_if_native_long_side_at_least
    # and src/data/dataset.py's module docstring for why).
    threshold = CFG.dataset.skip_upsample_if_native_long_side_at_least
    if threshold is not None:
        long_side = df[["width", "height"]].max(axis=1)
        n_before = len(df)
        df = df[long_side < threshold]
        n_skipped = n_before - len(df)
        if n_skipped:
            print(f"[run_swinir] skipping {n_skipped} image(s) with native long side "
                  f">= {threshold}px (dataset.py will read Original pixels for these instead)")

    all_rel_paths = list(df["rel_path"])
    batch_size = args.batch_size or CFG.sr.precompute_batch_size
    out_root = CFG.processed_root / "swinir_large"
    out_root.mkdir(parents=True, exist_ok=True)

    batches = [all_rel_paths[i:i + batch_size] for i in range(0, len(all_rel_paths), batch_size)]
    print(f"[run_swinir] {len(all_rel_paths)} images across {df['subject_id'].nunique()} "
          f"subjects -> {len(batches)} batch(es) of up to {batch_size} images")

    for batch in tqdm(batches, desc="SwinIR-Large batches"):
        run_one_batch(batch, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
