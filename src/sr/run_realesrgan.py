"""
Precompute Real-ESRGAN x4 outputs for every image in images.csv.

Mirrors the EXACT rel_path from images.csv (relative to data_root) under
processed_root/realesrgan/, e.g.:

    data_root/001.ALI_HD/foo.jpg   -> processed_root/realesrgan/001.ALI_HD/foo.png

PERFORMANCE NOTE: this batches MANY images per subprocess call
(configs/paths.yaml -> sr.precompute_batch_size, default 1000), not one
call per subject. Each subprocess call pays a fixed cost (interpreter
startup, torch/basicsr import, loading the checkpoint onto the GPU) that
is independent of how many images it processes -- for tiny ear crops,
that fixed cost dominates total runtime if paid once per subject (~164
times) instead of once per large batch (~28 times for 28k images at
batch_size=1000). Images within one batch still get processed ONE AT A
TIME on the GPU by the underlying repo's own script (it doesn't support
true multi-image batched forward passes) -- this fix removes the
redundant model-reload overhead, it does not add GPU-side batching.

We use INDEX-based flat filenames inside each batch's temp folder (not
the original filename) so images from different subjects never collide
even though many subjects may share generic filenames.

Usage:
    python -m src.sr.run_realesrgan                 # all subjects
    python -m src.sr.run_realesrgan --subjects 1 2 3 # just a few (debug)
    python -m src.sr.run_realesrgan --limit 5        # first 5 images/subject (debug)
    python -m src.sr.run_realesrgan --dry-run        # print the commands, do nothing
    python -m src.sr.run_realesrgan --batch-size 500 # override configs/paths.yaml
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


def run_one_batch(rel_paths: list[str], out_root: Path, dry_run: bool = False):
    """rel_paths: images.csv rel_path values (can span many subjects)."""
    todo = [rp for rp in rel_paths if not (out_root / rp).with_suffix(".png").exists()]
    if not todo:
        return

    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        tmp_in = Path(tmp_in)
        index_to_relpath = {}
        for i, rp in enumerate(todo):
            flat_name = f"{i:06d}{Path(rp).suffix}"
            (tmp_in / flat_name).symlink_to((CFG.data_root / rp).resolve())
            index_to_relpath[i] = rp

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

        def move_outputs():
            moved = 0
            for i, rp in index_to_relpath.items():
                stem = f"{i:06d}"
                candidates = list(Path(tmp_out).glob(f"{stem}*.png"))
                if not candidates:
                    continue
                out_path = (out_root / rp).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(candidates[0]), str(out_path))
                moved += 1
            return moved

        try:
            subprocess.run(cmd, cwd=str(CFG.realesrgan_repo), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
        except subprocess.CalledProcessError as e:
            # Save whatever DID finish before the crash, so a batch failure
            # doesn't throw away already-completed work in that batch.
            moved = move_outputs()
            print(f"\n[run_realesrgan] FAILED (exit {e.returncode}) after "
                  f"saving {moved}/{len(todo)} images from this batch. Full output:\n")
            print(e.stdout)
            print(f"\n[run_realesrgan] cmd: {' '.join(cmd)}")
            print(f"[run_realesrgan] cwd: {CFG.realesrgan_repo}")
            raise

        missing = move_outputs()
        if missing < len(todo):
            print(f"  [warn] only {missing}/{len(todo)} outputs found for this batch "
                  f"(some images may have failed silently inside inference_realesrgan.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                     help="only process these subject_ids (debug)")
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N images per subject (debug)")
    ap.add_argument("--batch-size", type=int, default=None,
                     help="override configs/paths.yaml -> sr.precompute_batch_size")
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
            print(f"[run_realesrgan] skipping {n_skipped} image(s) with native long side "
                  f">= {threshold}px (dataset.py will read Original pixels for these instead)")

    all_rel_paths = list(df["rel_path"])
    batch_size = args.batch_size or CFG.sr.precompute_batch_size
    out_root = CFG.processed_root / "realesrgan"
    out_root.mkdir(parents=True, exist_ok=True)

    batches = [all_rel_paths[i:i + batch_size] for i in range(0, len(all_rel_paths), batch_size)]
    print(f"[run_realesrgan] {len(all_rel_paths)} images across {df['subject_id'].nunique()} "
          f"subjects -> {len(batches)} batch(es) of up to {batch_size} images "
          f"({'one model load per batch' if not args.dry_run else 'dry run'})")

    for batch in tqdm(batches, desc="Real-ESRGAN batches"):
        run_one_batch(batch, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
