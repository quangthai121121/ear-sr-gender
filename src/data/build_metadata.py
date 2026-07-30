"""
Scan <data_root>/images/<subject_id>/* and build:

  data/meta/images.csv    : image_id, rel_path, subject_id, gender, width, height, short_side
  data/meta/subjects.csv  : subject_id, gender, n_images

Gender rule (per EarVN1.0's own documentation): subject folders "01".."98"
are male, "99".."164" are female. VERIFY this against the readme shipped
with your Mendeley download before trusting it blindly -- if your local
copy numbers folders differently, adjust `subject_gender()` below.

Usage:
    python -m src.data.build_metadata
"""
from __future__ import annotations
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.config import CFG


def subject_gender(subject_id: int, n_male: int) -> str:
    return "male" if subject_id <= n_male else "female"


def iter_subject_dirs(data_root: Path):
    img_root = data_root / "images"
    if not img_root.exists():
        raise FileNotFoundError(
            f"Expected {img_root} to exist (data_root/images/<subject_id>/...). "
            f"Check configs/paths.yaml -> paths.data_root."
        )
    subdirs = sorted(
        [d for d in img_root.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )
    return subdirs


def main():
    CFG.ensure_dirs()
    exts = tuple(CFG.dataset.image_extensions)
    n_male = CFG.dataset.n_male_subjects

    subject_dirs = iter_subject_dirs(CFG.data_root)
    if len(subject_dirs) == 0:
        raise RuntimeError(f"No subject folders found under {CFG.data_root/'images'}")

    images_rows = []
    subject_counts = {}

    for sdir in tqdm(subject_dirs, desc="subjects"):
        if not sdir.name.isdigit():
            print(f"  [skip] non-numeric subject folder: {sdir.name}")
            continue
        subject_id = int(sdir.name)
        gender = subject_gender(subject_id, n_male)

        files = sorted(
            p for p in sdir.iterdir() if p.is_file() and p.suffix in exts
        )
        subject_counts[subject_id] = {"gender": gender, "n_images": len(files)}

        for fpath in files:
            try:
                with Image.open(fpath) as im:
                    w, h = im.size
            except Exception as e:
                print(f"  [warn] could not read {fpath}: {e}")
                continue
            rel_path = fpath.relative_to(CFG.data_root).as_posix()
            image_id = f"{subject_id:03d}_{fpath.name}"
            images_rows.append({
                "image_id": image_id,
                "rel_path": rel_path,          # relative to data_root ("images/01/xxx.jpg")
                "subject_id": subject_id,
                "gender": gender,
                "width": w,
                "height": h,
                "short_side": min(w, h),
            })

    if not images_rows:
        raise RuntimeError("No images found — check data_root / image_extensions in paths.yaml")

    # --- write images.csv ---
    with open(CFG.images_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(images_rows[0].keys()))
        writer.writeheader()
        writer.writerows(images_rows)

    # --- write subjects.csv ---
    with open(CFG.subjects_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "gender", "n_images"])
        writer.writeheader()
        for sid in sorted(subject_counts):
            row = subject_counts[sid]
            writer.writerow({"subject_id": sid, "gender": row["gender"], "n_images": row["n_images"]})

    n_male_img = sum(1 for r in images_rows if r["gender"] == "male")
    n_female_img = sum(1 for r in images_rows if r["gender"] == "female")
    n_male_subj = sum(1 for v in subject_counts.values() if v["gender"] == "male")
    n_female_subj = sum(1 for v in subject_counts.values() if v["gender"] == "female")

    print(f"\nWrote {CFG.images_csv}  ({len(images_rows)} images)")
    print(f"Wrote {CFG.subjects_csv} ({len(subject_counts)} subjects)")
    print(f"  male:   {n_male_subj} subjects, {n_male_img} images")
    print(f"  female: {n_female_subj} subjects, {n_female_img} images")
    print("Sanity check against the paper: 98 male subjects / 66 female subjects, "
          "~17,571 male images / ~10,841 female images, 28,412 total.")


if __name__ == "__main__":
    main()
