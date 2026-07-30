"""
Scan the raw EarVN1.0 folder and build:

  data/meta/images.csv    : image_id, rel_path, subject_id, gender, width, height, short_side
  data/meta/subjects.csv  : subject_id, gender, n_images

Handles two on-disk layouts automatically:
  (a) <data_root>/images/<subject_folder>/*.jpg   (an "images" subdir)
  (b) <data_root>/<subject_folder>/*.jpg           (subject folders directly
      under data_root -- e.g. "001.ALI_HD", "002.LeDuong_BL", ...)

Subject folder NAMES can be almost anything ("01", "001_", "001.ALI_HD",
...) -- we only require that the folder name STARTS with the subject's
number (optionally zero-padded). The numeric prefix is extracted with a
regex; everything after it (".ALI_HD", "_", ...) is ignored for id
purposes but the exact on-disk name is preserved in rel_path.

Gender rule (per EarVN1.0's own documentation): subjects numbered 1-98
are male, 99-164 are female. VERIFY this against the readme shipped with
your Mendeley download before trusting it blindly -- if your copy orders
subjects differently, adjust `subject_gender()` below.

Usage:
    python -m src.data.build_metadata
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.config import CFG

# Matches a leading run of digits, e.g. "001" in "001.ALI_HD" or "01" in "01".
_SUBJECT_ID_RE = re.compile(r"^0*(\d+)")


def subject_gender(subject_id: int, n_male: int) -> str:
    return "male" if subject_id <= n_male else "female"


def parse_subject_id(folder_name: str) -> int | None:
    m = _SUBJECT_ID_RE.match(folder_name)
    if not m:
        return None
    return int(m.group(1))


def find_images_root(data_root: Path) -> Path:
    """Auto-detect layout (a) vs (b) described in the module docstring."""
    candidate = data_root / "images"
    if candidate.is_dir():
        return candidate
    return data_root


def iter_subject_dirs(images_root: Path):
    if not images_root.exists():
        raise FileNotFoundError(
            f"{images_root} does not exist. Check configs/paths.yaml -> paths.data_root "
            f"(it should point at the folder that directly contains your subject "
            f"folders, or a parent that contains an 'images/' subfolder of them)."
        )
    subdirs = [d for d in images_root.iterdir() if d.is_dir()]
    parsed = [(d, parse_subject_id(d.name)) for d in subdirs]
    unparsed = [d.name for d, sid in parsed if sid is None]
    if unparsed:
        print(f"  [skip] {len(unparsed)} folder(s) with no leading number, "
              f"e.g. {unparsed[:5]}")
    parsed = [(d, sid) for d, sid in parsed if sid is not None]
    parsed.sort(key=lambda t: t[1])
    return parsed


def main():
    CFG.ensure_dirs()
    exts = tuple(CFG.dataset.image_extensions)
    n_male = CFG.dataset.n_male_subjects

    images_root = find_images_root(CFG.data_root)
    print(f"Using images root: {images_root}")
    subject_dirs = iter_subject_dirs(images_root)
    if len(subject_dirs) == 0:
        raise RuntimeError(f"No subject folders found under {images_root}")

    images_rows = []
    subject_counts = {}

    for sdir, subject_id in tqdm(subject_dirs, desc="subjects"):
        gender = subject_gender(subject_id, n_male)

        files = sorted(
            p for p in sdir.iterdir() if p.is_file() and p.suffix in exts
        )
        subject_counts[subject_id] = {
            "gender": gender, "n_images": len(files), "folder_name": sdir.name,
        }

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
                "rel_path": rel_path,          # relative to data_root, exact on-disk path
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
        writer = csv.DictWriter(f, fieldnames=["subject_id", "gender", "n_images", "folder_name"])
        writer.writeheader()
        for sid in sorted(subject_counts):
            row = subject_counts[sid]
            writer.writerow({"subject_id": sid, "gender": row["gender"],
                              "n_images": row["n_images"], "folder_name": row["folder_name"]})

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
    print("If these don't match, double-check that subject numbering in your "
          "download really goes 1-98=male, 99-164=female (see subjects.csv's "
          "folder_name column to inspect).")


if __name__ == "__main__":
    main()
