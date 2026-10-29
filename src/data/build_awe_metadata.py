"""
Build metadata for the AWE cross-dataset evaluation (Annotated Web Ears:
100 subject folders 001..100, ten PNG ear crops each, subject-level
gender in <folder>/annotations.json as {"gender": "m"|"f", ...}).

Writes, under the AWE config's meta_root:
  images.csv               same schema as EarVN1.0's images.csv
  splits/awe_test.csv      every image in split "test" (AWE is never trained on)

Usage:
    EAR_SR_CONFIG=configs/paths_awe.yaml python -m src.data.build_awe_metadata
"""
from __future__ import annotations
import json

import pandas as pd
from PIL import Image

from src.config import CFG

GENDER = {"m": "male", "f": "female"}


def main():
    CFG.ensure_dirs()
    root = CFG.data_root
    subject_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    if not subject_dirs:
        raise RuntimeError(f"No numeric subject folders under {root}; check paths_awe.yaml data_root")

    rows = []
    for sdir in subject_dirs:
        ann = json.loads((sdir / "annotations.json").read_text())
        gender = GENDER[ann["gender"]]
        for img in sorted(sdir.glob("*.png")):
            with Image.open(img) as im:
                w, h = im.size
            rows.append({"image_id": f"awe_{sdir.name}_{img.stem}",
                         "rel_path": img.relative_to(root).as_posix(),
                         "subject_id": int(sdir.name), "gender": gender,
                         "width": w, "height": h, "short_side": min(w, h)})

    images = pd.DataFrame(rows)
    images.to_csv(CFG.images_csv, index=False)
    split = images[["image_id", "subject_id", "gender"]].assign(split="test")
    split_csv = CFG.splits_dir / "awe_test.csv"
    split.to_csv(split_csv, index=False)

    subj = images.drop_duplicates("subject_id")["gender"].value_counts().to_dict()
    print(f"[awe] {len(images)} images, {images['subject_id'].nunique()} subjects, "
          f"subjects by gender {subj}, images by gender {images['gender'].value_counts().to_dict()}")
    print(f"[awe] -> {CFG.images_csv}\n[awe] -> {split_csv}")


if __name__ == "__main__":
    main()
