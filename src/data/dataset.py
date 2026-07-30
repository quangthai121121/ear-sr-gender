"""
EarGenderDataset implements Eq. (1)-(4) of the paper draft:

    xOrig = PadResize224(x)
    xBic  = PadResize224(Bicubic_x4(x))
    xRES  = PadResize224(RealESRGAN_x4(x))     # precomputed offline
    xSwin = PadResize224(SwinIR_x4(x))         # precomputed offline

`variant` controls which pixels are loaded:
  - "orig":       read directly from data_root, no upsampling
  - "bicubic":    read from data_root, upsample x4 with PIL bicubic on the fly
  - "realesrgan": read the precomputed x4 PNG from processed_root/realesrgan
  - "swinir":     read the precomputed x4 PNG from processed_root/swinir_large

PadResize224 = aspect-ratio-preserving resize so the longer side is 224,
then zero-pad the shorter side to 224 (letterbox), then ImageNet normalize.
This final geometry step is IDENTICAL across variants, which is what makes
same-checkpoint evaluation across variants apples-to-apples.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.config import CFG

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GENDER_TO_IDX = {"male": 0, "female": 1}
IDX_TO_GENDER = {v: k for k, v in GENDER_TO_IDX.items()}


def pad_resize_224(img: Image.Image, size: int = 224) -> Image.Image:
    """Aspect-ratio preserving resize to `size` on the long side, then
    zero-pad (black) the short side to make a size x size square."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(img, (left, top))
    return canvas


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t


class EarGenderDataset(Dataset):
    """
    Args:
        split_csv: path to a protocol_a.csv or protocol_b_fold{k}.csv
                   (columns: image_id, subject_id, gender, split)
        images_csv: path to images.csv (columns include image_id, rel_path)
        split_name: "train" | "val" | "test"
        variant: "orig" | "bicubic" | "realesrgan" | "swinir"
        return_meta: if True, __getitem__ also returns (image_id, short_side)
                     for flip/native-size analysis.
    """

    def __init__(self, split_csv, images_csv=None, split_name="train",
                 variant="orig", return_meta=False):
        images_csv = images_csv or CFG.images_csv
        images_df = pd.read_csv(images_csv)
        split_df = pd.read_csv(split_csv)

        merged = split_df.merge(images_df[["image_id", "rel_path", "short_side"]],
                                 on="image_id", how="left")
        if merged["rel_path"].isna().any():
            missing = merged[merged["rel_path"].isna()]["image_id"].tolist()[:5]
            raise ValueError(f"{len(missing)}+ image_ids in split not found in images.csv, "
                              f"e.g. {missing}. Did you regenerate images.csv after splitting?")

        self.df = merged[merged["split"] == split_name].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows with split == '{split_name}' in {split_csv}")

        self.variant = variant
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> Path:
        if self.variant == "orig" or self.variant == "bicubic":
            return CFG.data_root / rel_path
        elif self.variant in ("realesrgan", "swinir"):
            root = CFG.variant_root(self.variant)
            # cached SR outputs are saved as .png regardless of original ext
            return root / Path(rel_path).with_suffix(".png")
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve_path(row["rel_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pixel data for variant='{self.variant}': {path}\n"
                f"If variant is 'realesrgan' or 'swinir', did you run "
                f"scripts/05_precompute_sr.sh yet?"
            )

        img = Image.open(path).convert("RGB")

        if self.variant == "bicubic":
            w, h = img.size
            img = img.resize((w * 4, h * 4), Image.BICUBIC)
        # realesrgan/swinir are already x4 on disk; orig needs no upsampling.

        img = pad_resize_224(img)
        x = to_normalized_tensor(img)
        y = GENDER_TO_IDX[row["gender"]]

        if self.return_meta:
            return x, y, row["image_id"], int(row["short_side"])
        return x, y


def make_loader(split_csv, split_name, variant, batch_size=32, shuffle=False,
                 num_workers=4, return_meta=False):
    ds = EarGenderDataset(split_csv, split_name=split_name, variant=variant,
                           return_meta=return_meta)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
