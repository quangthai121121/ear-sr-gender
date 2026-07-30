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

CONDITIONAL SKIP (configs/paths.yaml -> dataset.skip_upsample_if_native_long_side_at_least,
DEFAULT: null / DISABLED):
For any image whose native long side is already >= this threshold,
PadResize224 would downscale it anyway -- there is no missing detail for
SR/bicubic to add, and applying the explicit x4 step first only adds a
redundant extra resize pass. We measured this compounding-resize artifact
empirically: negligible for genuinely tiny crops (<0.5% sharpness loss
around native short-side 16-45px) but growing substantial for larger crops
(~23% sharpness loss at native short-side ~140px) -- and critically, it is
NOT symmetric across variants (Original does one resize pass; Bicubic/SR
do two), so it can bias Original-vs-SR comparisons for larger crops.

IMPORTANT: this is disabled by default and should stay that way for the
MAIN benchmark (Table 3 / Figure 2). Enabling it forces Delta_SR=0 by
construction for every image at/above the threshold -- which would turn
"does SR help large crops?" from an empirical question Figure 2 is meant
to answer into a circular assumption baked into the data pipeline, and it
silently swaps the studied condition from Eq. 1-4's unconditional SR
preprocessing into a different, size-adaptive pipeline. Only enable this
for a separately-reported robustness/sensitivity re-run (different
results/ output dir), to check whether a measured Delta_SR at large native
sizes survives once the redundant-resize confound is removed -- report it
alongside the main table, never as a silent replacement for it.
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

        merged = split_df.merge(
            images_df[["image_id", "rel_path", "short_side", "width", "height"]],
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
        # None disables the skip entirely (always apply the explicit x4 step).
        self.skip_threshold = CFG.dataset.skip_upsample_if_native_long_side_at_least

    def __len__(self):
        return len(self.df)

    def _effective_variant(self, row) -> str:
        """Returns the variant to actually USE for this specific image.
        Falls back to 'orig' for bicubic/realesrgan/swinir when the native
        long side is already >= skip_threshold (see module docstring) --
        PadResize224 would downscale such images anyway, so there is no
        missing detail for SR/bicubic to add, and skipping avoids a
        redundant extra resize pass that we measured to bias Original-vs-SR
        comparisons for larger crops."""
        if self.variant == "orig" or self.skip_threshold is None:
            return self.variant
        native_long_side = max(row["width"], row["height"])
        if native_long_side >= self.skip_threshold:
            return "orig"
        return self.variant

    def _resolve_path(self, rel_path: str, effective_variant: str) -> Path:
        if effective_variant == "orig" or effective_variant == "bicubic":
            return CFG.data_root / rel_path
        elif effective_variant in ("realesrgan", "swinir"):
            root = CFG.variant_root(effective_variant)
            # cached SR outputs are saved as .png regardless of original ext
            return root / Path(rel_path).with_suffix(".png")
        else:
            raise ValueError(f"Unknown variant: {effective_variant}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        effective_variant = self._effective_variant(row)
        path = self._resolve_path(row["rel_path"], effective_variant)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pixel data for variant='{effective_variant}': {path}\n"
                f"If variant is 'realesrgan' or 'swinir', did you run "
                f"scripts/05_precompute_sr.sh yet?"
            )

        img = Image.open(path).convert("RGB")

        if effective_variant == "bicubic":
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
        # Keep worker processes alive across epochs instead of respawning
        # them every time (only valid when num_workers > 0). Saves
        # repeated process-startup + dataset-repickling overhead per epoch.
        persistent_workers=(num_workers > 0),
    )
