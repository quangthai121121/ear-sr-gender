"""
Single entry point for reading configs/paths.yaml.

Every script in this repo does:

    from src.config import CFG
    print(CFG.data_root)

instead of hard-coding paths, so you only ever edit configs/paths.yaml.
"""
from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATHS_YAML = _REPO_ROOT / "configs" / "paths.yaml"


def _load_raw(path: Path = _PATHS_YAML) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Did you rename/move configs/paths.yaml?"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(p: str) -> Path:
    """Resolve a path relative to the repo root unless it's already absolute.

    IMPORTANT: this deliberately does NOT use Path.resolve() -- that
    follows symlinks, and a venv's bin/python is typically a symlink to
    the system interpreter (e.g. `python3 -m venv` on Debian/Ubuntu).
    Resolving it away breaks venv isolation: subprocess would then run
    the bare system Python, silently missing every package pip-installed
    into that venv (basicsr, realesrgan, timm, ...), and fail with
    ModuleNotFoundError deep inside the child process. Path.absolute()
    (via os.path.normpath) keeps the symlink path intact instead.
    """
    pp = Path(os.path.expanduser(p))
    if pp.is_absolute():
        return Path(os.path.normpath(str(pp)))
    combined = _REPO_ROOT / pp
    return Path(os.path.normpath(str(combined)))


class Config:
    def __init__(self, raw: dict):
        self._raw = raw
        p = raw["paths"]

        self.repo_root = _REPO_ROOT
        self.data_root = _resolve(p["data_root"])
        self.processed_root = _resolve(p["processed_root"])
        self.meta_root = _resolve(p["meta_root"])
        self.realesrgan_repo = _resolve(p["realesrgan_repo"])
        self.realesrgan_python = _resolve(p["realesrgan_python"])
        self.swinir_repo = _resolve(p["swinir_repo"])
        self.swinir_python = _resolve(p["swinir_python"])
        self.realesrgan_ckpt = _resolve(p["realesrgan_ckpt"])
        self.swinir_large_ckpt = _resolve(p["swinir_large_ckpt"])
        self.classifier_ckpt_root = _resolve(p["classifier_ckpt_root"])
        self.results_root = _resolve(p["results_root"])
        self.locked_config_path = _resolve(p["locked_config"])

        self.sr = SimpleNamespace(**raw["sr"])
        self.dataset = SimpleNamespace(**raw["dataset"])
        self.protocol_a = SimpleNamespace(**raw["protocol_a"])
        self.protocol_b = SimpleNamespace(**raw["protocol_b"])

        # Derived, commonly used paths
        self.images_csv = self.meta_root / "images.csv"
        self.subjects_csv = self.meta_root / "subjects.csv"
        self.splits_dir = self.meta_root / "splits"
        self.protocol_a_csv = self.splits_dir / "protocol_a.csv"

        def protocol_b_csv(fold: int) -> Path:
            return self.splits_dir / f"protocol_b_fold{fold}.csv"

        self.protocol_b_csv = protocol_b_csv

        self.predictions_dir = self.results_root / "predictions"
        self.train_logs_dir = self.results_root / "train_logs"

    def variant_root(self, variant: str) -> Path:
        """Root directory for a given input variant's pixel data.

        variant='orig'      -> data_root (images live here directly)
        variant='bicubic'    -> computed on the fly, no cache dir needed
        variant='realesrgan' -> processed_root/realesrgan
        variant='swinir'     -> processed_root/swinir_large
        """
        if variant == "orig":
            return self.data_root
        if variant == "bicubic":
            return None  # computed on the fly
        if variant == "realesrgan":
            return self.processed_root / "realesrgan"
        if variant == "swinir":
            return self.processed_root / "swinir_large"
        raise ValueError(f"Unknown variant: {variant}")

    def ensure_dirs(self):
        for d in [
            self.meta_root, self.splits_dir, self.classifier_ckpt_root,
            self.results_root, self.predictions_dir, self.train_logs_dir,
            self.processed_root / "realesrgan", self.processed_root / "swinir_large",
        ]:
            d.mkdir(parents=True, exist_ok=True)


CFG = Config(_load_raw())

if __name__ == "__main__":
    # quick sanity check: `python -m src.config`
    for k, v in vars(CFG).items():
        if not k.startswith("_"):
            print(f"{k:24s} = {v}")
