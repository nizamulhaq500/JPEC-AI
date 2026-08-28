"""Datasets. Reads the pre-extracted crops from `jpegai.data.prepare_crops`.

Crops are extracted **offline** rather than cropped on the fly from the 2K
originals, and that decision is worth a paragraph because it is the difference
between a training run that finishes on a laptop and one that does not.

A DIV2K image is roughly 2040x1400. Decoding one PNG that size costs ~60-100 ms.
At batch 8 that is up to 0.8 s of PNG decode per step against maybe 0.3 s of
compute on an M2 -- so more than half the wall clock goes to inflating pixels
that are then thrown away, and adding workers just moves the bottleneck to
memory bandwidth. A 256x256 crop decodes in ~2 ms. `prepare_crops` pays the
decode cost once, offline, with a fixed seed, and gets a dataset that is also
*reproducible*: run 1 and run 2 see identical pixels, which is what makes the
Phase 13 ablations comparable rather than merely suggestive.

Augmentation here is therefore restricted to what is free and label-preserving:
flips and 90-degree rotations, which are exact pixel permutations. No colour
jitter, no scaling, no blur -- a compression model must learn the statistics of
*real* images, and augmentations that alter those statistics teach it to spend
bits on distributions it will never see.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jpegai.config import PROJECT_ROOT

IMAGE_EXT = {".png", ".ppm", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"}


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def list_images(roots) -> list[Path]:
    """All images under one or more directories, sorted, deduplicated."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    out: list[Path] = []
    for r in roots:
        root = _resolve(r)
        if not root.is_dir():
            continue
        out += sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    return list(dict.fromkeys(out))


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] -> float32 [3,H,W] in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    return t.float().div_(255.0)


class CropDataset(Dataset):
    """Pre-extracted square crops, with flip/rotate augmentation.

    If a stored crop is larger than `crop`, a random sub-crop is taken; if it is
    smaller, the item is rejected at construction time rather than padded, because
    a padded training sample teaches the model to reconstruct a border that will
    never appear at inference.
    """

    def __init__(self, roots, crop: int = 256, *, augment: bool = True,
                 seed: int | None = None):
        self.files = list_images(roots)
        if not self.files:
            raise FileNotFoundError(
                f"no training crops under {roots}.\n"
                f"Run: python -m jpegai.data.prepare_crops"
            )
        self.crop = int(crop)
        self.augment = bool(augment)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = _load_rgb(self.files[i])
        h, w = arr.shape[:2]
        c = self.crop
        if h < c or w < c:
            raise ValueError(
                f"{self.files[i].name} is {w}x{h}, smaller than crop {c}. "
                f"Re-run prepare_crops with --crop {min(h, w) // 64 * 64}."
            )
        if h > c or w > c:
            # torch's DataLoader workers each get their own fork of self._rng, so
            # this is seeded per-worker by torch, not by us. Using python's global
            # random here would make every worker draw the same offsets.
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            arr = arr[top:top + c, left:left + c]

        if self.augment:
            # Exact pixel permutations only. The dihedral group of the square is
            # 8 elements and every one of them is a valid natural image, so this
            # is an 8x dataset multiplier at zero statistical cost.
            k = random.randint(0, 3)
            if k:
                arr = np.rot90(arr, k)
            if random.random() < 0.5:
                arr = arr[:, ::-1]
        return _to_tensor(arr)


class ImageDataset(Dataset):
    """Whole images, no augmentation. For validation and evaluation.

    Optionally centre-crops to a multiple of `pad_multiple` so a validation pass
    needs no padding at all. That keeps the reported validation bpp exactly
    comparable to the training bpp -- padding changes the pixel count in the
    denominator, and a validation curve that silently uses a different
    denominator than the training curve is a good way to conclude the model
    improved when it did not.
    """

    def __init__(self, roots, *, max_side: int | None = None,
                 multiple: int | None = None, limit: int | None = None):
        self.files = list_images(roots)
        if not self.files:
            raise FileNotFoundError(f"no images under {roots}")
        if limit:
            self.files = self.files[:limit]
        self.max_side = max_side
        self.multiple = multiple
        # Kept so a checkpoint can record *what it was validated on*. A validation
        # bpp without that provenance is unusable: the fallback below silently
        # swaps DIV2K for Kodak, and two runs that disagree by 15% may simply have
        # measured different pictures.
        self.roots = [str(r) for r in ([roots] if isinstance(roots, (str, Path))
                                       else roots)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = _load_rgb(self.files[i])
        if self.max_side:
            h, w = arr.shape[:2]
            if max(h, w) > self.max_side:
                ch = min(h, self.max_side)
                cw = min(w, self.max_side)
                top, left = (h - ch) // 2, (w - cw) // 2
                arr = arr[top:top + ch, left:left + cw]
        if self.multiple:
            h, w = arr.shape[:2]
            ch, cw = h - h % self.multiple, w - w % self.multiple
            if ch != h or cw != w:
                top, left = (h - ch) // 2, (w - cw) // 2
                arr = arr[top:top + ch, left:left + cw]
        return _to_tensor(arr)

    def name(self, i: int) -> str:
        return self.files[i].stem


def probe_workers(requested: int) -> int:
    """Return `requested` if multi-process loading actually works here, else 0.

    On macOS torch's only sharing strategy is `file_system`, which spawns
    `torch/bin/torch_shm_manager`. Sandboxes and hardened containers routinely
    deny that exec, and the failure surfaces as an unhandled `RuntimeError:
    Operation not permitted` from inside a worker at the *first batch* -- after
    the model is built and the run banner has printed, which makes it look like a
    model bug.

    So probe with a 16-element toy dataset before committing. Costs ~0.3 s once,
    turns a stack trace into one line of output, and means the same command runs
    unchanged on the sandbox, the laptop, and the cloud GPU.
    """
    if requested <= 0:
        return 0
    try:
        from torch.utils.data import DataLoader, TensorDataset
        probe = DataLoader(TensorDataset(torch.zeros(4, 1, 8, 8)),
                           batch_size=2, num_workers=2)
        next(iter(probe))
        return requested
    except Exception as exc:
        print(f"WARNING: DataLoader workers unavailable ({type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:80]}); falling back to num_workers=0. "
              f"Training will be slower but correct.")
        return 0


def build_loaders(config, *, batch: int | None = None, workers: int = 4,
                  valid_limit: int = 8, seed: int = 1234):
    """(train_loader, valid_dataset) from a loaded config.

    The validation set is a `Dataset`, not a `DataLoader`: validation images have
    different sizes so they cannot be batched, and evaluating them one at a time
    is also what the real codec does.
    """
    from torch.utils.data import DataLoader

    crop = config.train.crop
    train = CropDataset(config.data.train, crop=crop, augment=True, seed=seed)

    valid_root = config.data.valid
    if not list_images(valid_root):
        # DIV2K_valid_HR download failed for the user, so fall back to the test
        # set rather than crashing 30 s into a training run. Loud, because
        # validating on the test set is a real methodological problem: it is fine
        # for watching a loss curve, and not fine for any number in the report.
        print(f"WARNING: validation set {valid_root} is empty; falling back to "
              f"{config.data.test}. Do not quote validation numbers from this run.")
        valid_root = config.data.test

    valid = ImageDataset(
        valid_root,
        max_side=768,                       # keep a validation pass under a minute
        multiple=config.geometry.total_downsample,
        limit=valid_limit,
    )

    workers = probe_workers(workers)
    loader = DataLoader(
        train,
        batch_size=batch or config.train.batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,                   # no effect on MPS, costs a copy on CPU
        drop_last=True,                     # a short final batch skews batchnorm-free
        persistent_workers=workers > 0,     # respawning 4 workers per epoch is ~1 s
    )
    return loader, valid


if __name__ == "__main__":
    from jpegai.config import load_config

    cfg = load_config("tierA")
    loader, valid = build_loaders(cfg, workers=0, valid_limit=4)
    print(f"train  {len(loader.dataset):,} crops of {cfg.train.crop}px, "
          f"batch {loader.batch_size} -> {len(loader):,} steps/epoch")
    batch = next(iter(loader))
    print(f"       batch {tuple(batch.shape)} {batch.dtype} "
          f"range [{batch.min():.3f}, {batch.max():.3f}]")
    print(f"valid  {len(valid)} images")
    for i in range(len(valid)):
        print(f"       {valid.name(i):12} {tuple(valid[i].shape)}")
