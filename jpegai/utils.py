"""Small shared helpers: device selection, seeding, parameter counting.

The device probe is the reason this file exists. `torch.backends.mps.is_available()`
returns True on this machine and then the first allocation raises

    RuntimeError: The MPS backend is supported on macOS 14.0+

even under Darwin 25.5 (macOS 26), because the sandbox this project is developed in
does not expose Metal. A codec that dies three hours into a training run because
the *first* tensor allocation failed is a bad way to find that out, so the probe
allocates and matmuls a real tensor before promising the device works.
"""

from __future__ import annotations

import os
import random

import torch


_DEVICE_CACHE: dict[str, torch.device] = {}


def _works(name: str) -> bool:
    """Allocate and use a tensor on `name`. True only if that fully succeeds."""
    try:
        d = torch.device(name)
        a = torch.ones(8, 8, device=d)
        # A matmul, not just an allocation: MPS can allocate and then fail to
        # dispatch a kernel, and conv/matmul is what training actually needs.
        b = (a @ a).sum().item()
        return b == 512.0
    except Exception:
        return False


def pick_device(prefer: str | None = None, *, verbose: bool = False) -> torch.device:
    """Best usable device: explicit `prefer`, else cuda, else mps, else cpu.

    `prefer` is honoured only if it actually works; a typo'd or unavailable
    device falls back with a warning rather than crashing, because the same
    command has to run on the Mac and on the cloud GPU.
    """
    key = prefer or "auto"
    if key in _DEVICE_CACHE:
        return _DEVICE_CACHE[key]

    order = [prefer] if prefer else []
    order += ["cuda", "mps", "cpu"]

    for name in order:
        if not name:
            continue
        if name.startswith("cuda") and not torch.cuda.is_available():
            continue
        if name == "mps" and not torch.backends.mps.is_built():
            continue
        if name == "cpu" or _works(name):
            dev = torch.device(name)
            if verbose and prefer and name != prefer:
                print(f"device {prefer!r} unusable; falling back to {name!r}")
            _DEVICE_CACHE[key] = dev
            return dev

    _DEVICE_CACHE[key] = torch.device("cpu")
    return _DEVICE_CACHE[key]


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        i = dev.index or 0
        p = torch.cuda.get_device_properties(i)
        return f"cuda:{i} {p.name} ({p.total_memory / 2**30:.0f} GiB)"
    if dev.type == "mps":
        return "mps (Apple Metal)"
    return f"cpu ({os.cpu_count()} cores)"


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch. Not a guarantee of bit-reproducibility --
    cuDNN and MPS kernel selection are still free to vary -- but it makes runs
    comparable, which is what the ablations in Phase 13 need."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = module.parameters()
    return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.1f} GiB"


def macs_breakdown(module: torch.nn.Module, input_shape=(1, 3, 256, 256),
                   parts=None) -> dict[str, float]:
    """MAC/pixel of the *image*, split by submodule. Includes "TOTAL".

    Counts Conv2d and ConvTranspose2d only, which is where >98% of a codec's
    arithmetic lives. This is the number the paper reports as kMAC/pxl (Table III:
    SOP/BOP/HOP decoders at roughly 8 / 28 / 215), and having it measurable from
    day one is what lets Phase 7 design three heads to a budget rather than to
    taste.

    Two convention points, both of which the paper's numbers depend on:

    * MACs are attributed per pixel of the *original image*, not per pixel of each
      layer's own output. That is what makes a stride-2 conv four times cheaper
      than the same conv at full resolution.
    * The paper's figures are **decoder-side**. Comparing a whole-model total
      against 215 kMAC/pxl is comparing the wrong things, which is why this
      returns a breakdown instead of one number: for us the decoder is
      ``g_s + h_s``, and the encoder-only cost of ``g_a + h_a`` never runs on the
      device that has to hit the budget.

    `parts` is an optional `[(label, submodule)]` list overriding the default
    "attribute to the top-level child" rule. Phase 4's two-branch model needs it:
    its hyper networks live *inside* `branch_y`/`branch_uv`, so the default rule
    would bill `h_a` and `h_s` to one bucket and there would be no way to separate
    the encoder-only half -- silently inflating the decoder figure this function
    exists to compare against the paper's.
    """
    per_module: dict[torch.nn.Module, float] = {}
    handles = []

    def hook(mod, inp, out):
        oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
        kh, kw = mod.kernel_size
        ic = mod.in_channels // mod.groups
        per_module[mod] = per_module.get(mod, 0.0) + float(oc * oh * ow * kh * kw * ic)

    leaves = [m for m in module.modules()
              if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    for m in leaves:
        handles.append(m.register_forward_hook(hook))

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(input_shape)
            p = next(module.parameters(), None)
            if p is not None:
                x = x.to(p.device)
            module(x)
    finally:
        for h in handles:
            h.remove()
        module.train(was_training)

    # Attribute each counted leaf to a bucket. With `parts` given, that is the
    # listed submodule that owns it; otherwise the top-level child, which is
    # enough for the single-branch models because they are flat at the top:
    # g_a / g_s / h_a / h_s / entropy models.
    owner: dict[torch.nn.Module, str] = {}
    if parts:
        for label, sub in parts:
            for leaf in sub.modules():
                owner.setdefault(leaf, label)
    else:
        for name, child in module.named_modules():
            if not name:
                continue
            top = name.split(".")[0]
            for leaf in child.modules():
                owner.setdefault(leaf, top)

    npix = input_shape[-1] * input_shape[-2]
    out: dict[str, float] = {}
    for mod, macs in per_module.items():
        out[owner.get(mod, "?")] = out.get(owner.get(mod, "?"), 0.0) + macs / npix
    out["TOTAL"] = sum(v for k, v in out.items() if k != "TOTAL")
    return out


def macs_per_pixel(module: torch.nn.Module, input_shape=(1, 3, 256, 256)) -> float:
    """Total MAC/pixel. See :func:`macs_breakdown` for the conventions."""
    return macs_breakdown(module, input_shape)["TOTAL"]
