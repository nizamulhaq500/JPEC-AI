"""Configuration package.

    from jpegai.config import load_config
    cfg = load_config("tierA")          # development, reduced width
    cfg = load_config("full")           # paper widths, for the cloud run

Inspect a config from the shell:

    python -m jpegai.config.loader tierA
    python -m jpegai.config.loader full --dump
    python -m jpegai.config.loader tierA --set train.batch=4
"""

from .loader import CONFIG_DIR, PROJECT_ROOT, AttrDict, apply_overrides, load_config

__all__ = ["load_config", "apply_overrides", "AttrDict", "CONFIG_DIR", "PROJECT_ROOT"]
