"""YAML configuration loading with single-inheritance deep merge.

    from jpegai.config import load_config
    cfg = load_config("tierA")
    cfg.channels.primary_latent        # 96
    cfg["channels"]["primary_latent"]  # same thing

A config may declare `_base: <other>.yaml`. The base is loaded first (recursively)
and this file's keys are deep-merged over it, so `full.yaml` only has to state
what differs from `tierA.yaml`.

Merge rules, chosen deliberately:

* **dict + dict -> recursive merge.** So `channels: {primary_latent: 160}` in a
  derived config keeps every other key under `channels`.
* **anything else -> wholesale replacement.** In particular **lists are replaced,
  never merged element-wise.** `decoders:` is a list of dicts; element-wise
  merging would silently combine an override's decoder 0 with the base's decoder 0
  and produce a configuration that appears in no file. Replacement is loud: if you
  override `decoders`, you own the whole list.

Every loaded config records its own provenance in `_files`, so a training log can
state exactly which files produced the run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent.parent

__all__ = ["AttrDict", "load_config", "apply_overrides", "CONFIG_DIR", "PROJECT_ROOT"]


class AttrDict(dict):
    """dict with attribute access, applied recursively on construction.

    Exists so model code reads `cfg.channels.primary_latent` instead of
    `cfg["channels"]["primary_latent"]` forty times. Still a plain dict, so yaml
    dumping, `**` expansion and `in` all behave normally.

    A missing key raises AttributeError naming the key and listing what *is*
    present -- a typo'd config key should be obvious, not a silent None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = _wrap(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"config has no key {name!r}. Available: {sorted(self.keys())}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))


def _wrap(v: Any) -> Any:
    if isinstance(v, AttrDict):
        return v
    if isinstance(v, dict):
        return AttrDict(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into `base`. See module docstring for the rules."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(name_or_path: str | Path) -> Path:
    """Accept 'tierA', 'tierA.yaml', or any path. Search CONFIG_DIR for bare names."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        if p.is_absolute() or p.exists():
            return p.resolve()
        cand = CONFIG_DIR / p.name
        if cand.exists():
            return cand.resolve()
        raise FileNotFoundError(f"config not found: {name_or_path}")

    for ext in (".yaml", ".yml"):
        cand = CONFIG_DIR / f"{p.name}{ext}"
        if cand.exists():
            return cand.resolve()

    available = sorted(f.stem for f in CONFIG_DIR.glob("*.y*ml"))
    raise FileNotFoundError(
        f"config {name_or_path!r} not found in {CONFIG_DIR}. Available: {available}"
    )


def _load_raw(path: Path, seen: list[Path]) -> tuple[dict, list[Path]]:
    if path in seen:
        chain = " -> ".join(p.name for p in seen + [path])
        raise ValueError(f"circular _base chain: {chain}")
    seen = seen + [path]

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")

    base_ref = data.pop("_base", None)
    if base_ref is None:
        return data, seen

    # `_base` is resolved relative to the referring file's directory, so a config
    # copied elsewhere with its base still works.
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        local = (path.parent / base_path).resolve()
        base_path = local if local.exists() else _resolve(base_ref)
    else:
        base_path = base_path.resolve()

    base, seen = _load_raw(base_path, seen)
    return _deep_merge(base, data), seen


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` strings in place. Values are parsed as YAML.

    So `--set train.batch=4` gives an int, `--set tools.rvs.enable=false` gives a
    bool, and `--set rate.beta_eval_points=[0,16]` gives a list. Quoting a number
    (`='4'`) gives a string, as YAML dictates.

    Overriding a key that does not already exist is an error, not a silent
    addition: `--set trian.batch=4` should fail loudly rather than create a
    `trian` section that nothing reads. This has caught more of my mistakes than
    any other check in this file.
    """
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        parts = [p for p in dotted.strip().split(".") if p]
        if not parts:
            raise ValueError(f"empty key path in override {item!r}")

        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"override {item!r}: {p!r} is not an existing section")
            node = node[p]
        leaf = parts[-1]
        if leaf not in node:
            raise KeyError(
                f"override {item!r}: no existing key {leaf!r} in that section "
                f"(has {sorted(node.keys())}). Refusing to invent config keys."
            )
        node[leaf] = _wrap(yaml.safe_load(raw))
    return cfg


def load_config(
    name_or_path: str | Path = "tierA",
    *,
    overrides: list[str] | None = None,
    resolve_paths: bool = True,
) -> AttrDict:
    """Load a config, following `_base`, then apply `--set` style overrides.

    `resolve_paths=True` turns the relative paths under `data:` into absolute
    paths anchored at the project root, so scripts work from any cwd.
    """
    path = _resolve(name_or_path)
    raw, seen = _load_raw(path, [])
    cfg = AttrDict(raw)
    apply_overrides(cfg, overrides)

    cfg["_files"] = [str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT)
                     else str(p) for p in reversed(seen)]
    cfg["_root"] = str(PROJECT_ROOT)

    if resolve_paths and "data" in cfg:
        for key, val in list(cfg["data"].items()):
            if isinstance(val, str):
                cfg["data"][key] = str(PROJECT_ROOT / val)
            elif isinstance(val, list):
                cfg["data"][key] = [str(PROJECT_ROOT / v) for v in val]
    return cfg


def _summary(cfg: AttrDict) -> str:
    ch, geo = cfg.channels, cfg.geometry
    dec = ", ".join(
        f"{d['name']}({d['final_channels']})"
        for d in cfg.decoders
        if d.get("enabled", True)
    )
    tools = ", ".join(k for k, v in cfg.tools.items() if v.get("enable"))
    return "\n".join([
        f"config      {cfg.name}   (from {' <- '.join(cfg['_files'])})",
        f"latent      primary {ch.primary_latent}  secondary {ch.secondary_latent}"
        f"  hyper {ch.hyper_latent}/{ch.hyper_secondary_latent}",
        f"geometry    /{2 ** geo.analysis_stages} latent, /{geo.total_downsample} hyper,"
        f" pad multiple {geo.total_downsample}",
        f"entropy     {cfg.entropy.mcm_stages}-stage MCM, {cfg.entropy.sigma_quant_level}"
        f" sigma classes, coder={cfg.entropy.coder}",
        f"decoders    {dec}",
        f"tools       {tools}",
        f"rate        variable={cfg.rate.variable_rate}, beta {cfg.rate.beta_range},"
        f" eval at {cfg.rate.beta_eval_points}",
        f"train       {cfg.train.crop}px x{cfg.train.batch}, {cfg.train.iterations:,} it,"
        f" lr {cfg.train.lr}",
    ])


def main(argv: list[str] | None = None) -> None:
    """CLI: `python -m jpegai.config [name] [--set k=v] [--dump]`."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m jpegai.config",
                                 description="inspect a jpegai config")
    ap.add_argument("config", nargs="?", default="tierA")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. --set train.batch=4")
    ap.add_argument("--dump", action="store_true", help="print the fully merged YAML")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    print(_summary(cfg))

    if args.dump:
        print("\n--- fully merged ---")
        plain = yaml.safe_load(yaml.safe_dump(dict(cfg), default_flow_style=False))
        print(yaml.safe_dump(plain, sort_keys=False, default_flow_style=False))

    # Cross-check the two shipped configs against invariants that must hold for
    # the architecture to be JPEG AI at all. Cheap, and catches an edit that
    # breaks eq. (3) or the pixel-shuffle arithmetic months before training does.
    if args.config == "tierA":
        print("\ninvariant checks")
        for name in ("tierA", "full"):
            c = load_config(name)
            ch = c.channels
            checks = [
                ("analysis_stages == 4",
                 c.geometry.analysis_stages == 4),
                ("total_downsample == 2^(analysis+hyper)",
                 c.geometry.total_downsample == 2 ** (c.geometry.analysis_stages
                                                      + c.geometry.hyper_stages)),
                ("mcm_stages == 4",
                 c.entropy.mcm_stages == 4),
                ("mcm_group_order covers the 2x2 tile exactly once",
                 sorted(tuple(g) for g in c.entropy.mcm_group_order)
                 == [(0, 0), (0, 1), (1, 0), (1, 1)]),
                ("pred_primary_preshuffle == 4 * primary_latent  (x2 pixel shuffle)",
                 ch.pred_primary_preshuffle == 4 * ch.primary_latent),
                ("pred_secondary_preshuffle == 4 * secondary_latent",
                 ch.pred_secondary_preshuffle == 4 * ch.secondary_latent),
                ("secondary_synthesis_in == secondary + primary  (eq. 3)",
                 ch.secondary_synthesis_in == ch.secondary_latent + ch.primary_latent),
                # The next three come from the WG1 reference software, not the
                # paper. Each one, if violated, produces a model that builds and
                # trains but is not JPEG AI -- see docs/06-normative-constants.md.
                #
                # MCM_phases.py chs2group() asserts chs % 32 == 0 outright. MCM is
                # luma-only, so this binds the primary latent only.
                ("primary_latent % 32 == 0  (MCM_phases.chs2group assert)",
                 ch.primary_latent % 32 == 0),
                # The hyper autoencoder is channel-preserving and is constructed
                # with chs=chs_ls of its own branch (common_modules.py:116-128),
                # so the hyper latent width is not a free parameter.
                ("hyper_latent == primary_latent  (channel-preserving hyper AE)",
                 ch.hyper_latent == ch.primary_latent),
                ("hyper_secondary_latent == secondary_latent",
                 ch.hyper_secondary_latent == ch.secondary_latent),
                # sigma_idx_max_value = (level-1) * 2^precision - 1, so the table
                # spans [0, 3967] -> 3968 entries. Ties three constants together.
                ("isigma_table_size == (sigma_quant_level-1) * 2^sigma_precision",
                 c.entropy.isigma_table_size
                 == (c.entropy.sigma_quant_level - 1) * 2 ** c.entropy.sigma_precision),
                # scaled_sigma_precision is hardcoded to 17 in lsbs_scale_mode.py.
                ("scaler_precision == gain_vector_precision + beta_displacement_precision",
                 c.entropy.scaler_precision
                 == c.rate.gain_vector_precision + c.rate.beta_displacement_precision),
                ("hyper_max_symbol == z_range - 1",
                 c.entropy.hyper_max_symbol == c.entropy.z_range - 1),
                ("the 18-entry beta ladder contains all four base-model betas",
                 all(b in c.rate.beta_list
                     for b in (m["beta_train"] for m in c.rate.models))),
                # Phase 8. Eq. (10)'s P_beta is 2^sigma_precision and its S_sigma is
                # the sigma grid's log step, which is what makes Delta_beta an
                # additive offset on Isigma rather than a second quantiser. If these
                # two ever drift apart the gain unit still runs -- encoder and decoder
                # share the constant either way -- but sigma' stops being delta_beta
                # times sigma, and every rate request lands somewhere else.
                ("p_beta_precision == sigma_precision  (eq. 10's P_beta)",
                 c.rate.p_beta_precision == c.entropy.sigma_precision),
                # The clamp is written down twice, from two different files of the
                # reference software. Two copies that can disagree is one copy too
                # many, so the loader makes them agree or fails.
                ("beta_range == bdl_clipping_range",
                 list(c.rate.beta_range) == list(c.entropy.bdl_clipping_range)),
                ("the anchor is on the eval ladder and the ladder is inside the clamp",
                 0 in c.rate.beta_eval_points
                 and min(c.rate.beta_eval_points) >= c.rate.beta_range[0]
                 and max(c.rate.beta_eval_points) <= c.rate.beta_range[1]),
                ("the training sample range is inside the clamp",
                 c.rate.beta_train_sample[0] >= c.rate.beta_range[0]
                 and c.rate.beta_train_sample[1] <= c.rate.beta_range[1]),
                # Table I. 17 entries, and index 0 must be scale 1.0 -- that is the
                # entry the anchor uses, so if it were anything else a picture with
                # no quality map would be coded at the wrong rate.
                ("Table I has 17 entries spanning q_index_range, with 1.0 at index 0",
                 len(c.rate.q_scale_table) == 17
                 == c.rate.q_index_range[1] - c.rate.q_index_range[0] + 1
                 and c.rate.q_scale_table[-c.rate.q_index_range[0]] == 1.0),
                ("four variable-rate models with distinct ids and rising beta",
                 [m["id"] for m in c.rate.models] == [0, 1, 2, 3]
                 and [m["beta_train"] for m in c.rate.models]
                 == sorted(m["beta_train"] for m in c.rate.models)),
                ("synthesis_width is analysis_width reversed in stage count",
                 len(ch.synthesis_width) == len(ch.analysis_width)
                 == c.geometry.analysis_stages),
                # The final analysis stage IS the projection to the latent -- there
                # is no separate 1x1 after it. If this is violated the model still
                # builds (AnalysisTransform raises, but only when instantiated), so
                # catching it at config load is the difference between a clear
                # message and a shape error deep in a training run.
                ("analysis_width[-1] == primary_latent  (last stage is the projection)",
                 ch.analysis_width[-1] == ch.primary_latent),
                ("three enabled decoders with distinct ids",
                 len({d["id"] for d in c.decoders if d.get("enabled", True)}) == 3),
            ]
            bad = [msg for msg, ok in checks if not ok]
            status = "ok" if not bad else "FAILED: " + "; ".join(bad)
            print(f"  {name:8} {len(checks) - len(bad)}/{len(checks)}  {status}")


if __name__ == "__main__":
    main()
