"""Streamlit front-end for the JPEG AI demo -- a thin shell over `demo/engine.py`.

Run it (outside the sandbox; it binds a local port):

    PYTHONPATH=. streamlit run demo/app.py

Every panel is backed by the real codec: the sidebar loads a trained checkpoint with the
CLI's own loader, and each control drives `engine.encode/decode/latent_maps`, which call
`model.compress/decompress/forward` -- the exact path `jpegai encode/decode` takes, not a
parallel reimplementation. The "Not in this build" expander renders
`engine.CAPABILITIES["out"]` verbatim, so the honesty boundary is on screen.

Responsiveness: Streamlit reruns the whole script on every widget change, so each heavy
codec call is memoised with `st.cache_data` -- keyed on the image id + the panel's knobs,
with the model and arrays passed as un-hashed `_`-prefixed args -- and only ONE panel is
rendered per run (a server-side radio, not `st.tabs`, which executes all five bodies every
rerun). A slider move re-runs just that panel's changed compute; the rest is a cache hit,
and the plain Δβ=0 / no-RoI encode is shared across panels, so it is computed once.
"""
from __future__ import annotations

import glob
import hashlib
import os
import tempfile

import numpy as np
import streamlit as st

from demo import engine
from jpegai.codestream import write_packet

DEFAULT_CKPT = "checkpoints/p8_vr_mcm_IV/final.pt"
KODAK = "data/kodak"
# Table I's Δβ range the gain unit is trained over; the demo steps in the cached grid.
BETA_LO, BETA_HI, BETA_STEP = -1000, 700, 100
PANELS = ("Accounting", "Variable rate", "Region of interest", "Base vs full",
          "Latent bits")


@st.cache_resource(show_spinner="Loading checkpoint ...")
def _load(checkpoint: str, device: str):
    """Load once per (checkpoint, device); Streamlit keeps the model across reruns."""
    return engine.load(checkpoint, device or None)


def _img_id(rgb: np.ndarray) -> str:
    """A short content hash: an image's cache key, so it isn't re-hashed on every call."""
    return hashlib.sha1(rgb.tobytes()).hexdigest()[:16]


# --- memoised codec calls --------------------------------------------------------------
# Keyed on (image id, checkpoint, device, knobs); the model/arrays ride along un-hashed as
# `_`-prefixed args. `roi_key` (the box rect, or None) distinguishes an RoI encode from the
# plain one without hashing the mask, and lets every panel share the plain encode.

@st.cache_data(show_spinner=False)
def _encode(img_id, ckpt, device, delta_beta, roi_key, roi_boost,
            _model, _dev, _rgb, _mask):
    return engine.encode(_model, _rgb, _dev, delta_beta=delta_beta,
                         roi_mask=_mask, roi_boost=roi_boost)


@st.cache_data(show_spinner=False)
def _decode(img_id, ckpt, device, delta_beta, roi_key, roi_boost, luma_only,
            _model, _dev, _rgb, _mask):
    res = _encode(img_id, ckpt, device, delta_beta, roi_key, roi_boost,
                  _model, _dev, _rgb, _mask)
    return engine.decode(_model, res.packet, _dev, luma_only=luma_only)


@st.cache_data(show_spinner=False)
def _disk(img_id, ckpt, device, delta_beta, roi_key, roi_boost,
          _model, _dev, _rgb, _mask):
    """On-disk accounting: write the cached packet to a real container, read its size and
    streams, then delete the file -- the demo only needs the numbers, not the bytes."""
    res = _encode(img_id, ckpt, device, delta_beta, roi_key, roi_boost,
                  _model, _dev, _rgb, _mask)
    fd, path = tempfile.mkstemp(suffix=".jpegai")
    os.close(fd)
    try:
        write_packet(path, res.packet)
        d = engine.inspect(path)
    finally:
        os.unlink(path)
    return {"coded": res.coded_bytes, "disk": d["disk_bytes"], "bpp": res.bpp,
            "scaffold": 100.0 * d["overhead_bytes"] / max(d["disk_bytes"], 1),
            "streams": d["streams"]}


@st.cache_data(show_spinner=False)
def _latent(img_id, ckpt, device, delta_beta, _model, _dev, _rgb):
    return engine.latent_maps(_model, _rgb, _dev, delta_beta=delta_beta)


def _sample_images() -> list[str]:
    return sorted(glob.glob(f"{KODAK}/*.png"))


def _heatmap(arr, title: str, cmap: str = "magma"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _rect_mask(h: int, w: int, cx: float, cy: float, cw: float, ch: float) -> np.ndarray:
    """A full-resolution rectangle mask (255 inside) from fractional x/y/w/h."""
    m = np.zeros((h, w), np.uint8)
    y0, x0 = int(cy * h), int(cx * w)
    y1, x1 = min(h, y0 + int(ch * h)), min(w, x0 + int(cw * w))
    m[y0:y1, x0:x1] = 255
    return m


def _sidebar():
    """Checkpoint + device + image source.

    Returns (rgb uint8 HxWx3, model, meta, dev, checkpoint, device); the two strings ride
    along as cache keys, so the memoised codec calls invalidate when either changes.
    """
    st.sidebar.header("Codec")
    checkpoint = st.sidebar.text_input("Checkpoint (.pt)", DEFAULT_CKPT)
    device = st.sidebar.selectbox("Device", ["", "cpu", "mps", "cuda"],
                                  help="blank = auto (pick_device)")

    st.sidebar.header("Image")
    samples = _sample_images()
    up = st.sidebar.file_uploader("Upload a PNG/JPEG", type=["png", "jpg", "jpeg"])
    pick = st.sidebar.selectbox("...or a Kodak sample", samples,
                                format_func=lambda p: p.split("/")[-1]) if samples else None
    if up is not None:
        from PIL import Image
        rgb = np.asarray(Image.open(up).convert("RGB"), np.uint8)
    elif pick:
        from PIL import Image
        rgb = np.asarray(Image.open(pick).convert("RGB"), np.uint8)
    else:
        st.sidebar.warning("Upload an image or add Kodak samples under data/kodak/.")
        st.stop()

    try:
        model, meta, dev = _load(checkpoint, device)
    except Exception as e:                     # a bad path shouldn't dump a raw traceback
        st.sidebar.error(f"Could not load the checkpoint:\n\n{e}")
        st.stop()
    st.sidebar.caption(f"model={meta.get('model', '?')}  tier={meta.get('tier', '?')}  "
                       f"format={getattr(model.fmt, 'name', '?')}  "
                       f"gain={'yes' if getattr(model, 'gain', False) else 'no'}")
    st.sidebar.caption("Each panel runs the real codec on first use (a few seconds on "
                       "CPU/MPS); repeats are cached and instant.")
    return rgb, model, meta, dev, checkpoint, device


def _capabilities_panel():
    """Render the honesty boundary verbatim: what the build shows, what it does not."""
    a, b = st.columns(2)
    with a:
        st.markdown("**In scope — backed by the codec**")
        for line in engine.CAPABILITIES["in"]:
            st.markdown(f"- {line}")
    with b:
        with st.expander("Not in this build (named, not faked)"):
            for line in engine.CAPABILITIES["out"]:
                st.markdown(f"- {line}")


def _panel_accounting(img_id, ckpt, device, model, rgb, dev):
    st.subheader("Encode & container accounting")
    st.caption("Two sizes, never conflated: the model's coded rate (what an RD plot is "
               "built from) and the on-disk file, which also carries JSON scaffolding "
               "the normative T.840-1 codestream would fold away.")
    acct = _disk(img_id, ckpt, device, (0, 0), None, 4, model, dev, rgb, None)
    rec = _decode(img_id, ckpt, device, (0, 0), None, 4, False, model, dev, rgb, None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Coded rate", f"{acct['coded']:,} B", f"{acct['bpp']:.4f} bpp")
    c2.metric("On disk", f"{acct['disk']:,} B",
              f"+{acct['disk'] - acct['coded']:,} B scaffolding")
    c3.metric("Reconstruction", f"{engine.psnr(rgb, rec):.2f} dB")
    st.caption(f"scaffolding {acct['scaffold']:.1f}% of the file   •   "
               f"streams (bytes) {acct['streams']}")

    i1, i2 = st.columns(2)
    i1.image(rgb, caption="original", width="stretch")
    i2.image(rec, caption="full decode", width="stretch")


def _panel_vr(img_id, ckpt, device, model, rgb, dev):
    st.subheader("Variable rate — one set of weights, a header offset")
    if not getattr(model, "gain", False):
        st.info("This checkpoint is fixed-rate; load a gain (variable-rate) checkpoint "
                "such as checkpoints/p8_vr_mcm_IV/final.pt to steer Δβ.")
        return
    bl = st.slider("Δβ luma", BETA_LO, BETA_HI, 0, BETA_STEP)
    bc = st.slider("Δβ chroma", BETA_LO, BETA_HI, 0, BETA_STEP)
    res = _encode(img_id, ckpt, device, (bl, bc), None, 4, model, dev, rgb, None)
    rec = _decode(img_id, ckpt, device, (bl, bc), None, 4, False, model, dev, rgb, None)
    c1, c2 = st.columns(2)
    c1.metric("Rate", f"{res.bpp:.4f} bpp", f"{res.coded_bytes:,} B")
    c2.metric("PSNR", f"{engine.psnr(rgb, rec):.2f} dB")
    st.image(rec, caption=f"Δβ = ({bl}, {bc})", width="stretch")
    st.caption("Negative Δβ lowers the rate, positive raises it — the weights never change.")


def _panel_roi(img_id, ckpt, device, model, rgb, dev):
    st.subheader("Region of interest — boost a region, same codec")
    if not getattr(model, "gain", False):
        st.info("RoI needs a gain checkpoint; the quality map is the gain unit's own knob.")
        return
    h, w = rgb.shape[:2]
    c = st.columns(5)
    cx = c[0].slider("box x", 0.0, 0.95, 0.25, 0.05)
    cy = c[1].slider("box y", 0.0, 0.95, 0.25, 0.05)
    cw = c[2].slider("box width", 0.05, 1.0, 0.5, 0.05)
    ch = c[3].slider("box height", 0.05, 1.0, 0.5, 0.05)
    boost = c[4].slider("boost (+Δβ)", 1, 8, 4)
    mask = _rect_mask(h, w, cx, cy, cw, ch)
    rk = (cx, cy, cw, ch)

    base = _encode(img_id, ckpt, device, (0, 0), None, 4, model, dev, rgb, None)
    roi = _encode(img_id, ckpt, device, (0, 0), rk, boost, model, dev, rgb, mask)
    r_base = _decode(img_id, ckpt, device, (0, 0), None, 4, False, model, dev, rgb, None)
    r_roi = _decode(img_id, ckpt, device, (0, 0), rk, boost, False, model, dev, rgb, mask)
    bi, bo = engine.region_psnr(rgb, r_base, mask)
    ri, ro = engine.region_psnr(rgb, r_roi, mask)

    st.markdown(f"Inside the box **{bi:.2f} → {ri:.2f} dB** (Δ {ri - bi:+.2f}) &nbsp;•&nbsp; "
                f"outside **{bo:.2f} → {ro:.2f} dB** (Δ {ro - bo:+.2f}) &nbsp;•&nbsp; "
                f"rate {base.bpp:.4f} → {roi.bpp:.4f} bpp")
    overlay = rgb.copy()
    sel = mask > 127
    overlay[sel] = (0.5 * overlay[sel] + np.array([127, 0, 0])).astype(np.uint8)
    cols = st.columns(3)
    cols[0].image(overlay, caption="RoI box", width="stretch")
    cols[1].image(r_base, caption="baseline", width="stretch")
    cols[2].image(r_roi, caption=f"RoI +{boost}", width="stretch")
    st.caption("The box gains dB and the rest gives a little back — the quality map just "
               "reweights the latent, it is not a second pass over the pixels.")


def _panel_base_full(img_id, ckpt, device, model, rgb, dev):
    st.subheader("Base vs full decode — one codestream")
    from jpegai.models.twobranch import TwoBranchCodec

    res = _encode(img_id, ckpt, device, (0, 0), None, 4, model, dev, rgb, None)
    full = _decode(img_id, ckpt, device, (0, 0), None, 4, False, model, dev, rgb, None)
    base = _decode(img_id, ckpt, device, (0, 0), None, 4, True, model, dev, rgb, None)
    sb = TwoBranchCodec.stream_bytes(res.packet)
    chroma = sb.get("y_uv", 0) + sb.get("z_uv", 0)
    share = 100.0 * chroma / max(res.coded_bytes, 1)

    c1, c2 = st.columns(2)
    c1.metric("Full decode", f"{engine.psnr(rgb, full):.2f} dB")
    c2.metric("Luma-only base", f"{engine.psnr(rgb, base):.2f} dB",
              f"-{share:.0f}% rate (chroma skipped)")
    i1, i2 = st.columns(2)
    i1.image(full, caption="full (luma + chroma)", width="stretch")
    i2.image(base, caption="luma-only base", width="stretch")
    st.caption("The base is a prefix of the same file — a luma-only decoder reads it and "
               "stops, skipping the chroma substreams (about a third of the payload).")


def _panel_latent(img_id, ckpt, device, model, rgb, dev):
    st.subheader("Latent bit-allocation — where the budget goes")
    maps = _latent(img_id, ckpt, device, (0, 0), model, dev, rgb)
    st.caption(f"luma latent grid {maps['grid']} &nbsp;•&nbsp; total "
               f"{maps['total_bits']:,.0f} bits over {int(np.prod(maps['grid']))} positions")
    cols = st.columns(2)
    cols[0].pyplot(_heatmap(maps["bits"], "bits per position  (−Σ log₂ p)"))
    if maps["sigma"] is not None:
        cols[1].pyplot(_heatmap(maps["sigma"], "mean σ-index", cmap="viridis"))
    else:
        cols[1].info("No integer σ on this codec (fused hyper); bit map only.")
    st.caption("Bright = expensive: the coder spends where the image is hard. The σ-index "
               "is the gain unit's own quality signal, averaged over channels.")


def main():
    st.set_page_config(page_title="JPEG AI demo", layout="wide")
    st.title("JPEG AI — learning-based image coding")
    st.caption("A thin shell over the real codec: every panel drives the same "
               "compress / decompress / forward path as `jpegai encode/decode`.")
    _capabilities_panel()
    rgb, model, meta, dev, ckpt, device = _sidebar()
    img_id = _img_id(rgb)

    panel = st.radio("Panel", PANELS, horizontal=True, label_visibility="collapsed")
    st.divider()
    with st.spinner(f"Running the codec on {dev} ..."):
        if panel == "Accounting":
            _panel_accounting(img_id, ckpt, device, model, rgb, dev)
        elif panel == "Variable rate":
            _panel_vr(img_id, ckpt, device, model, rgb, dev)
        elif panel == "Region of interest":
            _panel_roi(img_id, ckpt, device, model, rgb, dev)
        elif panel == "Base vs full":
            _panel_base_full(img_id, ckpt, device, model, rgb, dev)
        else:
            _panel_latent(img_id, ckpt, device, model, rgb, dev)


if __name__ == "__main__":
    main()
