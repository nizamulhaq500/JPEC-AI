#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# JPEG AI reimplementation — one-time environment + dataset setup.
#
#   cd "/Users/nizam/JPEC AI" && bash setup.sh
#
# Everything here touches the network, which is why you run it and not Claude.
# Safe to re-run: each step is skipped if already done.
# Total download is ~16 GB. Steps 3-5 can be interrupted and resumed.
# ---------------------------------------------------------------------------
set -u   # deliberately NOT -e: a failed optional download shouldn't abort the rest

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    ! %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    ok %s\033[0m\n' "$*"; }

# --- 1. virtualenv ---------------------------------------------------------
say "1/6  Python virtual environment"
if [ ! -d .venv ]; then
  python3 -m venv .venv || { warn "venv creation failed"; exit 1; }
  ok "created .venv"
else
  ok ".venv already exists"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

# --- 2. dependencies ------------------------------------------------------
say "2/6  Python packages (a few minutes)"
python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
  warn "some packages failed. Most likely culprits and what to do:"
  warn "  pyiqa            -> optional; metrics.py degrades gracefully without it"
  warn "  pillow-avif      -> try:  brew install libavif  then re-run"
  warn "  compressai       -> needs a C++ toolchain: xcode-select --install"
  warn "Re-run this script after fixing; installed packages are kept."
else
  ok "all packages installed"
fi

# --- 3. Kodak (24 images, ~25 MB) — mandatory, matches paper Table V ------
say "3/6  Kodak test set"
mkdir -p data/kodak
if [ "$(ls -1 data/kodak/*.png 2>/dev/null | wc -l | tr -d ' ')" -ge 24 ]; then
  ok "Kodak already present"
else
  for i in $(seq -w 1 24); do
    f="data/kodak/kodim${i}.png"
    [ -s "$f" ] && continue
    curl -sSfL -o "$f" "https://r0k.us/graphics/kodak/kodak/kodim${i}.png" \
      || { warn "kodim${i} failed"; rm -f "$f"; }
  done
  ok "Kodak: $(ls -1 data/kodak/*.png 2>/dev/null | wc -l | tr -d ' ')/24 images"
fi

# --- 4. DIV2K (train + valid, ~8 GB) — training data ---------------------
say "4/6  DIV2K (~8 GB, the long one)"
mkdir -p data/div2k
DIV2K_BASE="https://data.vision.ee.ethz.ch/cvl/DIV2K"
for z in DIV2K_train_HR.zip DIV2K_valid_HR.zip; do
  if [ -d "data/div2k/${z%_HR.zip}_HR" ] || [ -d "data/div2k/${z%.zip}" ]; then
    ok "$z already extracted"; continue
  fi
  if [ ! -s "data/div2k/$z" ]; then
    echo "    downloading $z ..."
    curl -# -fL -o "data/div2k/$z" "$DIV2K_BASE/$z" || { warn "$z failed"; continue; }
  fi
  echo "    extracting $z ..."
  ( cd data/div2k && unzip -q -o "$z" ) && rm -f "data/div2k/$z" && ok "$z done"
done

# --- 5. Flickr2K (~8 GB) — optional, more training data ------------------
say "5/6  Flickr2K (optional, ~8 GB) — skip with Ctrl-C, DIV2K alone is workable"
mkdir -p data/flickr2k
if [ -d data/flickr2k/Flickr2K ]; then
  ok "Flickr2K already extracted"
else
  if [ ! -s data/flickr2k/Flickr2K.tar ]; then
    curl -# -fL -o data/flickr2k/Flickr2K.tar \
      "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar" || warn "Flickr2K failed (fine — optional)"
  fi
  if [ -s data/flickr2k/Flickr2K.tar ]; then
    ( cd data/flickr2k && tar xf Flickr2K.tar ) && rm -f data/flickr2k/Flickr2K.tar && ok "Flickr2K done"
  fi
fi

# --- 6. report ------------------------------------------------------------
say "6/6  Summary"
python - <<'PY'
import importlib, pathlib, sys
def has(m):
    try: importlib.import_module(m); return True
    except Exception: return False

print("    python      ", sys.version.split()[0])
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("    torch       ", torch.__version__, "| device:", dev)
except Exception:
    print("    torch        MISSING  <-- blocking, fix this first")

for name, mod in [("compressai","compressai"), ("piq","piq"),
                  ("pytorch_msssim","pytorch_msssim"), ("pyiqa","pyiqa"),
                  ("avif","pillow_avif"), ("streamlit","streamlit"),
                  ("onnxruntime","onnxruntime")]:
    print(f"    {name:12} {'ok' if has(mod) else 'missing'}")

for label, pat in [("kodak","data/kodak/*.png"),
                   ("div2k train","data/div2k/DIV2K_train_HR/*.png"),
                   ("div2k valid","data/div2k/DIV2K_valid_HR/*.png"),
                   ("flickr2k","data/flickr2k/Flickr2K/*.png")]:
    n = len(list(pathlib.Path('.').glob(pat)))
    print(f"    {label:12} {n} images")

# ffmpeg/libvmaf is separate: VMAF is the one metric we shell out for
import shutil
print("    ffmpeg      ", "ok" if shutil.which("ffmpeg") else "missing (VMAF needs it: brew install ffmpeg)")
PY

cat <<'EOF'

Next:
  source .venv/bin/activate
  python -m jpegai.data.prepare_crops     # one-time: 256x256 crop extraction
  python -m jpegai.eval.runbench --codecs jpeg,webp,avif --dataset kodak

Then tell Claude what the summary above printed, especially any "missing".
EOF
