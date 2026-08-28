"""Tests for training-data preparation.

    python tests/test_data.py
    pytest tests/test_data.py

No torch needed. Uses synthetic source images that include the awkward cases:
textured (should yield crops), perfectly flat (should be rejected), and smaller
than the crop size (should be reported as an error, not crash).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jpegai.data.prepare_crops import prepare  # noqa: E402


def _make_sources(src: Path) -> None:
    from PIL import Image

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:600, 0:800]
    for i in range(4):
        a = np.stack([(xx * 3 + i * 40) % 256, (yy * 3) % 256, ((xx + yy) * 2) % 256],
                     axis=-1).astype(np.int16)
        a += rng.integers(-30, 31, a.shape)
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(src / f"tex{i}.png")
    Image.fromarray(np.full((600, 800, 3), 128, np.uint8)).save(src / "flat.png")
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(src / "tiny.png")


def _run(src: Path, out: Path, **kw):
    return prepare([str(src)], out, crop=256, per_image={str(src): 3}, **kw)


def test_extraction():
    td = Path(tempfile.mkdtemp())
    try:
        src, out = td / "src", td / "out"
        src.mkdir()
        out.mkdir()
        _make_sources(src)

        m = _run(src, out, workers=0)
        label = next(iter(m["sources"]))
        s = m["sources"][label]

        # The output directory must NOT be the source directory. `out_root / name`
        # with an absolute `name` used to resolve to `name` itself, writing crops
        # into the dataset being read. That is the bug this test exists for.
        assert Path(s["out"]).resolve() != src.resolve()
        assert not list(src.glob("*_c??.png")), "crops leaked into the source dir"

        assert s["crops"] == 12, f"4 textured x 3 crops expected, got {s['crops']}"
        assert s["rejected_flat"] == 3, f"flat.png x 3 expected, got {s['rejected_flat']}"
        assert s["n_errors"] == 1 and "too small" in s["errors"][0][1]

        crops = sorted(Path(s["out"]).glob("*.png"))
        assert len(crops) == 12

        from PIL import Image
        stds = []
        for p in crops:
            arr = np.asarray(Image.open(p))
            assert arr.shape == (256, 256, 3) and arr.dtype == np.uint8
            stds.append(float(arr.std()))
        assert min(stds) >= 8.0, f"a crop flatter than the threshold got through: {min(stds)}"
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_deterministic_and_resumable_and_parallel_agree():
    td = Path(tempfile.mkdtemp())
    try:
        src = td / "src"
        src.mkdir()
        _make_sources(src)

        outs = {}
        for tag, workers in (("serial", 0), ("again", 0), ("parallel", 3)):
            d = td / tag
            d.mkdir()
            outs[tag] = _run(src, d, workers=workers)

        def files(tag):
            label = next(iter(outs[tag]["sources"]))
            root = Path(outs[tag]["sources"][label]["out"])
            return {p.name: p.read_bytes() for p in root.glob("*.png")}

        a, b, c = files("serial"), files("again"), files("parallel")

        # Determinism: the crop set must not depend on the run, and must not
        # depend on the worker count either. The per-image seed comes from
        # crc32(filename); Python's hash() would be randomised per process and
        # would break exactly this property.
        assert a.keys() == b.keys() == c.keys()
        assert a == b, "two identical runs produced different crops"
        assert a == c, "parallel and serial runs disagree"

        # Resumability: rerunning into a populated directory recomputes nothing
        # for images that produced crops. The 4 textured images are skipped;
        # flat.png (all crops rejected) and tiny.png (too small) wrote no files,
        # so they are re-attempted. That is correct -- there is no output to
        # resume from -- and cheap, since neither decodes to anything useful.
        again = _run(src, td / "serial", workers=0)
        label = next(iter(again["sources"]))
        assert again["sources"][label]["images_already_done"] == 4, (
            f"expected the 4 crop-producing images to be skipped, got "
            f"{again['sources'][label]['images_already_done']}"
        )
        assert files("serial") == a, "a resumed run modified existing crops"
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_refuses_to_write_into_source():
    td = Path(tempfile.mkdtemp())
    try:
        src = td / "src"
        src.mkdir()
        _make_sources(src)
        # out_root such that out_root/<basename> == src
        try:
            prepare([str(src)], td, crop=256, per_image={str(src): 2}, workers=0)
        except ValueError as exc:
            assert "source directory" in str(exc)
        else:
            raise AssertionError("must refuse to write crops into the source dir")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
