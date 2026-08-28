"""Tests for the truncated-zip salvage tool.

The tool's one safety property is that it never writes a file it could not verify
against the CRC32 the archive recorded. Everything else it does is convenience; if
the CRC gate does not hold, the tool is a corruption laundering machine -- it would
emit plausible PNGs whose wrong pixels then produce wrong PSNRs that nothing
downstream could flag. So the CRC gate is tested directly, by corrupting a byte and
asserting the file is refused.

Pure stdlib: builds its own archives with `zipfile`, so there is nothing to
download and no dependency on the DIV2K file that motivated the tool.
"""

from __future__ import annotations

import zipfile

import pytest

from jpegai.data.salvage_zip import extract, salvage, scan

# Deliberately incompressible-ish and compressible content, so both branches of
# the extractor are exercised with realistic data rather than zeros.
MEMBERS = {
    "d/a.bin": bytes(range(256)) * 40,
    "d/b.txt": b"the quick brown fox\n" * 500,
    "d/c.bin": bytes((i * 37 + 11) % 256 for i in range(9000)),
    "d/last.bin": b"Z" * 20000,
}


def _build(path, *, method=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=method) as z:
        for name, data in MEMBERS.items():
            z.writestr(name, data)
    return path.read_bytes()


def _truncate(path, raw, keep):
    path.write_bytes(raw[:keep])


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_intact_archive_recovers_everything(tmp_path, method):
    z = tmp_path / "a.zip"
    _build(z, method=method)
    out = tmp_path / "out"
    assert salvage(z, out, verbose=False) == 0
    for name, data in MEMBERS.items():
        assert (out / name).read_bytes() == data


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_truncation_loses_only_the_straddling_member(tmp_path, method):
    """The real failure: no central directory, last member cut mid-data."""
    z = tmp_path / "a.zip"
    raw = _build(z, method=method)
    # Cut 10 bytes into the last member's data. Located via `scan` rather than by
    # searching for the name: the name appears twice (local header and central
    # directory), and cutting at the *second* one leaves all member data and the
    # end-of-central-directory record intact, which is not the failure under test.
    entries, _ = scan(memoryview(raw))
    last = [e for e in entries if not e.name.endswith("/")][-1]
    _truncate(z, raw, last.data_offset + 10)

    assert b"PK\x05\x06" not in z.read_bytes()      # index really is gone
    out = tmp_path / "out"
    assert salvage(z, out, verbose=False) == 0

    survivors = sorted(p.name for p in (out / "d").iterdir())
    assert survivors == ["a.bin", "b.txt", "c.bin"]
    for name in ("d/a.bin", "d/b.txt", "d/c.bin"):
        assert (out / name).read_bytes() == MEMBERS[name]


def test_corrupt_byte_is_refused_not_written(tmp_path):
    """The safety property. A flipped bit must fail the CRC and produce no file."""
    z = tmp_path / "a.zip"
    raw = bytearray(_build(z))
    # Flip a byte inside the first member's data, leaving every length and the
    # stored CRC untouched -- exactly what silent disk or transfer corruption
    # looks like, and undetectable by any check other than the CRC.
    entries, _ = scan(memoryview(bytes(raw)))
    first = next(e for e in entries if not e.name.endswith("/"))
    raw[first.data_offset + 5] ^= 0xFF
    z.write_bytes(bytes(raw))

    blob = memoryview(z.read_bytes())
    entries, _ = scan(blob)
    out = tmp_path / "out"
    out.mkdir()
    target = next(e for e in entries if e.name == first.name)
    ok, note = extract(target, blob, out)
    assert not ok
    assert "crc" in note
    assert not (out / first.name).exists()          # refused, not written-then-flagged


def test_the_other_members_still_survive_one_corrupt_member(tmp_path):
    z = tmp_path / "a.zip"
    raw = bytearray(_build(z))
    entries, _ = scan(memoryview(bytes(raw)))
    first = next(e for e in entries if not e.name.endswith("/"))
    raw[first.data_offset + 5] ^= 0xFF
    z.write_bytes(bytes(raw))

    out = tmp_path / "out"
    assert salvage(z, out, verbose=False) == 0      # partial recovery is success
    assert not (out / first.name).exists()
    remaining = {n for n in MEMBERS if n != first.name}
    for name in remaining:
        assert (out / name).read_bytes() == MEMBERS[name]


def test_scan_walks_forward_and_does_not_grep(tmp_path):
    """A local-header signature *inside* member data must not be mistaken for one.

    This is why `scan` walks header-to-header using the declared sizes instead of
    searching for `PK\\x03\\x04`. A grep-based salvage tool would find five members
    here and emit garbage for the phantom one.
    """
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("real.bin", b"xx" + b"PK\x03\x04" + b"\x00" * 40 + b"yy")
        f.writestr("also.bin", b"tail")

    entries, _ = scan(memoryview(z.read_bytes()))
    names = [e.name for e in entries if not e.name.endswith("/")]
    assert names == ["real.bin", "also.bin"]


def test_nothing_recoverable_is_a_failure(tmp_path):
    """Exit status must distinguish "salvaged some" from "salvaged nothing"."""
    z = tmp_path / "a.zip"
    raw = _build(z)
    # Keep only part of the very first member, so no complete file exists.
    _truncate(z, raw, raw.find(b"d/a.bin") + 20)
    assert salvage(z, tmp_path / "out", verbose=False) == 1


def test_dry_run_writes_nothing(tmp_path):
    z = tmp_path / "a.zip"
    _build(z)
    out = tmp_path / "out"
    assert salvage(z, out, dry_run=True, verbose=False) == 0
    assert not out.exists() or not any(out.rglob("*.bin"))
