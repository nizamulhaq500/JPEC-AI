"""Recover files from a truncated zip archive.

    python -m jpegai.data.salvage_zip data/div2k/DIV2K_valid_HR.zip data/div2k

`DIV2K_valid_HR.zip` arrived 30 MiB short of its 428 MiB, so `unzip` refuses it
outright: it looks for the end-of-central-directory record, which lives in the
*last* bytes of the file and is exactly what a truncated download loses.

But the central directory is only an index. Every member also carries its own
**local file header** immediately before its data -- signature `PK\\x03\\x04`,
then the compression method, the CRC32, the compressed and uncompressed sizes,
and the name. So the archive can be read forward from the start, entry by entry,
and every member that fully arrived can be recovered *and verified* against its
stored CRC32. Only the entry straddling the truncation point is lost.

This is worth doing rather than re-downloading because it is 100 files of
validation data that are already on disk, and because a re-download of the same
428 MiB over the same link is what produced the truncation in the first place.

Why the CRC check is the whole point
------------------------------------
A salvage tool that emits plausible-looking files without verifying them is worse
than no tool: a subtly corrupt validation image produces a subtly wrong PSNR, and
nothing downstream would ever flag it. Every file written here has had its CRC32
compared against the value the archive recorded before compression. Files that
fail are not written. The exit status is non-zero if anything failed, so this can
be scripted.

Scope: handles stored (method 0) and deflated (method 8) entries, with or without
a trailing data descriptor. It does not handle encryption, zip64 members over 4
GiB, or multi-disk archives -- none of which occur in the datasets this project
downloads.
"""

from __future__ import annotations

import argparse
import binascii
import struct
import sys
import zlib
from pathlib import Path

LOCAL_SIG = b"PK\x03\x04"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")   # 30 bytes
#: Bit 3 of the general-purpose flags: sizes are zero in the local header and the
#: real values follow the data in a descriptor record.
FLAG_DATA_DESCRIPTOR = 0x08


class Entry:
    __slots__ = ("name", "method", "crc", "csize", "usize", "data_offset", "streamed")

    def __init__(self, name, method, crc, csize, usize, data_offset, streamed):
        self.name = name
        self.method = method
        self.crc = crc
        self.csize = csize
        self.usize = usize
        self.data_offset = data_offset
        self.streamed = streamed


def scan(blob: memoryview) -> tuple[list[Entry], int]:
    """Walk local file headers from the start. Returns (entries, bytes_consumed).

    Walked forward rather than searched for, so a `PK\\x03\\x04` byte pair that
    happens to occur *inside* compressed data cannot be mistaken for a header --
    which is why this does not simply grep for the signature.
    """
    entries: list[Entry] = []
    pos = 0
    n = len(blob)
    while pos + LOCAL_HEADER.size <= n:
        if bytes(blob[pos:pos + 4]) != LOCAL_SIG:
            break
        (_sig, _ver, flag, method, _t, _d, crc, csize, usize,
         nlen, elen) = LOCAL_HEADER.unpack(blob[pos:pos + LOCAL_HEADER.size])
        name_at = pos + LOCAL_HEADER.size
        data_at = name_at + nlen + elen
        if data_at > n:
            break                                   # header itself is cut off
        name = bytes(blob[name_at:name_at + nlen]).decode("utf-8", "replace")
        streamed = bool(flag & FLAG_DATA_DESCRIPTOR) or (csize == 0 and usize > 0)

        if streamed:
            # Size unknown until the stream ends. Inflate to find out where that
            # is; `unused_data` then gives the true compressed length.
            if method != 8:
                break                               # stored + streamed is unreadable
            d = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                d.decompress(bytes(blob[data_at:]))
            except zlib.error:
                break
            if not d.eof:
                entries.append(Entry(name, method, crc, 0, 0, data_at, True))
                pos = n                             # truncated inside this member
                break
            csize = len(blob) - data_at - len(d.unused_data)
        entries.append(Entry(name, method, crc, csize, usize, data_at, streamed))
        pos = data_at + csize
        if streamed:
            pos += 16                               # data descriptor, sans signature
    return entries, pos


def extract(entry: Entry, blob: memoryview, out_root: Path,
            *, dry_run: bool = False) -> tuple[bool, str]:
    """Returns (ok, note). Never writes a file whose CRC32 does not match."""
    if entry.name.endswith("/"):
        return True, "directory"

    end = entry.data_offset + entry.csize
    if end > len(blob):
        have = len(blob) - entry.data_offset
        return False, f"truncated: {have:,} of {entry.csize:,} bytes present"
    raw = bytes(blob[entry.data_offset:end])

    if entry.method == 0:
        data = raw
    elif entry.method == 8:
        try:
            data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
        except zlib.error as exc:
            return False, f"inflate failed: {exc}"
    else:
        return False, f"unsupported method {entry.method}"

    if entry.usize and len(data) != entry.usize:
        return False, f"size {len(data):,} != declared {entry.usize:,}"
    got = binascii.crc32(data) & 0xFFFFFFFF
    if entry.crc and got != entry.crc:
        return False, f"crc {got:08x} != declared {entry.crc:08x}"

    # `..` in a member name would let an archive write outside out_root. Not a
    # concern for a dataset from ETH Zurich, but this tool is generic and the
    # check is two lines.
    dest = (out_root / entry.name).resolve()
    if not str(dest).startswith(str(out_root.resolve())):
        return False, "path escapes the output directory"
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return True, f"{len(data):,} bytes, crc ok"


def salvage(zip_path: Path, out_root: Path, *, dry_run: bool = False,
            verbose: bool = True) -> int:
    raw = zip_path.read_bytes()
    blob = memoryview(raw)
    complete = raw.rfind(b"PK\x05\x06") >= 0
    entries, consumed = scan(blob)

    print(f"archive  {zip_path}  ({len(raw):,} bytes)")
    print(f"index    {'intact' if complete else 'MISSING (truncated download)'}")
    # Clamped: the last header's declared size legitimately points past the end of
    # a truncated file, and reporting "399 MB of 397 MB" reads like a bug.
    accounted = min(consumed, len(raw))
    print(f"members  {len(entries)} local headers found, "
          f"{accounted:,} of {len(raw):,} bytes accounted for"
          + (f" (last member declares {consumed - len(raw):,} more bytes than exist)"
             if consumed > len(raw) else ""))
    if complete:
        print("note     this archive is not damaged; plain `unzip` is the right tool")
    print()

    ok = bad = 0
    for e in entries:
        good, note = extract(e, blob, out_root, dry_run=dry_run)
        if e.name.endswith("/"):
            continue
        if good:
            ok += 1
        else:
            bad += 1
        if verbose and (not good or ok <= 3 or ok % 25 == 0):
            print(f"  {'ok ' if good else 'FAIL'} {e.name:40} {note}")

    lost = len(raw) - accounted
    print(f"\nrecovered {ok} files, {bad} unrecoverable"
          + (f", {lost:,} trailing bytes unusable" if lost > 0 else ""))
    if dry_run:
        print("(dry run -- nothing written)")
    elif ok:
        print(f"wrote into {out_root}")
    # A partly-recovered archive is a success: the point is to get what is there.
    # Non-zero only if nothing at all came out.
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jpegai.data.salvage_zip",
                                 description="Recover files from a truncated zip.")
    ap.add_argument("zip", type=Path)
    ap.add_argument("out", type=Path, nargs="?", default=None,
                    help="output directory; default is the archive's own directory")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.zip.exists():
        print(f"no such file: {a.zip}", file=sys.stderr)
        return 2
    return salvage(a.zip, a.out or a.zip.parent, dry_run=a.dry_run,
                   verbose=not a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
