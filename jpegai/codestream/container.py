"""The on-disk `.jpegai` container: a lossless carrier for the codec's substreams.

Format (version 1), small, self-describing, and deliberately plain:

    offset  size   field
    0       4      MAGIC = b"JPAI"
    4       1      CONTAINER_VERSION (uint8)
    5       4      header length H (uint32, big-endian)
    9       H      UTF-8 JSON header (see below)
    9+H     ...    payload: every entropy string concatenated in `_STREAMS` order,
                   then the q_residual bytes if the packet carries a spatial map

The header records the per-stream *lengths*, never the bytes, so the payload is the
raw substreams and nothing else and `read_packet` rebuilds the exact packet
`TwoBranchCodec.compress` produced. The header also stores the model's own rate
accounting (`coded_bytes`, `header_bytes`, per-stream `stream_bytes`) so `inspect`
can report it without a checkpoint and without re-running the coder.

Two sizes, never conflated. `coded_bytes(packet)` is what the *model* charges -- the
number an RD plot is built from. The file on disk is larger: it also holds the JSON
scaffolding (shapes, lengths, format, the delta_beta fields, the spatial map) that
T.840-1 would fold into 12-bit fields and a proper substream. `describe` reports both
and their difference, so the container never flatters the rate it carries.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from jpegai.models.twobranch import TwoBranchCodec

MAGIC = b"JPAI"
CONTAINER_VERSION = 1

# The order substreams are laid down in the payload: luma before chroma, y before z,
# so a luma-only decoder reads a prefix of the file and stops.
_STREAMS = (("luma", "y_strings"), ("luma", "z_strings"),
            ("chroma", "y_strings"), ("chroma", "z_strings"))

_HEAD = struct.Struct(">4sBI")   # MAGIC, version, header length

def coded_bytes(packet: dict, *, luma_only: bool = False) -> int:
    """The model's own rate for `packet`, in bytes -- payload plus its header fields.

    This is `TwoBranchCodec.packet_bytes`, surfaced here so a caller holding a packet
    (or a freshly `read_packet`-ed one) gets the rate without reaching into the codec
    class. It is the number that goes on an RD plot; the file on disk is larger by the
    container scaffolding, which `describe` reports separately.
    """
    return TwoBranchCodec.packet_bytes(packet, luma_only=luma_only)


def _encode_tensor(t: Tensor) -> tuple[dict, bytes]:
    """Serialize an integer map at the narrowest lossless width.

    The spatial q_residual arrives as int64 but its values are in Table I's residual
    range (roughly [-16, 16]), so int8 holds it exactly at an eighth the bytes. Storing
    the wide dtype would inflate the on-disk size with scaffolding the normative
    codestream would never carry -- and `describe`'s overhead number is only honest if
    the container isn't padding it. Decode upcasts (`spatial_reconstruct` goes to int64),
    so the narrowed width is invisible downstream; only the values are load-bearing.
    """
    arr = t.detach().cpu().contiguous().numpy()
    if np.issubdtype(arr.dtype, np.integer) and arr.size:
        lo, hi = int(arr.min()), int(arr.max())
        for dt in (np.int8, np.int16, np.int32):
            info = np.iinfo(dt)
            if lo >= info.min and hi <= info.max:
                arr = arr.astype(dt)
                break
    return ({"shape": list(arr.shape), "dtype": arr.dtype.str,
             "nbytes": int(arr.nbytes)}, arr.tobytes())


def _decode_tensor(meta: dict, raw: bytes) -> Tensor:
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"]))
    arr = arr.reshape(meta["shape"]).copy()   # copy: frombuffer is read-only
    return torch.from_numpy(arr)


def write_packet(path, packet: dict) -> int:
    """Serialize `packet` to `path`; return the on-disk size in bytes.

    Lossless: the byte strings go down verbatim and come back as the same `list[bytes]`,
    the q_residual map is preserved to the bit, and every header-affecting key
    (`delta_beta`, `q_residual`) round-trips, so `packet_bytes` reads identically
    before and after.
    """
    path = Path(path)
    for branch in ("luma", "chroma"):
        if branch not in packet:
            raise ValueError(f"packet has no {branch!r} branch; not a TwoBranch packet")

    header = {
        "shape": [int(v) for v in packet["shape"]],
        "pad": [int(v) for v in packet["pad"]],
        "internal_format": str(packet["internal_format"]),
        "branches": {},
        "accounting": {
            "coded_bytes": TwoBranchCodec.packet_bytes(packet),
            "header_bytes": TwoBranchCodec.header_bytes(packet),
            "streams": TwoBranchCodec.stream_bytes(packet),
        },
    }
    payload = bytearray()
    for branch in ("luma", "chroma"):
        part = packet[branch]
        rec = {"z_shape": [int(v) for v in part["z_shape"]]}
        for key, short in (("y_strings", "y"), ("z_strings", "z")):
            lengths = []
            for s in part[key]:
                payload += s
                lengths.append(len(s))
            rec[short] = lengths
        header["branches"][branch] = rec

    if "delta_beta" in packet:
        d_y, d_uv = packet["delta_beta"]
        header["delta_beta"] = [int(d_y), int(d_uv)]
    if packet.get("q_residual") is not None:
        meta, raw = _encode_tensor(packet["q_residual"])
        header["q_residual"] = meta
        payload += raw
    if packet.get("tools") is not None:
        header["tools"] = sorted(str(t) for t in packet["tools"])

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(_HEAD.pack(MAGIC, CONTAINER_VERSION, len(blob)))
        f.write(blob)
        f.write(payload)
    return path.stat().st_size


def _read_header(f) -> tuple[dict, int]:
    head = f.read(_HEAD.size)
    if len(head) != _HEAD.size:
        raise ValueError("truncated file: no container header")
    magic, version, hlen = _HEAD.unpack(head)
    if magic != MAGIC:
        raise ValueError(f"not a .jpegai container: magic {magic!r} != {MAGIC!r}")
    if version != CONTAINER_VERSION:
        raise ValueError(f"container version {version} != {CONTAINER_VERSION}")
    header = json.loads(f.read(hlen).decode("utf-8"))
    return header, version
def read_packet(path) -> dict:
    """Inverse of `write_packet`: the exact packet dict `compress` produced."""
    with open(path, "rb") as f:
        header, _ = _read_header(f)
        payload = f.read()

    off = 0
    packet: dict = {
        "shape": tuple(header["shape"]),
        "pad": tuple(header["pad"]),
        "internal_format": header["internal_format"],
    }
    for branch in ("luma", "chroma"):
        rec = header["branches"][branch]
        part = {"z_shape": tuple(rec["z_shape"])}
        for key, short in (("y_strings", "y"), ("z_strings", "z")):
            strings = []
            for n in rec[short]:
                strings.append(payload[off:off + n])
                off += n
            part[key] = strings
        packet[branch] = part

    if "delta_beta" in header:
        packet["delta_beta"] = (int(header["delta_beta"][0]),
                                int(header["delta_beta"][1]))
    if "q_residual" in header:
        meta = header["q_residual"]
        packet["q_residual"] = _decode_tensor(meta, payload[off:off + meta["nbytes"]])
        off += meta["nbytes"]
    if "tools" in header:
        packet["tools"] = list(header["tools"])
    return packet


def describe(path) -> dict:
    """A model-free summary of a container, for `jpegai inspect`.

    Reads only the header, so it is cheap and needs no checkpoint. Reports the model's
    coded size (what the rate *is*) beside the on-disk size (what the file *costs*) and
    their difference -- the structural overhead the normative codestream would remove.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header, version = _read_header(f)
    disk = path.stat().st_size
    acct = header["accounting"]
    coded = int(acct["coded_bytes"])
    return {
        "version": version,
        "shape": tuple(header["shape"]),
        "internal_format": header["internal_format"],
        "coded_bytes": coded,
        "header_bytes": int(acct["header_bytes"]),
        "streams": dict(acct["streams"]),
        "disk_bytes": disk,
        "overhead_bytes": disk - coded,
        "delta_beta": tuple(header["delta_beta"]) if "delta_beta" in header else None,
        "tools": list(header["tools"]) if "tools" in header else None,
        "q_residual": header.get("q_residual"),
    }



