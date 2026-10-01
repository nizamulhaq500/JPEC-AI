"""On-disk codestream container for the `jpegai` CLI.

The paper's normative bitstream (T.840-1's markers and substreams) is Track B, gated on
tables that are not in hand -- see docs/03 §0. What this package provides instead is an
honest *container*: a small self-describing file that carries the exact entropy substreams
`TwoBranchCodec.compress` produces, losslessly, so `jpegai encode`/`decode` round-trip a
real file. The rate that file *reports* is the model's own accounting
(`packet_bytes + header_bytes`), not the container's on-disk size -- the difference is the
structural scaffolding the real codestream would fold into 12-bit fields, and `inspect`
prints both so the gap is never hidden.
"""
from jpegai.codestream.container import (CONTAINER_VERSION, MAGIC, coded_bytes,
                                         describe, read_packet, write_packet)

__all__ = ["MAGIC", "CONTAINER_VERSION", "write_packet", "read_packet",
           "describe", "coded_bytes"]
