"""Encoder-side, non-normative machinery: the searches a compliant encoder may skip.

Nothing in here goes in the bitstream. `jpegai.codestream` will hold what does; this
package holds the decisions an encoder makes *before* writing one, which the standard
leaves entirely open. Keeping the two apart is the point: a change under
`jpegai/coder/` can cost BD-rate and encode time but can never make a stream
undecodable, and a change under `jpegai/codestream/` can.

Currently one module, `brm` -- bit-rate matching, section III-B2. Import from it
directly (`from jpegai.coder.brm import brm`); this file stays free of re-exports so
that `python -m jpegai.coder.brm` runs the module's self-test without importing it
twice.
"""
