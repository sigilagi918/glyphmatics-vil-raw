# GlyphMatics GGUF Tensor Prior v0.8

Branch:

dev/v0.8-gguf-tensor-prior

Purpose:

Build a GlyphMatics structural prior for GGUF models by parsing:

- GGUF header
- metadata key/value table
- tensor names
- tensor ranks
- tensor shapes
- tensor GGML quantization types
- tensor offsets
- repeated architecture patterns

Boundary:

This is structural metadata compression. It is not yet full byte-lossless GGUF tensor reconstruction.

Rule:

No base64.
No byte-to-Braille.
No byte-to-Hanzi.
No hidden payloads.
Only structural prior extraction and glyphline dictionary construction.
