# GlyphMatics VIL-RAW Artifact Runtime

Locked version: v0.6

## Core Glyphstrings

- pgvayh = install GlyphMatics core
- p hcbme tn:20 b:64 TMGE cv ruz = run real MPS tensor-network self-test and seal
- pPvhz = pack full runtime into .gma.json
- pAvhz = activate pack in isolated restore space
- pLvhz = append lineage block
- pVvhz = encode pack into raw visual memory image

## Quick Start

python src/install_glyphmatics.py pgvayh
python src/glyphmatics_tn_vm.py p hcbme tn:20 b:64 TMGE cv ruz

## Status

G14 accepted: installable, restorable, executable, verifiable, lineage-linked, and image-encoded.

## Literal Image Execution Artifact

The repo now includes a literal glyphline image execution layer.

File:

- `src/glyphmatics_literal_image.py`

Example:

```bash
python src/glyphmatics_literal_image.py make --input artifacts/glyphlines.txt --out artifacts/glyphmatics_transport.png
python src/glyphmatics_literal_image.py extract artifacts/glyphmatics_transport.png
python src/glyphmatics_literal_image.py run artifacts/glyphmatics_transport.png
The image contains only visible glyphlines. Execution is recovered from OCR plus canonical correction.
Verified semantic execution compression ratio:
Plain text
4,636,425 expanded artifact bytes / 61 glyphline bytes = 76,006.97 : 1
