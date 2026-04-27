# GlyphMatics Horizontal Consolidation + Vertical Integration Prior Fusion v0.9

Branch:

dev/v0.9-horizontal-vertical-priors

## Purpose

Build shared-prior datasets across public/owned competitor software and model-weight structures.

## Horizontal Consolidation

Groups artifacts by equivalent functional layer across systems:

- data ingest
- tokenization
- model weights
- inference runtime
- agent runtime
- benchmark solver
- vector memory
- visual artifact
- tensor network
- deployment
- UI/web
- security verification

## Vertical Integration

Maps each project into full stack stages:

source → ingest → normalize → tokenize → train → weights → inference → agent → evaluate → package → deploy → verify → visualize

## Weight Boundary

Weights are structurally indexed only:

- file type
- size
- sample hash
- GGUF metadata/tensor table when available
- Safetensors header when available
- NPZ member structure when available

Raw weight tensors are not copied into the prior dataset.

## Rule

No private data.
No stolen repositories.
No hidden payloads.
No base64 byte wrapping.
No byte-to-Braille.
No byte-to-Hanzi.

## Glyphline

pHVZvhz

Meaning:

Python → horizontal/vertical prior fusion → verify → hash-lock → seal
