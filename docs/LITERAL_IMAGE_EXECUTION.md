# GlyphMatics Literal Image Execution

This layer makes a single image the transport and execution artifact.

The image contains only visible glyphline text. No base64. No hidden byte payload. No Braille or Hanzi byte wrapper.

## Canonical Glyphlines

pgvayh
p hcbme tn:20 b:64 TMGE cv ruz
pPvhz
pAvhz
pLvhz
pVvhz

## Verified Execution

The image was OCR-extracted, canonicalized, and executed.

Raw OCR included one noisy line:

pPAvhz

Canonical correction:

pAvhz

## Verified Chain

1. Install GlyphMatics core.
2. Run TN/MPS selftest.
3. Pack runtime.
4. Activate latest pack.
5. Append lineage.
6. Encode latest pack to visual image.

## Semantic Compression Ratio

Expanded verified artifact bytes: 4,636,425
Canonical glyphline bytes: 61

Ratio: 76,006.97 : 1

This is command/execution compression, not byte compression.
