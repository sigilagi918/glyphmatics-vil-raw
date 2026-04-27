# ARCAGI3 Notebook Glyph Encoding

This artifact encodes top ARCAGI3 notebooks as GlyphMatics structural glyphlines.

It does not use base64.
It does not use byte-to-Braille.
It does not use byte-to-Hanzi.
It does not hide payload bytes in the image.

The encoder extracts functional code cells from notebooks and compresses repeated code lines into GlyphMatics dictionary tokens.

## Build

python src/encode_arcagi3_notebooks.py --top 12 --render --decode-test

## Outputs

artifacts/arcagi3/arcagi3_top_notebooks.glyphlines.txt
artifacts/arcagi3/arcagi3_top_notebooks_manifest.json
artifacts/arcagi3/arcagi3_top_notebooks_glyphline_image.png

## Rule

GlyphMatics compression is structural/semantic compression, not byte wrapping.
