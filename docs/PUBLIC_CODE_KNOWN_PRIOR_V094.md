# GlyphMatics Public Code Known Prior v0.9.4

Branch:

dev/v0.9-horizontal-vertical-priors

## Purpose

Add actual public/owned source code as a known prior.

## Glyphline

pCODEZvhz

Meaning:

Python → public code known prior → verify → hash-lock → seal

## Included Material

This prior may include actual source text from local roots when the code is:

- owned by the repository owner, or
- public and redistributable under a permissive license, or
- explicitly included by the user as an owned local artifact.

## Indexed Structures

- actual code text
- source SHA256
- code SHA256
- detected license file
- project bucket
- code layer bucket
- repeated code lines
- file extension distribution

## Boundary

Public code is still copyrighted unless the license allows redistribution.

Unknown-license code should be treated as owned/local-only unless reviewed.

Excluded:

- secrets
- tokens
- private keys
- .env files
- credential files
- ssh material
- caches
- virtual environments
- node_modules
- git internals

## Rule

No private data.
No stolen repositories.
No secrets.
No hidden payloads.
No base64 byte wrapping.
No byte-to-Braille.
No byte-to-Hanzi.
