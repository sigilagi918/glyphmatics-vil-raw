# GlyphMatics Known OS Ubuntu Prior v0.9.2

Branch:

dev/v0.9-horizontal-vertical-priors

## Purpose

Add Ubuntu as a known operating-system prior.

This prior indexes public OS structure:

- OS identity
- apt source structure
- dpkg package database summary
- filesystem layout
- executable names
- config names
- service names
- library/config/doc/development layers

## Boundary

This is not a full byte-copy of Ubuntu.

Excluded:

- /home
- /root
- /proc
- /sys
- /dev
- /run
- /tmp
- /var/log
- secrets
- tokens
- ssh keys
- shell histories
- raw package archives

## Glyphline

pUBUNTUZvhz

Meaning:

Python → Ubuntu known OS prior → verify → hash-lock → seal

## Rule

Known OS structure can be represented as a shared prior. Re-expansion depends on legal package sources, package manifests, and compatible runtime environment.
