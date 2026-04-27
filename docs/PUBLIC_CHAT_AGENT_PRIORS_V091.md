# GlyphMatics Public Major Chat-Agent Prior Families v0.9.1

Branch:

dev/v0.9-horizontal-vertical-priors

## Purpose

Build prior families across public/owned major model chat-agent systems.

## Included Prior Families

- open-weight model family priors
- chat interface priors
- tool-calling priors
- agent-runtime priors
- memory/RAG priors
- inference-runtime priors
- tokenizer priors
- eval/benchmark priors
- deployment priors
- safety/alignment interface priors
- structural weight priors

## Major Family Buckets

- Llama
- Qwen
- DeepSeek
- Mistral/Mixtral
- Gemma
- Phi
- Grok
- ChatGPT/OpenAI interface
- Claude/Anthropic interface
- Gemini interface
- Cohere/Command-R
- Falcon
- Yi
- Granite
- StableLM
- SmolLM

## Boundary

Public/owned artifacts only.

Weights are structurally indexed only:

- file extension
- file size
- first-1MB sample hash
- GGUF metadata/tensor table when parseable
- Safetensors header when parseable
- NPZ member table when parseable

Raw tensors are not copied into the prior dataset.

## Rule

No private data.
No stolen repositories.
No raw tensor copying.
No hidden payloads.
No base64 byte wrapping.
No byte-to-Braille.
No byte-to-Hanzi.

## Glyphline

pCHATZvhz

Meaning:

Python → public chat-agent prior family build → verify → hash-lock → seal.
