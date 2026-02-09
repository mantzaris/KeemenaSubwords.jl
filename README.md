# KeemenaSubwords.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/)
[![Build Status](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia-native subword tokenization for BPE, Byte-level BPE, WordPiece, SentencePiece, tiktoken, and Hugging Face `tokenizer.json` workflows.

## Installation

```julia
] add https://github.com/mantzaris/KeemenaSubwords.jl
```

## Quickstart

```julia
using KeemenaSubwords

# Built-in
bpe = load_tokenizer(:core_bpe_en)
tokenize(bpe, "hello world")

# Prefetch artifact-backed models for offline use
prefetch_models([:tiktoken_cl100k_base, :qwen2_5_bpe])

# Bring your own local tokenizer file
local_tok = load_tokenizer("/path/to/tokenizer.json"; format=:hf_tokenizer_json)
```

## Featured Models

- `:tiktoken_cl100k_base`
- `:tiktoken_o200k_base`
- `:mistral_v3_sentencepiece`
- `:qwen2_5_bpe`
- `:phi2_bpe`
- `:roberta_base_bpe`

## Documentation

- Model inventory and metadata: `docs/src/models.md`
- Tokenizer formats reference: `docs/src/formats.md`
- Loading from local paths: `docs/src/loading_local.md`
- LLM cookbook (OpenAI/Mistral/Qwen/LLaMA): `docs/src/llm_cookbook.md`
- Installable gated models (LLaMA): `docs/src/gated_models.md`
- Troubleshooting detection/layout issues: `docs/src/troubleshooting.md`

Run docs locally:

```julia
julia --project=docs docs/make.jl
```
