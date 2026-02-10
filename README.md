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

# 1) built-in model
prefetch_models([:core_bpe_en])
bpe = load_tokenizer(:core_bpe_en)
tokenize(bpe, "hello world")

# 2) local path with explicit format override
local_tiktoken = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)

# 3) gated install workflow
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
llama = load_tokenizer(:llama3_8b_tokenizer)
```

## Featured Models

_Generated from registry metadata via `tools/sync_readme_models.jl`._

- **`tiktoken`**: `:tiktoken_cl100k_base`, `:tiktoken_o200k_base`
- **`bpe_gpt2`**: `:openai_gpt2_bpe`, `:phi2_bpe`, `:roberta_base_bpe`
- **`sentencepiece_model`**: `:mistral_v3_sentencepiece`, `:xlm_roberta_base_sentencepiece_bpe`
- **`hf_tokenizer_json`**: `:qwen2_5_bpe`
- **`wordpiece_vocab`**: `:bert_base_multilingual_cased_wordpiece`, `:bert_base_uncased_wordpiece`

## Documentation

- [Models](https://mantzaris.github.io/KeemenaSubwords.jl/dev/models/)
- [Formats and Required Files](https://mantzaris.github.io/KeemenaSubwords.jl/dev/formats/)
- [Loading Tokenizers From Local Paths](https://mantzaris.github.io/KeemenaSubwords.jl/dev/loading_local/)
- [LLM Cookbook](https://mantzaris.github.io/KeemenaSubwords.jl/dev/llm_cookbook/)
- [Installable Gated Models](https://mantzaris.github.io/KeemenaSubwords.jl/dev/gated_models/)
- [Troubleshooting](https://mantzaris.github.io/KeemenaSubwords.jl/dev/troubleshooting/)
- [API Reference](https://mantzaris.github.io/KeemenaSubwords.jl/dev/api/)

Run docs locally:

```julia
julia --project=docs docs/make.jl
```
