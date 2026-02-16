# KeemenaSubwords.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/)
[![Build Status](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia-native subword tokenization for BPE, Byte-level BPE, WordPiece, SentencePiece, tiktoken, and Hugging Face `tokenizer.json` workflows.

## Installation

Requires Julia `1.10` or newer.

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

## Key concepts

- KeemenaSubwords token ids are 1-based.
- Use `encode_result(...; return_offsets=true, return_masks=true)` when you need ids plus alignment metadata.
- Offsets are UTF-8 codeunit half-open spans (`[start, stop)`), with `(0, 0)` as the no-span sentinel.
- Start with [Concepts](https://mantzaris.github.io/KeemenaSubwords.jl/dev/concepts/) and the [Normalization and Offsets Contract](https://mantzaris.github.io/KeemenaSubwords.jl/dev/normalization_offsets_contract/).

## Structured encode example

```julia
using KeemenaSubwords

tok = load_tokenizer(:core_bpe_en)
text = "hello world"
tokenization_text = tokenization_view(tok, text)

result = encode_result(
    tok,
    tokenization_text;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

result.ids
result.tokens
result.offsets
result.special_tokens_mask
# Offsets are UTF-8 codeunit spans and may not always be safe Julia string slice boundaries for byte-level tokenizers.
```

## Featured Models

_Generated from registry metadata via `tools/sync_readme_models.jl`._

- **`tiktoken`**: `:tiktoken_cl100k_base`, `:tiktoken_o200k_base`
- **`bpe_gpt2`**: `:openai_gpt2_bpe`, `:phi2_bpe`, `:roberta_base_bpe`
- **`sentencepiece_model`**: `:mistral_v3_sentencepiece`, `:xlm_roberta_base_sentencepiece_bpe`
- **`hf_tokenizer_json`**: `:qwen2_5_bpe`
- **`wordpiece_vocab`**: `:bert_base_multilingual_cased_wordpiece`, `:bert_base_uncased_wordpiece`

## Documentation

- [Concepts](https://mantzaris.github.io/KeemenaSubwords.jl/dev/concepts/)
- [Models](https://mantzaris.github.io/KeemenaSubwords.jl/dev/models/)
- [Normalization and Offsets Contract](https://mantzaris.github.io/KeemenaSubwords.jl/dev/normalization_offsets_contract/)
- [Training (Experimental)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/training/)
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
