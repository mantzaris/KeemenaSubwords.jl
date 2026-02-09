# KeemenaSubwords.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/)
[![Build Status](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml?query=branch%3Amain)

Subword tokenization methods in Julia for Keemena* preprocessing pipelines.

Implemented core scope from plan sections 1-3:
- Classic BPE
- Byte-level BPE (GPT-2 style byte mapping)
- WordPiece
- Unigram LM
- SentencePiece `.model` compatibility wrapper (lightweight text + protobuf unigram loading)

Built-in model keys currently exposed by the registry:
- `:bert_base_uncased_wordpiece`
- `:core_bpe_en`
- `:core_sentencepiece_unigram_en`
- `:core_wordpiece_en`
- `:mistral_v1_sentencepiece`
- `:mistral_v3_sentencepiece`
- `:openai_gpt2_bpe`
- `:phi2_bpe`
- `:roberta_base_bpe`
- `:t5_small_sentencepiece_unigram`
- `:tiktoken_cl100k_base`
- `:tiktoken_o200k_base`
- `:tiktoken_p50k_base`
- `:tiktoken_r50k_base`
- `:xlm_roberta_base_sentencepiece_bpe`

## Installation

```julia
] add https://github.com/mantzaris/KeemenaSubwords.jl
```

## Quick start

```julia
using KeemenaSubwords

bpe = load_tokenizer(:core_bpe_en)
wp = load_tokenizer(:core_wordpiece_en)
sp = load_tokenizer(:core_sentencepiece_unigram_en)
tiktoken = load_tokenizer(:tiktoken_cl100k_base)

tokenize(bpe, "hello world")
tokenize(wp, "hello keemena subwords")
tokenize(sp, "hello world")
decode(tiktoken, encode(tiktoken, "hello world"))
```

## Loading options

```julia
# model registry
available_models()
describe_model(:core_bpe_en)
describe_model(:tiktoken_cl100k_base)

# path loading
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")
load_tokenizer("/path/to/o200k_base.tiktoken")

# explicit BPE paths
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"))
```

## Artifact-backed vs in-repo models

`load_tokenizer(:model_key)` resolves tokenizer assets from `Artifacts.toml` first.  
Only the tiny `:core_*` models are shipped in-repo as fallbacks under `models/`.

```julia
using KeemenaSubwords

status = prefetch_models([
    :tiktoken_o200k_base,
    :tiktoken_cl100k_base,
    :openai_gpt2_bpe,
    :bert_base_uncased_wordpiece,
    :t5_small_sentencepiece_unigram,
    :mistral_v1_sentencepiece,
    :mistral_v3_sentencepiece,
    :phi2_bpe,
    :roberta_base_bpe,
    :xlm_roberta_base_sentencepiece_bpe,
])
```

`prefetch_models(...)` triggers lazy artifact installation up front so later calls to `load_tokenizer(:key)` work offline.

Artifact build helper for maintainers:
- `julia --project=. tools/build_public_model_artifact.jl`
- `julia --project=. tools/build_curated_model_artifacts.jl`

## KeemenaPreprocessing integration

```julia
using KeemenaPreprocessing
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)
cfg = PreprocessConfiguration(tokenizer_name = keemena_callable(tokenizer))
bundle = preprocess_corpus(["hello world"]; config=cfg)

# Callables are stored under Symbol(typeof(tokenizer))
lvl = level_key(tokenizer)
subword_corpus = get_corpus(bundle, lvl)
```

## Save and export

```julia
tok = load_tokenizer(:core_wordpiece_en)
save_tokenizer(tok, "out/wp")                          # internal format
export_tokenizer(tok, "out/wp_vocab"; format=:wordpiece_vocab)

bpe = load_tokenizer(:core_bpe_en)
export_tokenizer(bpe, "out/gpt2_style"; format=:bpe_gpt2)
```

## Training entrypoints

Implemented baseline trainers:
- `train_bpe(...)`
- `train_unigram(...)`

Planned later:
- `train_wordpiece(...)` (currently raises `ArgumentError`)
