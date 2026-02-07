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
- SentencePiece `.model` compatibility wrapper (Unigram + BPE lightweight format)

Built-in core models:
- `:core_bpe_en`
- `:core_wordpiece_en`
- `:core_sentencepiece_unigram_en`

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

tokenize(bpe, "hello world")
tokenize(wp, "hello keemena subwords")
tokenize(sp, "hello world")
```

## Loading options

```julia
# model registry
available_models()
describe_model(:core_bpe_en)

# path loading
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")

# explicit BPE paths
load_tokenizer(("/path/to/vocab.txt", "/path/to/merges.txt"))
```

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

The API surface exists for forward compatibility:
- `train_bpe(...)`
- `train_unigram(...)`
- `train_wordpiece(...)`

These methods currently raise `ArgumentError` (not implemented yet).
