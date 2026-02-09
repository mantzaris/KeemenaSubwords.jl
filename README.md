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

Built-in and installable model keys currently exposed by the registry:

<!-- KEEMENA_MODELS_START -->
_Generated from the registry by `tools/sync_readme_models.jl` (excluding `:user_local` entries)._

### `bpe` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_bpe_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | vocab.txt, merges.txt | Tiny built-in English classic BPE model (vocab.txt + merges.txt). |

### `bpe_gpt2` / `openai`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:openai_gpt2_bpe` | `artifact_public` | `MIT` | `openaipublic/gpt-2` | `openaipublic:gpt-2/encodings/main` | vocab.json + merges.txt, encoder.json + vocab.bpe | OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe). |

### `bpe_gpt2` / `phi`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:phi2_bpe` | `artifact_public` | `MIT` | `microsoft/phi-2` | `huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0` | vocab.json + merges.txt, encoder.json + vocab.bpe | Microsoft Phi-2 GPT2-style tokenizer files (vocab.json + merges.txt). |

### `bpe_gpt2` / `roberta`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:roberta_base_bpe` | `artifact_public` | `MIT` | `FacebookAI/roberta-base` | `huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b` | vocab.json + merges.txt, encoder.json + vocab.bpe | RoBERTa-base byte-level BPE tokenizer files (vocab.json + merges.txt). |

### `hf_tokenizer_json` / `llama`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:llama3_8b_tokenizer` | `installable_gated` | `Llama-3.1-Community-License` | `meta-llama/Meta-Llama-3-8B-Instruct` | `huggingface:meta-llama/Meta-Llama-3-8B-Instruct@main` | tokenizer.json (preferred), vocab.json + merges.txt (fallback) | Meta Llama 3 8B tokenizer (gated; install with install_model!). |

### `hf_tokenizer_json` / `qwen`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:qwen2_5_bpe` | `artifact_public` | `Apache-2.0` | `Qwen/Qwen2.5-7B` | `huggingface:Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796` | tokenizer.json (preferred), vocab.json + merges.txt (fallback) | Qwen2.5 BPE tokenizer assets (tokenizer.json with vocab/merges fallback). |

### `sentencepiece_model` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_sentencepiece_unigram_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Tiny built-in SentencePiece Unigram model (.model). |

### `sentencepiece_model` / `llama`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:llama2_tokenizer` | `installable_gated` | `Llama-2-Community-License` | `meta-llama/Llama-2-7b-hf` | `huggingface:meta-llama/Llama-2-7b-hf@main` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Meta Llama 2 tokenizer (gated; install with install_model!). |

### `sentencepiece_model` / `mistral`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:mistral_v1_sentencepiece` | `artifact_public` | `Apache-2.0` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Mistral/Mixtral tokenizer.model SentencePiece model. |
| `:mistral_v3_sentencepiece` | `artifact_public` | `Apache-2.0` | `mistralai/Mistral-7B-Instruct-v0.3` | `huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Mistral-7B-Instruct-v0.3 tokenizer.model.v3 SentencePiece model. |

### `sentencepiece_model` / `t5`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:t5_small_sentencepiece_unigram` | `artifact_public` | `Apache-2.0` | `google-t5/t5-small` | `huggingface:google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Hugging Face google-t5/t5-small SentencePiece model (Unigram). |

### `sentencepiece_model` / `xlm_roberta`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:xlm_roberta_base_sentencepiece_bpe` | `artifact_public` | `MIT` | `FacebookAI/xlm-roberta-base` | `huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | XLM-RoBERTa-base sentencepiece.bpe.model file. |

### `tiktoken` / `openai`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:tiktoken_cl100k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/cl100k_base.tiktoken` | *.tiktoken | OpenAI tiktoken cl100k_base encoding. |
| `:tiktoken_o200k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/o200k_base.tiktoken` | *.tiktoken | OpenAI tiktoken o200k_base encoding. |
| `:tiktoken_p50k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/p50k_base.tiktoken` | *.tiktoken | OpenAI tiktoken p50k_base encoding. |
| `:tiktoken_r50k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/r50k_base.tiktoken` | *.tiktoken | OpenAI tiktoken r50k_base encoding. |

### `wordpiece_vocab` / `bert`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:bert_base_multilingual_cased_wordpiece` | `artifact_public` | `Apache-2.0` | `google-bert/bert-base-multilingual-cased` | `huggingface:google-bert/bert-base-multilingual-cased@3f076fdb1ab68d5b2880cb87a0886f315b8146f8` | vocab.txt | Hugging Face bert-base-multilingual-cased WordPiece vocabulary. |
| `:bert_base_uncased_wordpiece` | `artifact_public` | `Apache-2.0` | `bert-base-uncased` | `huggingface:bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` | vocab.txt | Hugging Face bert-base-uncased WordPiece vocabulary. |

### `wordpiece_vocab` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_wordpiece_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | vocab.txt | Tiny built-in English WordPiece model. |
<!-- KEEMENA_MODELS_END -->

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
available_models(format=:bpe_gpt2)
available_models(family=:mistral)
available_models(distribution=:artifact_public)
available_models(distribution=:installable_gated)
available_models(shipped=true)
describe_model(:core_bpe_en)
describe_model(:tiktoken_cl100k_base)
recommended_defaults_for_llms()

# path loading
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/spm.model")
load_tokenizer("/path/to/o200k_base.tiktoken")
load_tokenizer("/path/to/tokenizer.json"; format=:hf_tokenizer_json)

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
    :bert_base_multilingual_cased_wordpiece,
    :t5_small_sentencepiece_unigram,
    :mistral_v1_sentencepiece,
    :mistral_v3_sentencepiece,
    :phi2_bpe,
    :qwen2_5_bpe,
    :roberta_base_bpe,
    :xlm_roberta_base_sentencepiece_bpe,
])
```

`prefetch_models(...)` triggers lazy artifact installation up front so later calls to `load_tokenizer(:key)` work offline.

External user-supplied tokenizers (supported, not shipped):
- Llama 2 style SentencePiece:
  - `load_tokenizer("/path/to/tokenizer.model")`
- Llama 3 style tiktoken:
  - `load_tokenizer("/path/to/llama3_tokenizer.tiktoken"; format=:tiktoken)`
- One-command gated install workflow (requires accepted license + HF token):
  - `install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])`
  - `load_tokenizer(:llama3_8b_tokenizer)`
- Mistral Tekken:
  - No built-in is shipped by default; use user-supplied tiktoken-compatible files.
  - `load_tokenizer("/path/to/tekken_like.tiktoken"; format=:tiktoken)`
- Optional registry helper for local paths:
  - `register_local_model!(:my_llama, "/path/to/tokenizer.model"; format=:sentencepiece_model, family=:llama)`
- Optional authenticated Hugging Face file helper (for user-managed/gated assets):
  - `download_hf_files("meta-llama/Llama-3.1-8B", ["tokenizer.model"]; revision="main", token=get(ENV, "HF_TOKEN", nothing))`

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
