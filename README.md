# KeemenaSubwords.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mantzaris.github.io/KeemenaSubwords.jl/dev/)
[![Build Status](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mantzaris/KeemenaSubwords.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia-native subword tokenization and tokenizer loading for common LLM and NLP workflows.

KeemenaSubwords provides loaders and tokenization primitives for:

- classic BPE
- byte-level BPE (GPT-2 style)
- WordPiece
- SentencePiece
- tiktoken
- Hugging Face `tokenizer.json`

It also includes a small registry of practical pretrained tokenizer assets (some shipped, some downloadable, some gated), plus optional experimental tokenizer training.

## Installation

Requires Julia `1.10` or newer.

If/when registered:

```julia
] add KeemenaSubwords
```

Development install (always works):

```julia
] add https://github.com/mantzaris/KeemenaSubwords.jl
```

## Quick start

### Load a tokenizer and do a simple round-trip

Use `tokenize` when you want readable pieces. Use `encode` when you want integer ids.

```julia
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)

token_pieces = tokenize(tokenizer, "hello world")
token_ids = encode(tokenizer, "hello world"; add_special_tokens=true)
decoded_text = decode(tokenizer, token_ids)

(token_pieces=token_pieces, token_ids=token_ids, decoded_text=decoded_text)
```

### One-call helpers for common workflows

If you want a quick, inspectable output bundle (ids, pieces, offsets, masks), use the quick handlers:

```julia
using KeemenaSubwords

output = quick_tokenize(:core_bpe_en, "hello world")

(token_pieces=output.token_pieces, token_ids=output.token_ids, decoded_text=output.decoded_text)
```

Batch encoding (per-sequence outputs, no padding):

```julia
using KeemenaSubwords

batch_output = quick_encode_batch(:core_wordpiece_en, ["hello world", "hello"])

sequence_lengths = batch_output.sequence_lengths
first_tokens = batch_output.results[1].tokens
(first_sequence_lengths=sequence_lengths, first_tokens_preview=first(first_tokens, min(8, length(first_tokens))))
```

Training-ready matrices for causal language modeling (padded ids, attention mask, labels):

```julia
using KeemenaSubwords

training_batch = quick_causal_lm_batch(:core_wordpiece_en, ["hello world", "hello"])

(ids_size=size(training_batch.ids), labels_size=size(training_batch.labels), pad_token_id=training_batch.pad_token_id)
```

### Discover models and download what you need

You can list model keys and inspect provenance/metadata before using them:

```julia
using KeemenaSubwords

available_models()
available_models(shipped=true)
available_models(distribution=:artifact_public)
available_models(distribution=:installable_gated)

describe_model(:qwen2_5_bpe)
recommended_defaults_for_llms()
```

To download (prefetch) tokenizer assets ahead of time:

```julia
using KeemenaSubwords

model_keys = recommended_defaults_for_llms()
prefetch_models(model_keys)
```

### Gated models (explicit install)

Some upstream models require an access token. KeemenaSubwords makes this explicit:

```julia
using KeemenaSubwords

install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
tokenizer = load_tokenizer(:llama3_8b_tokenizer)
```

### Bring your own tokenizer files

You can load local tokenizer assets by pointing at a directory or file. Auto-detection is supported, and you can override with `format=...` when needed.

```julia
using KeemenaSubwords

# auto-detect from a local directory
tokenizer = load_tokenizer("/path/to/model_directory")

# explicit loaders (useful when you know the format)
tokenizer = load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")
tokenizer = load_sentencepiece("/path/to/tokenizer.model")
tokenizer = load_tiktoken("/path/to/tokenizer.model")
tokenizer = load_hf_tokenizer_json("/path/to/tokenizer.json")

# format override
tokenizer = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```

### Train your own tokenizer (experimental)

Tokenizer training is available but currently experimental. The quickest path is `quick_train_bundle`, which writes a small bundle you can reload later.

```julia
using KeemenaSubwords

training_output = quick_train_bundle(
    ["hello world", "hello tokenizer", "world tokenizer"];
    vocab_size=256,
    min_frequency=1,
)

trained_tokenizer = training_output.tokenizer
token_ids = encode(trained_tokenizer, "hello world"; add_special_tokens=false)

(bundle_directory=training_output.bundle_directory, token_ids=token_ids)
```

If you want a Hugging Face compatible `tokenizer.json` export:

```julia
using KeemenaSubwords

export_tokenizer(trained_tokenizer, "my_tokenizer_export"; format=:hf_tokenizer_json)
reloaded_tokenizer = load_hf_tokenizer_json("my_tokenizer_export/tokenizer.json")
```

## Works well with KeemenaPreprocessing.jl

KeemenaPreprocessing.jl focuses on corpus-level cleaning, normalization, vocabulary building, and aligned preprocessing bundles. KeemenaSubwords focuses on subword tokenizers (LLM-oriented ids, offsets, masks, batching helpers).

Integration is simple because KeemenaSubwords tokenizers are callable:

```julia
using KeemenaPreprocessing
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)

config = PreprocessConfiguration(tokenizer_name = keemena_callable(tokenizer))
bundle = preprocess_corpus(["hello world", "hello keemena"]; config=config)

subword_level = level_key(tokenizer)
subword_corpus = get_corpus(bundle, subword_level)
```

If you care about strict normalization and span alignment workflows, the docs describe the recommended contract:
`clean_text -> tokenization_view(tokenizer, clean_text) -> encode_result(...; assume_normalized=true, return_offsets=true, return_masks=true)`.

## Key concepts

- Token ids in KeemenaSubwords are 1-based.
- `tokenize(tokenizer, text)` returns token pieces (`Vector{String}`); `encode(tokenizer, text)` returns token ids (`Vector{Int}`); `decode(tokenizer, ids)` maps ids back to text.
- `add_special_tokens=true` requests model-specific boundary tokens (for example BOS/EOS or CLS/SEP). Inserted specials typically have sentinel offsets `(0, 0)`.
- Offsets (when requested) are 1-based UTF-8 codeunit half-open spans `[start, stop)`. `(0, 0)` is the no-span sentinel.
- For byte-level tokenizers, offsets are valid codeunit spans but may not always be safe Julia string slice boundaries on multibyte text.
- Built-in models come from a registry: some are shipped, some are downloadable artifacts, and some are gated and require an explicit `install_model!`.
- For repeated use, prefer caching helpers like `get_tokenizer_cached(...)` and clear when needed via `clear_tokenizer_cache!()`.

## Featured models

KeemenaSubwords includes a registry of ready-to-use model keys. Many are downloadable automatically (as Julia artifacts), and some are gated by upstream licensing (installed explicitly). You can always discover the full inventory with `available_models()`.

A small sample of commonly useful keys:

- tiktoken: `:tiktoken_cl100k_base`, `:tiktoken_o200k_base`
- bpe_gpt2: `:openai_gpt2_bpe`, `:phi2_bpe`, `:roberta_base_bpe`
- sentencepiece_model: `:mistral_v3_sentencepiece`, `:xlm_roberta_base_sentencepiece_bpe`
- hf_tokenizer_json: `:qwen2_5_bpe`
- wordpiece_vocab: `:bert_base_multilingual_cased_wordpiece`, `:bert_base_uncased_wordpiece`

Example:

```julia
using KeemenaSubwords

prefetch_models([:tiktoken_o200k_base])
tokenizer = load_tokenizer(:tiktoken_o200k_base)
```

## Documentation

- Home / Start Here: https://mantzaris.github.io/KeemenaSubwords.jl/dev/
- Concepts: https://mantzaris.github.io/KeemenaSubwords.jl/dev/concepts/
- Quick Guide Recipes (recommended onboarding): https://mantzaris.github.io/KeemenaSubwords.jl/dev/quick_guide_recipes/
- Built-In Models: https://mantzaris.github.io/KeemenaSubwords.jl/dev/models/
- Loading: https://mantzaris.github.io/KeemenaSubwords.jl/dev/loading/
- Loading Local: https://mantzaris.github.io/KeemenaSubwords.jl/dev/loading_local/
- Structured Outputs and Batching: https://mantzaris.github.io/KeemenaSubwords.jl/dev/structured_outputs_and_batching/
- Normalization and Offsets Contract: https://mantzaris.github.io/KeemenaSubwords.jl/dev/normalization_offsets_contract/
- Integration (KeemenaPreprocessing): https://mantzaris.github.io/KeemenaSubwords.jl/dev/integration/
- Training (Experimental): https://mantzaris.github.io/KeemenaSubwords.jl/dev/training/
- Gated Models: https://mantzaris.github.io/KeemenaSubwords.jl/dev/gated_models/
- Troubleshooting: https://mantzaris.github.io/KeemenaSubwords.jl/dev/troubleshooting/
- API Reference: https://mantzaris.github.io/KeemenaSubwords.jl/dev/api/
