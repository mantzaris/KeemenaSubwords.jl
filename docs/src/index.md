# KeemenaSubwords.jl

Downstream of [KeemenaPreprocessing.jl](https://github.com/mantzaris/KeemenaPreprocessing.jl).

Current implemented scope:
- classic BPE,
- byte-level BPE,
- WordPiece,
- Unigram,
- SentencePiece wrapper loading from `.model`.

## Installation

```julia
] add https://github.com/mantzaris/KeemenaSubwords.jl
```

## Quick Start

```julia
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)
pieces = tokenize(tokenizer, "hello world")
ids = encode(tokenizer, "hello world"; add_special_tokens=true)
text = decode(tokenizer, ids)

level_key(tokenizer)
```

## Built-in models

```julia
available_models()
available_models(format=:bpe_gpt2)
available_models(format=:hf_tokenizer_json)
available_models(family=:mistral)
available_models(shipped=true)
describe_model(:core_bpe_en)
describe_model(:core_wordpiece_en)
describe_model(:core_sentencepiece_unigram_en)
describe_model(:tiktoken_cl100k_base)
describe_model(:openai_gpt2_bpe)
describe_model(:mistral_v1_sentencepiece)
describe_model(:phi2_bpe)
describe_model(:qwen2_5_bpe)
recommended_defaults_for_llms()
```

For the generated full inventory table, see the **Built-In Models** page.
Use `available_models(distribution=:artifact_public)` for public artifact-backed built-ins and `available_models(distribution=:installable_gated)` for gated installable keys.

Prefetch artifacts (when available) for offline use:

```julia
prefetch_models([
    :tiktoken_cl100k_base,
    :openai_gpt2_bpe,
    :mistral_v3_sentencepiece,
    :qwen2_5_bpe,
])
```

Only tiny `:core_*` assets ship in this repository. Most real model files are lazy artifacts resolved from `Artifacts.toml`. Gated assets are supported via external user-supplied paths.

Use pure-Julia Hugging Face tokenizer loading when a model ships only `tokenizer.json`:

```julia
load_tokenizer("/path/to/tokenizer.json"; format=:hf_tokenizer_json)
```

For gated assets (for example LLaMA), use user-managed files and optional cache helpers:

```julia
register_local_model!(
    :my_llama,
    "/path/to/tokenizer.model";
    format=:sentencepiece_model,
    family=:llama,
    description="Local LLaMA tokenizer",
)

download_hf_files(
    "meta-llama/Llama-3.1-8B",
    ["tokenizer.model"];
    revision="main",
    token=get(ENV, "HF_TOKEN", nothing),
)

install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
load_tokenizer(:llama3_8b_tokenizer)
```

## KeemenaPreprocessing Integration

```julia
using KeemenaPreprocessing
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)
cfg = PreprocessConfiguration(tokenizer_name = keemena_callable(tokenizer))
bundle = preprocess_corpus(["hello world"]; config=cfg)

lvl = level_key(tokenizer)
subword_corpus = get_corpus(bundle, lvl)
```

## Save and Export

```julia
tok = load_tokenizer(:core_wordpiece_en)
save_tokenizer(tok, "out/wp")
export_tokenizer(tok, "out/wp_vocab"; format=:wordpiece_vocab)

bpe = load_tokenizer(:core_bpe_en)
export_tokenizer(bpe, "out/bpe"; format=:bpe_gpt2)
```

## Training API

```julia
train_bpe(corpus; vocab_size=30_000)
train_unigram(corpus; vocab_size=32_000)
train_wordpiece(corpus; vocab_size=30_000)
```

Current status:
- `train_bpe` and `train_unigram` are implemented baseline trainers.
- `train_wordpiece` is still intentionally unimplemented and throws `ArgumentError`.

## API

```@autodocs
Modules = [KeemenaSubwords]
Private = false
```
