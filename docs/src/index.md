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
available_models(family=:mistral)
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

Public baseline keys:
- `:tiktoken_o200k_base`
- `:tiktoken_cl100k_base`
- `:tiktoken_r50k_base`
- `:tiktoken_p50k_base`
- `:openai_gpt2_bpe`
- `:bert_base_uncased_wordpiece`
- `:t5_small_sentencepiece_unigram`
- `:mistral_v1_sentencepiece`
- `:mistral_v3_sentencepiece`
- `:phi2_bpe`
- `:qwen2_5_bpe`
- `:roberta_base_bpe`
- `:xlm_roberta_base_sentencepiece_bpe`

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
