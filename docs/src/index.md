# KeemenaSubwords.jl

Downstream of [KeemenaPreprocessing.jl](https://github.com/mantzaris/KeemenaPreprocessing.jl).

KeemenaSubwords provides Julia-native loaders and tokenization primitives for:
- classic BPE,
- byte-level BPE,
- WordPiece,
- SentencePiece,
- tiktoken,
- Hugging Face `tokenizer.json`.

## Quick Start

```julia
using KeemenaSubwords

tok = load_tokenizer(:core_bpe_en)
pieces = tokenize(tok, "hello world")
ids = encode(tok, "hello world"; add_special_tokens=true)
text = decode(tok, ids)
```

## Model Discovery

```julia
available_models()
available_models(distribution=:artifact_public)
available_models(distribution=:installable_gated)
describe_model(:qwen2_5_bpe)
recommended_defaults_for_llms()
```

## Key Workflows

```julia
# local path auto-detection
load_tokenizer("/path/to/model_dir")

# explicit loaders
load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")
load_sentencepiece("/path/to/tokenizer.model")
load_tiktoken("/path/to/tokenizer.model")

# gated install flow
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
```

## Documentation Map

- [Built-in model inventory](models.md)
- [Normalization and offsets contract](normalization_offsets_contract.md)
- [Training (experimental)](training.md)
- [Format contracts](formats.md)
- [Local path recipes](loading_local.md)
- [LLM cookbook and tokenizer.json roadmap](llm_cookbook.md)
- [Installable gated models](gated_models.md)
- [Troubleshooting](troubleshooting.md)
- [API reference](api.md)

## KeemenaPreprocessing Integration

```julia
using KeemenaPreprocessing
using KeemenaSubwords

tok = load_tokenizer(:core_bpe_en)
cfg = PreprocessConfiguration(tokenizer_name = keemena_callable(tok))
bundle = preprocess_corpus(["hello world"]; config=cfg)
```

See [API reference](api.md) for explicit loader APIs and the full exported reference.
