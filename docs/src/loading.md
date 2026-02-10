# Loading Tokenizers

Use `load_tokenizer` for key-based and auto-detected loading, and use explicit constructors when you want strict file contracts.

## Built-in keys

```julia
using KeemenaSubwords

tok = load_tokenizer(:core_bpe_en)
qwen = load_tokenizer(:qwen2_5_bpe)
```

## Auto-detected local paths

```julia
load_tokenizer("/path/to/model_dir")
load_tokenizer("/path/to/tokenizer.model")
load_tokenizer("/path/to/tokenizer.json")
```

## Force format override

```julia
load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```

## Explicit constructors

```julia
load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")
load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")
load_wordpiece("/path/to/vocab.txt")
load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)
load_tiktoken("/path/to/o200k_base.tiktoken")
load_hf_tokenizer_json("/path/to/tokenizer.json")
```

For complete local path recipes, see [Loading Tokenizers From Local Paths](loading_local.md).
For explicit file contracts and named-spec keys, see [Tokenizer Formats and Required Files](formats.md).
