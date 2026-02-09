# Troubleshooting

## Auto-detect picked the wrong format

Force the format explicitly:

```julia
load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```

Use detection helpers to inspect first:

```julia
detect_tokenizer_format("/path/to/model_dir")
detect_tokenizer_files("/path/to/model_dir")
```

## Missing `merges.txt`

For GPT-2 style BPE you need both files:

- `vocab.json` + `merges.txt`
- or `encoder.json` + `vocab.bpe`

Use explicit loaders for clearer errors:

```julia
load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")
load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")
```

## `tokenizer.model` is not SentencePiece

Some models (notably LLaMA3-style releases) provide tiktoken text in a file named `tokenizer.model`.

KeemenaSubwords sniffs `.model` files:
- tiktoken-like text lines => `:tiktoken`
- binary / SentencePiece-like payload => `:sentencepiece_model`

If needed, override manually:

```julia
load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```

## Gated model key fails to load

If `load_tokenizer(:llama3_8b_tokenizer)` says not installed, install first:

```julia
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
```

You must have accepted upstream license terms and have valid access.
