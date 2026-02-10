# Installable Gated Models

KeemenaSubwords supports gated tokenizers (for example some LLaMA variants) through opt-in installation into your local cache.

## What this does

`install_model!(key; token=...)`:
- downloads only tokenizer files from upstream with your credentials,
- stores them under your cache (`KEEMENA_SUBWORDS_CACHE_DIR` override supported),
- registers them locally so `load_tokenizer(:key)` works.

## What this does not do

- No gated tokenizer files are redistributed in this repository.
- No gated files are published in `Artifacts.toml`.
- No silent background downloads happen during `load_tokenizer(:key)`.

## Install flow

```julia
# LLaMA 2
install_model!(:llama2_tokenizer; token=ENV["HF_TOKEN"])

# LLaMA 3 8B
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])

# then load by key
llama3 = load_tokenizer(:llama3_8b_tokenizer)
```

If you already downloaded tokenizer files elsewhere, you can skip `install_model!` and load/register directly:

```julia
llama2 = load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
llama3 = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)

register_local_model!(:llama3_local, "/path/to/tokenizer.model"; format=:tiktoken)
```

## Discover gated keys

```julia
available_models(distribution=:installable_gated)
describe_model(:llama3_8b_tokenizer)
```
