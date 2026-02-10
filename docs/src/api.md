# API Reference

## Explicit Loader APIs

```@docs
load_bpe
load_bytebpe
load_bpe_gpt2
load_bpe_encoder
load_unigram
load_wordpiece
load_sentencepiece
load_tiktoken
load_hf_tokenizer_json
load_tokenizer
detect_tokenizer_format
detect_tokenizer_files
```

## Registry and Installation APIs

```@docs
available_models
describe_model
model_path
prefetch_models
register_local_model!
install_model!
install_llama2_tokenizer!
install_llama3_8b_tokenizer!
download_hf_files
recommended_defaults_for_llms
```

`register_external_model!` remains available as a deprecated compatibility alias; prefer `register_local_model!` in new code.

## Full Exported API

```@autodocs
Modules = [KeemenaSubwords]
Private = false
```
