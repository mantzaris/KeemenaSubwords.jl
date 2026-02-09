# Built-In Models

```julia
using KeemenaSubwords

available_models()
available_models(format=:tiktoken)
available_models(format=:bpe_gpt2)
available_models(format=:hf_tokenizer_json)
available_models(family=:qwen)
available_models(family=:mistral)
available_models(shipped=true)

describe_model(:core_bpe_en)
describe_model(:core_wordpiece_en)
describe_model(:core_sentencepiece_unigram_en)
describe_model(:tiktoken_o200k_base)
describe_model(:openai_gpt2_bpe)
describe_model(:bert_base_uncased_wordpiece)
describe_model(:bert_base_multilingual_cased_wordpiece)
describe_model(:t5_small_sentencepiece_unigram)
describe_model(:mistral_v1_sentencepiece)
describe_model(:mistral_v3_sentencepiece)
describe_model(:phi2_bpe)
describe_model(:qwen2_5_bpe)
describe_model(:roberta_base_bpe)
describe_model(:xlm_roberta_base_sentencepiece_bpe)
recommended_defaults_for_llms()

model_path(:core_bpe_en)
```

`describe_model(key)` includes provenance metadata such as `license`, `family`, `upstream_ref`, and `upstream_files`.

Built-ins resolve from artifact paths when present, with in-repo fallback model files only for tiny `:core_*` assets.

```julia
prefetch_models(recommended_defaults_for_llms())
```
