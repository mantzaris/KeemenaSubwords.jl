# Built-In Models

```julia
using KeemenaSubwords

available_models()

describe_model(:core_bpe_en)
describe_model(:core_wordpiece_en)
describe_model(:core_sentencepiece_unigram_en)
describe_model(:tiktoken_o200k_base)
describe_model(:openai_gpt2_bpe)
describe_model(:bert_base_uncased_wordpiece)
describe_model(:t5_small_sentencepiece_unigram)
describe_model(:mistral_v1_sentencepiece)
describe_model(:mistral_v3_sentencepiece)
describe_model(:phi2_bpe)
describe_model(:roberta_base_bpe)
describe_model(:xlm_roberta_base_sentencepiece_bpe)

model_path(:core_bpe_en)
```

Built-ins resolve from artifact paths when present, with in-repo fallback model files.

```julia
prefetch_models([:tiktoken_cl100k_base, :openai_gpt2_bpe])
```
