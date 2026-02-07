# Built-In Models

```julia
using KeemenaSubwords

available_models()

describe_model(:core_bpe_en)
describe_model(:core_wordpiece_en)
describe_model(:core_sentencepiece_unigram_en)

model_path(:core_bpe_en)
```

Built-ins resolve from artifact paths when present, with in-repo fallback model files.
