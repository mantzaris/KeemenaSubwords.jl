# Built-In Models

```julia
using KeemenaSubwords

available_models()
available_models(format=:tiktoken)
available_models(format=:bpe_gpt2)
available_models(format=:hf_tokenizer_json)
available_models(family=:qwen)
available_models(family=:mistral)
available_models(distribution=:artifact_public)
available_models(distribution=:installable_gated)
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
describe_model(:llama3_8b_tokenizer)
recommended_defaults_for_llms()

model_path(:core_bpe_en)
```

The table below is generated from the same registry used by `available_models()` and `describe_model(...)`.

<!-- KEEMENA_MODELS_START -->
_Generated from the registry by `tools/sync_readme_models.jl` (excluding `:user_local` entries)._

### `bpe` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_bpe_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | vocab.txt, merges.txt | Tiny built-in English classic BPE model (vocab.txt + merges.txt). |

### `bpe_gpt2` / `openai`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:openai_gpt2_bpe` | `artifact_public` | `MIT` | `openaipublic/gpt-2` | `openaipublic:gpt-2/encodings/main` | vocab.json + merges.txt, encoder.json + vocab.bpe | OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe). |

### `bpe_gpt2` / `phi`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:phi2_bpe` | `artifact_public` | `MIT` | `microsoft/phi-2` | `huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0` | vocab.json + merges.txt, encoder.json + vocab.bpe | Microsoft Phi-2 GPT2-style tokenizer files (vocab.json + merges.txt). |

### `bpe_gpt2` / `roberta`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:roberta_base_bpe` | `artifact_public` | `MIT` | `FacebookAI/roberta-base` | `huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b` | vocab.json + merges.txt, encoder.json + vocab.bpe | RoBERTa-base byte-level BPE tokenizer files (vocab.json + merges.txt). |

### `hf_tokenizer_json` / `llama`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:llama3_8b_tokenizer` | `installable_gated` | `Llama-3.1-Community-License` | `meta-llama/Meta-Llama-3-8B-Instruct` | `huggingface:meta-llama/Meta-Llama-3-8B-Instruct@main` | tokenizer.json (preferred), vocab.json + merges.txt (fallback) | Meta Llama 3 8B tokenizer (gated; install with install_model!). |

### `hf_tokenizer_json` / `qwen`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:qwen2_5_bpe` | `artifact_public` | `Apache-2.0` | `Qwen/Qwen2.5-7B` | `huggingface:Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796` | tokenizer.json (preferred), vocab.json + merges.txt (fallback) | Qwen2.5 BPE tokenizer assets (tokenizer.json with vocab/merges fallback). |

### `sentencepiece_model` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_sentencepiece_unigram_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Tiny built-in SentencePiece Unigram model (.model). |

### `sentencepiece_model` / `llama`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:llama2_tokenizer` | `installable_gated` | `Llama-2-Community-License` | `meta-llama/Llama-2-7b-hf` | `huggingface:meta-llama/Llama-2-7b-hf@main` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Meta Llama 2 tokenizer (gated; install with install_model!). |

### `sentencepiece_model` / `mistral`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:mistral_v1_sentencepiece` | `artifact_public` | `Apache-2.0` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Mistral/Mixtral tokenizer.model SentencePiece model. |
| `:mistral_v3_sentencepiece` | `artifact_public` | `Apache-2.0` | `mistralai/Mistral-7B-Instruct-v0.3` | `huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Mistral-7B-Instruct-v0.3 tokenizer.model.v3 SentencePiece model. |

### `sentencepiece_model` / `t5`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:t5_small_sentencepiece_unigram` | `artifact_public` | `Apache-2.0` | `google-t5/t5-small` | `huggingface:google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | Hugging Face google-t5/t5-small SentencePiece model (Unigram). |

### `sentencepiece_model` / `xlm_roberta`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:xlm_roberta_base_sentencepiece_bpe` | `artifact_public` | `MIT` | `FacebookAI/xlm-roberta-base` | `huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model | XLM-RoBERTa-base sentencepiece.bpe.model file. |

### `tiktoken` / `openai`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:tiktoken_cl100k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/cl100k_base.tiktoken` | *.tiktoken | OpenAI tiktoken cl100k_base encoding. |
| `:tiktoken_o200k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/o200k_base.tiktoken` | *.tiktoken | OpenAI tiktoken o200k_base encoding. |
| `:tiktoken_p50k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/p50k_base.tiktoken` | *.tiktoken | OpenAI tiktoken p50k_base encoding. |
| `:tiktoken_r50k_base` | `artifact_public` | `MIT` | `openaipublic/encodings` | `openaipublic:encodings/r50k_base.tiktoken` | *.tiktoken | OpenAI tiktoken r50k_base encoding. |

### `wordpiece_vocab` / `bert`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:bert_base_multilingual_cased_wordpiece` | `artifact_public` | `Apache-2.0` | `google-bert/bert-base-multilingual-cased` | `huggingface:google-bert/bert-base-multilingual-cased@3f076fdb1ab68d5b2880cb87a0886f315b8146f8` | vocab.txt | Hugging Face bert-base-multilingual-cased WordPiece vocabulary. |
| `:bert_base_uncased_wordpiece` | `artifact_public` | `Apache-2.0` | `bert-base-uncased` | `huggingface:bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594` | vocab.txt | Hugging Face bert-base-uncased WordPiece vocabulary. |

### `wordpiece_vocab` / `core`

| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `:core_wordpiece_en` | `shipped` | `MIT` | `in-repo/core` | `in-repo:core` | vocab.txt | Tiny built-in English WordPiece model. |
<!-- KEEMENA_MODELS_END -->

`describe_model(key)` includes provenance metadata such as `license`, `family`, `distribution`, `upstream_repo`, `upstream_ref`, and `upstream_files`.

Built-ins resolve from artifact paths when present, with in-repo fallback model files only for tiny `:core_*` assets.

```julia
prefetch_models(recommended_defaults_for_llms())
```
