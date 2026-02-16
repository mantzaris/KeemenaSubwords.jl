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

The inventory below is generated from the same registry used by `available_models()` and `describe_model(...)`.

_Generated from the registry by `tools/sync_readme_models.jl` (excluding `:user_local` entries)._

### `bpe` / `core`

- `:core_bpe_en`
  - Distribution: `shipped`
  - License: `MIT`
  - Upstream: `in-repo/core` @ `in-repo:core`
  - Expected files: vocab.txt, merges.txt
  - Description: Tiny built-in English classic BPE model (vocab.txt + merges.txt).

### `bpe_gpt2` / `openai`

- `:openai_gpt2_bpe`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `openaipublic/gpt-2` @ `openaipublic:gpt-2/encodings/main`
  - Expected files: vocab.json + merges.txt, encoder.json + vocab.bpe
  - Description: OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe).

### `bpe_gpt2` / `phi`

- `:phi2_bpe`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `microsoft/phi-2` @ `huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0`
  - Expected files: vocab.json + merges.txt, encoder.json + vocab.bpe
  - Description: Microsoft Phi-2 GPT2-style tokenizer files (vocab.json + merges.txt).

### `bpe_gpt2` / `roberta`

- `:roberta_base_bpe`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `FacebookAI/roberta-base` @ `huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b`
  - Expected files: vocab.json + merges.txt, encoder.json + vocab.bpe
  - Description: RoBERTa-base byte-level BPE tokenizer files (vocab.json + merges.txt).

### `hf_tokenizer_json` / `llama`

- `:llama3_8b_tokenizer`
  - Distribution: `installable_gated`
  - License: `Llama-3.1-Community-License`
  - Upstream: `meta-llama/Meta-Llama-3-8B-Instruct` @ `huggingface:meta-llama/Meta-Llama-3-8B-Instruct@main`
  - Expected files: tokenizer.json (preferred), vocab.json + merges.txt (fallback)
  - Description: Meta Llama 3 8B tokenizer (gated; install with install_model!).

### `hf_tokenizer_json` / `qwen`

- `:qwen2_5_bpe`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `Qwen/Qwen2.5-7B` @ `huggingface:Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796`
  - Expected files: tokenizer.json (preferred), vocab.json + merges.txt (fallback)
  - Description: Qwen2.5 BPE tokenizer assets (tokenizer.json with vocab/merges fallback).

### `sentencepiece_model` / `core`

- `:core_sentencepiece_unigram_en`
  - Distribution: `shipped`
  - License: `MIT`
  - Upstream: `in-repo/core` @ `in-repo:core`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: Tiny built-in SentencePiece Unigram model (.model).

### `sentencepiece_model` / `llama`

- `:llama2_tokenizer`
  - Distribution: `installable_gated`
  - License: `Llama-2-Community-License`
  - Upstream: `meta-llama/Llama-2-7b-hf` @ `huggingface:meta-llama/Llama-2-7b-hf@main`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: Meta Llama 2 tokenizer (gated; install with install_model!).

### `sentencepiece_model` / `mistral`

- `:mistral_v1_sentencepiece`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: Mistral/Mixtral tokenizer.model SentencePiece model.
- `:mistral_v3_sentencepiece`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `mistralai/Mistral-7B-Instruct-v0.3` @ `huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: Mistral-7B-Instruct-v0.3 tokenizer.model.v3 SentencePiece model.

### `sentencepiece_model` / `t5`

- `:t5_small_sentencepiece_unigram`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `google-t5/t5-small` @ `huggingface:google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: Hugging Face google-t5/t5-small SentencePiece model (Unigram).

### `sentencepiece_model` / `xlm_roberta`

- `:xlm_roberta_base_sentencepiece_bpe`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `FacebookAI/xlm-roberta-base` @ `huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`
  - Expected files: spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model
  - Description: XLM-RoBERTa-base sentencepiece.bpe.model file.

### `tiktoken` / `openai`

- `:tiktoken_cl100k_base`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `openaipublic/encodings` @ `openaipublic:encodings/cl100k_base.tiktoken`
  - Expected files: *.tiktoken or tokenizer.model (tiktoken text)
  - Description: OpenAI tiktoken cl100k_base encoding.
- `:tiktoken_o200k_base`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `openaipublic/encodings` @ `openaipublic:encodings/o200k_base.tiktoken`
  - Expected files: *.tiktoken or tokenizer.model (tiktoken text)
  - Description: OpenAI tiktoken o200k_base encoding.
- `:tiktoken_p50k_base`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `openaipublic/encodings` @ `openaipublic:encodings/p50k_base.tiktoken`
  - Expected files: *.tiktoken or tokenizer.model (tiktoken text)
  - Description: OpenAI tiktoken p50k_base encoding.
- `:tiktoken_r50k_base`
  - Distribution: `artifact_public`
  - License: `MIT`
  - Upstream: `openaipublic/encodings` @ `openaipublic:encodings/r50k_base.tiktoken`
  - Expected files: *.tiktoken or tokenizer.model (tiktoken text)
  - Description: OpenAI tiktoken r50k_base encoding.

### `wordpiece_vocab` / `bert`

- `:bert_base_multilingual_cased_wordpiece`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `google-bert/bert-base-multilingual-cased` @ `huggingface:google-bert/bert-base-multilingual-cased@3f076fdb1ab68d5b2880cb87a0886f315b8146f8`
  - Expected files: vocab.txt
  - Description: Hugging Face bert-base-multilingual-cased WordPiece vocabulary.
- `:bert_base_uncased_wordpiece`
  - Distribution: `artifact_public`
  - License: `Apache-2.0`
  - Upstream: `bert-base-uncased` @ `huggingface:bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`
  - Expected files: vocab.txt
  - Description: Hugging Face bert-base-uncased WordPiece vocabulary.

### `wordpiece_vocab` / `core`

- `:core_wordpiece_en`
  - Distribution: `shipped`
  - License: `MIT`
  - Upstream: `in-repo/core` @ `in-repo:core`
  - Expected files: vocab.txt
  - Description: Tiny built-in English WordPiece model.

`describe_model(key)` includes provenance metadata such as `license`, `family`, `distribution`, `upstream_repo`, `upstream_ref`, and `upstream_files`.

Built-ins resolve from artifact paths when present, with in-repo fallback model files only for tiny `:core_*` assets.

```julia
prefetch_models(recommended_defaults_for_llms())
```
