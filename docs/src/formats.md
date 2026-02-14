# Tokenizer Formats and Required Files

`detect_tokenizer_format(path)` and `load_tokenizer(path; format=:auto)` use the same detection rules. You can always override detection with `format=...`.

| Format Symbol | Accepted Input | Required Files | Canonical Named Spec Keys | Recommended Call |
| --- | --- | --- | --- | --- |
| `:hf_tokenizer_json` | file or directory | `tokenizer.json` | `path` (alias: `tokenizer_json`) | `load_hf_tokenizer_json("/path/to/tokenizer.json")` |
| `:bpe_gpt2` | file pair or directory | `vocab.json` + `merges.txt` | `vocab_json`, `merges_txt` | `load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")` |
| `:bpe_encoder` | file pair or directory | `encoder.json` + `vocab.bpe` | `encoder_json`, `vocab_bpe` | `load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")` |
| `:bpe` | directory or `vocab.txt` file | `vocab.txt` + `merges.txt` | `vocab`, `merges` (tuple/spec) | `load_bpe("/path/to/model_dir")` |
| `:bytebpe` | directory or file pair | `vocab.txt` + `merges.txt` | `vocab`, `merges` (tuple/spec) | `load_bytebpe("/path/to/model_dir")` |
| `:wordpiece` / `:wordpiece_vocab` | file or directory | `vocab.txt` | `path` (alias: `vocab_txt`) | `load_wordpiece("/path/to/vocab.txt")` |
| `:sentencepiece_model` | file or directory | Standard SentencePiece binary `.model`/`.model.v3` files, or Keemena text-exported `.model` files (`spm.model`, `spiece.model`, `tokenizer.model`, `tokenizer.model.v3`, `sentencepiece.bpe.model`) | `path` (alias: `model_file`) | `load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)` |
| `:tiktoken` | file or directory | `*.tiktoken` or tiktoken-text `tokenizer.model` | `path` (alias: `encoding_file`) | `load_tiktoken("/path/to/o200k_base.tiktoken")` |
| `:unigram` | file or directory | `unigram.tsv` | `path` (alias: `unigram_tsv`) | `load_unigram("/path/to/unigram.tsv")` |

## Exporting Hugging Face `tokenizer.json`

Use the HF export target to write `tokenizer.json` from any supported
KeemenaSubwords tokenizer:

```julia
export_tokenizer(tokenizer, "out_dir"; format=:hf_tokenizer_json)
# equivalent via save_tokenizer
save_tokenizer(tokenizer, "out_dir"; format=:hf_tokenizer_json)
```

Reload in Julia:

```julia
reloaded = load_tokenizer("out_dir"; format=:hf_tokenizer_json)
```

Load the same file in Python Fast tokenizers:

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="out_dir/tokenizer.json")
```

Current scope note:
- `tokenizer_config.json` and `special_tokens_map.json` are not emitted yet.
- `TemplateProcessing` is emitted in canonical HF JSON shape (`single`/`pair`
  object items and object-map `special_tokens`) for better external HF
  compatibility.
- Byte-level export writes explicit `ByteLevel` options
  (`add_prefix_space=false`, `trim_offsets=false`, `use_regex=false`) for
  Keemena ByteBPE interoperability, so Python/Rust HF loaders do not silently
  fall back to different defaults.

## Detection Notes

- Directory preference order:
  1) `tokenizer.json`
  2) `vocab.json + merges.txt`
  3) `encoder.json + vocab.bpe`
  4) SentencePiece model filenames
  5) `*.tiktoken`
- `.model` files are sniffed:
  - tiktoken-like text (`<base64> <int>`) => `:tiktoken`
  - binary SentencePiece protobuf payload or Keemena text SentencePiece payload => `:sentencepiece_model`
- LLaMA3-style files often use `tokenizer.model` containing tiktoken text. Use explicit override when needed.

```julia
load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```

## Byte-level behavior

- Byte-level tokenizers:
  - `:bytebpe`
  - `:bpe_gpt2` / `:bpe_encoder`
  - `:tiktoken`
  - some `:hf_tokenizer_json` pipelines when `ByteLevel` is configured.
- Non-byte-level tokenizers:
  - `:wordpiece`
  - `:sentencepiece_model`
  - `:unigram`.

Round-trip expectations:
- Byte-level families are generally robust for arbitrary UTF-8 input and usually satisfy stable `decode(encode(text))`.
- WordPiece/SentencePiece/Unigram operate on learned subword vocabularies; they are deterministic, but unknown-token fallback can reduce exact round-trip fidelity depending on vocab coverage.
