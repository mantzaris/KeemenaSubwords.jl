# Tokenizer Formats and Required Files

`detect_tokenizer_format(path)` and `load_tokenizer(path; format=:auto)` use the same detection rules. You can always override detection with `format=...`.

| Format Symbol | Accepted Input | Required Files | Canonical Named Spec Keys | Recommended Call |
| --- | --- | --- | --- | --- |
| `:hf_tokenizer_json` | file or directory | `tokenizer.json` | `tokenizer_json` | `load_hf_tokenizer_json("/path/to/tokenizer.json")` |
| `:bpe_gpt2` | file pair or directory | `vocab.json` + `merges.txt` | `vocab_json`, `merges_txt` | `load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")` |
| `:bpe_encoder` | file pair or directory | `encoder.json` + `vocab.bpe` | `encoder_json`, `vocab_bpe` | `load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")` |
| `:bpe` | directory or `vocab.txt` file | `vocab.txt` + `merges.txt` | `vocab`, `merges` (tuple/spec) | `load_bpe("/path/to/model_dir")` |
| `:bytebpe` | directory or file pair | `vocab.txt` + `merges.txt` | `vocab`, `merges` (tuple/spec) | `load_bytebpe("/path/to/model_dir")` |
| `:wordpiece` / `:wordpiece_vocab` | file or directory | `vocab.txt` | `vocab_txt` | `load_wordpiece("/path/to/vocab.txt")` |
| `:sentencepiece_model` | file or directory | Standard SentencePiece binary `.model`/`.model.v3` files, or Keemena text-exported `.model` files (`spm.model`, `spiece.model`, `tokenizer.model`, `tokenizer.model.v3`, `sentencepiece.bpe.model`) | `model_file` | `load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)` |
| `:tiktoken` | file or directory | `*.tiktoken` or tiktoken-text `tokenizer.model` | `encoding_file` | `load_tiktoken("/path/to/o200k_base.tiktoken")` |
| `:unigram` | file or directory | `unigram.tsv` | `path` | `load_unigram("/path/to/unigram.tsv")` |

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
