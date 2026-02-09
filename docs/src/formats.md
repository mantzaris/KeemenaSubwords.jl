# Tokenizer Formats and Required Files

`detect_tokenizer_format(path)` and `load_tokenizer(path; format=:auto)` use the same detection rules. You can always override detection with `format=...`.

| Format Symbol | Accepted Input | Required Files | Typical Source | Recommended Call |
| --- | --- | --- | --- | --- |
| `:hf_tokenizer_json` | file or directory | `tokenizer.json` | Hugging Face model repos | `load_hf_tokenizer_json("/path/to/tokenizer.json")` |
| `:bpe_gpt2` | file pair or directory | `vocab.json` + `merges.txt` | GPT-2, RoBERTa, Phi-2, Qwen fallback | `load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")` |
| `:bpe_encoder` | file pair or directory | `encoder.json` + `vocab.bpe` | OpenAI GPT-2 blob layout | `load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")` |
| `:bpe` | directory or `vocab.txt` file | `vocab.txt` + `merges.txt` | Keemena classic BPE fixtures | `load_bpe("/path/to/model_dir")` |
| `:bytebpe` | directory or file pair | `vocab.txt` + `merges.txt` | Byte-level custom BPE | `load_bytebpe("/path/to/model_dir")` |
| `:wordpiece` / `:wordpiece_vocab` | file or directory | `vocab.txt` | BERT-style WordPiece | `load_wordpiece("/path/to/vocab.txt")` |
| `:sentencepiece_model` | file or directory | `spm.model` / `spiece.model` / `tokenizer.model` / `tokenizer.model.v3` / `sentencepiece.bpe.model` | SentencePiece models (Unigram/BPE) | `load_sentencepiece("/path/to/tokenizer.model")` |
| `:tiktoken` | file or directory | `*.tiktoken` or text `tokenizer.model` | OpenAI encodings, LLaMA3-style tokenizer text | `load_tiktoken("/path/to/o200k_base.tiktoken")` |
| `:unigram` | file or directory | `unigram.tsv` | Internal unigram format | `load_unigram("/path/to/unigram.tsv")` |

## Detection Notes

- Directory preference order starts with `tokenizer.json`, then GPT-2 pairs, then SentencePiece model names.
- `.model` files are sniffed:
  - tiktoken-like text (`<base64> <int>`) => `:tiktoken`
  - binary or SentencePiece-like content => `:sentencepiece_model`
- If a path is ambiguous, use explicit override:

```julia
load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```
