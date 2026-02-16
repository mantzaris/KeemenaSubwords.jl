# Tokenizer Formats and Required Files

`detect_tokenizer_format(path)` and `load_tokenizer(path; format=:auto)` use the same detection rules. You can always override detection with `format=...`.

## `:hf_tokenizer_json`

- Format symbol: `:hf_tokenizer_json`
- Accepted input: file or directory
- Required files: `tokenizer.json`
- Canonical named spec key: `path` (alias: `tokenizer_json`)
- Recommended loader:
  `load_hf_tokenizer_json("/path/to/tokenizer.json")`
- Offset behavior: depends on pipeline components. If ByteLevel is configured, apply byte-level offset caveats.

## `:bpe_gpt2`

- Format symbol: `:bpe_gpt2`
- Accepted input: file pair or directory
- Required files: `vocab.json` + `merges.txt`
- Canonical named spec keys: `vocab_json`, `merges_txt`
- Recommended loader:
  `load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")`
- Offset behavior: byte-level family. Offsets are codeunit spans and may not always be safe Julia string slice boundaries on multibyte text.

## `:bpe_encoder`

- Format symbol: `:bpe_encoder`
- Accepted input: file pair or directory
- Required files: `encoder.json` + `vocab.bpe`
- Canonical named spec keys: `encoder_json`, `vocab_bpe`
- Recommended loader:
  `load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")`
- Offset behavior: byte-level family with the same slicing caveat as `:bpe_gpt2`.

## `:bpe`

- Format symbol: `:bpe`
- Accepted input: directory or `vocab.txt` file
- Required files: `vocab.txt` + `merges.txt`
- Canonical named spec keys: `vocab`, `merges` (tuple/spec)
- Recommended loader:
  `load_bpe("/path/to/model_dir")`
- Offset behavior: non-byte-level in standard usage; spanful offsets are expected to be string-safe.

## `:bytebpe`

- Format symbol: `:bytebpe`
- Accepted input: directory or file pair
- Required files: `vocab.txt` + `merges.txt`
- Canonical named spec keys: `vocab`, `merges` (tuple/spec)
- Recommended loader:
  `load_bytebpe("/path/to/model_dir")`
- Offset behavior: byte-level family. Use `try_span_substring` with `span_codeunits` fallback for non-boundary spans.

## `:wordpiece` / `:wordpiece_vocab`

- Format symbols: `:wordpiece`, `:wordpiece_vocab`
- Accepted input: file or directory
- Required files: `vocab.txt`
- Canonical named spec key: `path` (alias: `vocab_txt`)
- Recommended loader:
  `load_wordpiece("/path/to/vocab.txt")`
- Offset behavior: non-byte-level; spanful offsets are expected to be string-safe in normal usage.

## `:sentencepiece_model`

- Format symbol: `:sentencepiece_model`
- Accepted input: file or directory
- Required files:
  standard SentencePiece binary `.model` / `.model.v3`,
  or Keemena text-exported model files
  (`spm.model`, `spiece.model`, `tokenizer.model`, `tokenizer.model.v3`, `sentencepiece.bpe.model`)
- Canonical named spec key: `path` (alias: `model_file`)
- Recommended loader:
  `load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)`
- Offset behavior: usually non-byte-level. Spanful offsets are expected to be string-safe for standard SentencePiece pipelines.

## `:tiktoken`

- Format symbol: `:tiktoken`
- Accepted input: file or directory
- Required files: `*.tiktoken` or tiktoken-text `tokenizer.model`
- Canonical named spec key: `path` (alias: `encoding_file`)
- Recommended loader:
  `load_tiktoken("/path/to/o200k_base.tiktoken")`
- Offset behavior: byte-level family with non-boundary span cases on multibyte text.

## `:unigram` / `:unigram_tsv`

- Format symbols: `:unigram`, `:unigram_tsv`
- Accepted input: file or directory
- Required files: `unigram.tsv`
- Canonical named spec key: `path` (alias: `unigram_tsv`)
- Recommended loader:
  `load_unigram("/path/to/unigram.tsv")`
- Offset behavior: non-byte-level; spanful offsets are expected to be string-safe in normal usage.

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

### BERT Components in `tokenizer.json`

KeemenaSubwords now supports the common Hugging Face BERT pipeline components:
- `normalizer.type = "BertNormalizer"`
- `pre_tokenizer.type = "BertPreTokenizer"`

Offsets for these pipelines follow the same package contract:
- offsets are computed against `tokenization_view(tokenizer, text)` (the
  tokenizer-normalized text),
- inserted post-processor specials keep sentinel `(0,0)`,
- spanful offsets remain 1-based UTF-8 codeunit half-open spans.

KeemenaPreprocessing integration remains:
`tokenization_text = tokenization_view(tokenizer, clean_text)` then
`encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, return_masks=true)`.

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
