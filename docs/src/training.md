# Training (Experimental)

Training support is currently experimental and intentionally separated from the
pretrained tokenizer loading/encoding workflows.

```@meta
CurrentModule = KeemenaSubwords.Training
```

Available now:
- `train_bpe(...)`
- `train_bytebpe(...)`
- `train_unigram(...)`
- `train_wordpiece(...)`
- `train_wordpiece_result(...)`

Planned (stub entrypoints):
- `train_sentencepiece(...)`

## Training API

```@docs
train_bpe
train_bpe_result
train_bytebpe
train_bytebpe_result
train_unigram
train_unigram_result
train_wordpiece
train_wordpiece_result
train_sentencepiece
```

## Note on pretokenizer

- `pretokenizer` is used only during training to split input text into units
  for frequency counts.
- Trained tokenizers do not persist or apply the training `pretokenizer` at
  runtime.
- For consistent behavior, apply equivalent preprocessing upstream (for
  example via KeemenaPreprocessing) before calling `encode`/`encode_result`.
- ByteBPE exports as `vocab.txt + merges.txt`; when reloading exported files,
  use `format=:bytebpe` if format auto-detection is ambiguous.

Current behavior:
- SentencePiece training remains a discoverability stub and currently throws a
  clear `ArgumentError` because training is not yet implemented.
- Unigram training defaults to SentencePiece-style `whitespace_marker="▁"` so
  multi-word text can round-trip through `decode(encode(...))`.
- If `whitespace_marker=""`, runtime Unigram tokenization is still word-split,
  so decoding may collapse spaces in multi-word text (for example
  `"hello world"` -> `"helloworld"`).

The pretrained-tokenizer APIs (`load_tokenizer`, `tokenize`, `encode`,
`encode_result`, `decode`) remain stable and independent from training codepaths.
