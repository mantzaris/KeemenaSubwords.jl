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

Planned (stub entrypoints):
- `train_wordpiece(...)`
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
train_sentencepiece
```

Current behavior:
- WordPiece and SentencePiece training entrypoints exist for discoverability and
  currently throw clear `ArgumentError` messages because algorithms are not yet
implemented.

The pretrained-tokenizer APIs (`load_tokenizer`, `tokenize`, `encode`,
`encode_result`, `decode`) remain stable and independent from training codepaths.
