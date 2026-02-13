# Training (Experimental)

Training support is currently experimental and intentionally separated from the
pretrained tokenizer loading/encoding workflows.

Available now:
- `train_bpe(...)`
- `train_unigram(...)`

Planned (stub entrypoints):
- `train_wordpiece(...)`
- `train_sentencepiece(...)`

Current behavior:
- WordPiece and SentencePiece training entrypoints exist for discoverability and
  currently throw clear `ArgumentError` messages because algorithms are not yet
  implemented.

The pretrained-tokenizer APIs (`load_tokenizer`, `tokenize`, `encode`,
`encode_result`, `decode`) remain stable and independent from training codepaths.
