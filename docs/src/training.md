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
- `train_sentencepiece(...)`
- `train_sentencepiece_result(...)`
- `train_hf_bert_wordpiece(...)`
- `train_hf_bert_wordpiece_result(...)`
- `train_hf_roberta_bytebpe(...)`
- `train_hf_roberta_bytebpe_result(...)`
- `train_hf_gpt2_bytebpe(...)`
- `train_hf_gpt2_bytebpe_result(...)`

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
train_sentencepiece_result
train_hf_bert_wordpiece
train_hf_bert_wordpiece_result
train_hf_roberta_bytebpe
train_hf_roberta_bytebpe_result
train_hf_gpt2_bytebpe
train_hf_gpt2_bytebpe_result
```

## HF BERT WordPiece Preset

```julia
using KeemenaSubwords

corpus = [
    "Hello, world!",
    "Café naïve façade",
    "你好 世界",
]

tok = train_hf_bert_wordpiece(
    corpus;
    vocab_size=128,
    min_frequency=1,
    lowercase=true,
    strip_accents=nothing,
    handle_chinese_chars=true,
    clean_text=true,
)

export_tokenizer(tok, "out_hf_bert"; format=:hf_tokenizer_json)
reloaded = load_hf_tokenizer_json("out_hf_bert/tokenizer.json")
```

## HF RoBERTa ByteBPE Preset

```julia
using KeemenaSubwords

corpus = [
    "hello world",
    "hello, world!",
    "café costs 5 euros",
]

tok = train_hf_roberta_bytebpe(
    corpus;
    vocab_size=384,
    min_frequency=1,
)

export_tokenizer(tok, "out_hf_roberta"; format=:hf_tokenizer_json)
reloaded = load_hf_tokenizer_json("out_hf_roberta/tokenizer.json")
```

RoBERTa preset defaults are chosen for HF-style ByteLevel behavior:
- `use_regex=true` applies GPT-2 ByteLevel regex splitting.
- `add_prefix_space=true` matches RoBERTa-style leading-space handling.
- `trim_offsets=true` trims span edges for whitespace while preserving the
  offsets contract:
  non-span specials use sentinel `(0,0)`, while trimmed real tokens may become
  empty but remain in-bounds spans like `(k,k)` (never sentinel).

## HF GPT-2 ByteBPE Preset

```julia
using KeemenaSubwords

corpus = [
    "Hello my friend, how is your day going?",
    "café 🙂",
]

tok = train_hf_gpt2_bytebpe(
    corpus;
    vocab_size=384,
    min_frequency=1,
)

export_tokenizer(tok, "out_hf_gpt2"; format=:hf_tokenizer_json)
reloaded = load_hf_tokenizer_json("out_hf_gpt2/tokenizer.json")
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
- SentencePiece training supports both `model_type=:unigram` and
  `model_type=:bpe`.
- Unigram training defaults to SentencePiece-style `whitespace_marker="▁"` so
  multi-word text can round-trip through `decode(encode(...))`.
- If `whitespace_marker=""`, runtime Unigram tokenization is still word-split,
  so decoding may collapse spaces in multi-word text (for example
  `"hello world"` -> `"helloworld"`).

The pretrained-tokenizer APIs (`load_tokenizer`, `tokenize`, `encode`,
`encode_result`, `decode`) remain stable and independent from training codepaths.
