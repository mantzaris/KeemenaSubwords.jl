# Loading Tokenizers From Local Paths

Use explicit loader functions when you know the file contract. Use `load_tokenizer(path; format=:auto)` only when auto-detection is preferred.

Named-spec convention:
- use `path` as the canonical key for single-file formats,
- keep format-specific pair keys for multi-file formats (`vocab_json` + `merges_txt`, `encoder_json` + `vocab_bpe`).
- backward-compatible aliases (`vocab_txt`, `model_file`, `encoding_file`, `tokenizer_json`) are still accepted.

## 1) GPT-2 / RoBERTa style BPE (`vocab.json` + `merges.txt`)

```julia
tok = load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")

# equivalent named spec
load_tokenizer((format=:bpe_gpt2, vocab_json="/path/to/vocab.json", merges_txt="/path/to/merges.txt"))
```

## 2) OpenAI encoder variant (`encoder.json` + `vocab.bpe`)

```julia
tok = load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")

# equivalent named spec
load_tokenizer((format=:bpe_encoder, encoder_json="/path/to/encoder.json", vocab_bpe="/path/to/vocab.bpe"))
```

## 3) Classic BPE / Byte-level BPE (`vocab.txt` + `merges.txt`)

```julia
classic = load_bpe("/path/to/model_dir")
byte_level = load_bytebpe("/path/to/model_dir")
```

## 4) WordPiece (`vocab.txt`)

```julia
wp = load_wordpiece("/path/to/vocab.txt"; continuation_prefix="##")

# register via canonical key
register_local_model!(
    :my_wordpiece,
    (format=:wordpiece_vocab, path="/path/to/vocab.txt");
    description="local WordPiece",
)
```

## 5) SentencePiece (`.model`, `.model.v3`, `sentencepiece.bpe.model`)

`load_sentencepiece` accepts either:
- standard SentencePiece binary model files,
- or Keemena text-exported SentencePiece files (same filename patterns).

```julia
sp_auto = load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)
sp_uni = load_sentencepiece("/path/to/spm.model"; kind=:unigram)
sp_bpe = load_sentencepiece("/path/to/tokenizer.model.v3"; kind=:bpe)

register_local_model!(:my_sp, (format=:sentencepiece_model, path="/path/to/tokenizer.model"))
```

## 6) tiktoken (`*.tiktoken` or text `tokenizer.model`)

```julia
tt = load_tiktoken("/path/to/o200k_base.tiktoken")
llama3_style = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)

register_local_model!(:my_tiktoken, (format=:tiktoken, path="/path/to/tokenizer.model"))
```

## 7) Hugging Face `tokenizer.json`

```julia
hf = load_hf_tokenizer_json("/path/to/tokenizer.json")

register_local_model!(:my_hf, (format=:hf_tokenizer_json, path="/path/to/tokenizer.json"))
```

## 8) Generic auto-detect + override

```julia
auto_tok = load_tokenizer("/path/to/model_dir")
forced = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```
