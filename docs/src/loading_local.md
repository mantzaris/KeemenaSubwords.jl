# Loading Tokenizers From Local Paths

Use explicit format loaders when you know the file layout, or `load_tokenizer(path; format=:auto)` for detection.

## GPT-2 / RoBERTa style BPE (`vocab.json` + `merges.txt`)

```julia
tok = load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")
```

## OpenAI encoder variant (`encoder.json` + `vocab.bpe`)

```julia
tok = load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")
```

## Classic BPE and Byte-level BPE (`vocab.txt` + `merges.txt`)

```julia
classic = load_bpe("/path/to/model_dir")
byte_level = load_bytebpe("/path/to/model_dir")
```

## WordPiece (`vocab.txt`)

```julia
wp = load_wordpiece("/path/to/vocab.txt"; continuation_prefix="##")
```

## SentencePiece (`.model`, `.model.v3`, `sentencepiece.bpe.model`)

```julia
sp_auto = load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)
sp_uni = load_sentencepiece("/path/to/spm.model"; kind=:unigram)
sp_bpe = load_sentencepiece("/path/to/tokenizer.model.v3"; kind=:bpe)
```

## tiktoken (`*.tiktoken` or text `tokenizer.model`)

```julia
tt = load_tiktoken("/path/to/o200k_base.tiktoken")
llama3_style = load_tiktoken("/path/to/tokenizer.model")
```

## Hugging Face `tokenizer.json`

```julia
hf = load_hf_tokenizer_json("/path/to/tokenizer.json")
```

## Generic auto-detect + override

```julia
auto_tok = load_tokenizer("/path/to/model_dir")
forced = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
```

## Register a local model key

```julia
register_local_model!(:my_local_qwen, "/path/to/model_dir"; format=:auto, family=:qwen)
load_tokenizer(:my_local_qwen)

register_local_model!(
    :my_local_gpt2,
    (format=:bpe_gpt2, vocab_json="/path/to/vocab.json", merges_txt="/path/to/merges.txt");
    family=:local,
)
```
