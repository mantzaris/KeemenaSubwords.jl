# Integration With KeemenaPreprocessing

`KeemenaSubwords` tokenizers are callable and work with `KeemenaPreprocessing`'s callable tokenizer contract.

```julia
using KeemenaPreprocessing
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)

cfg = PreprocessConfiguration(tokenizer_name = keemena_callable(tokenizer))
bundle = preprocess_corpus(["hello world", "hello keemena"]; config=cfg)

# KeemenaPreprocessing stores callable levels under Symbol(typeof(tokenizer))
lvl = level_key(tokenizer)
subword_corpus = get_corpus(bundle, lvl)
```

For the normalization/offsets alignment contract (`clean_text -> tokenization_text -> encode_result(...; assume_normalized=true)`), see [Normalization and Offsets Contract](normalization_offsets_contract.md).

Alignment rule:
Use `tokenization_text = tokenization_view(tokenizer, clean_text)`, then call
`encode_result(tokenizer, tokenization_text; assume_normalized=true, ...)`.
Word and subword offsets must both be interpreted in the same `tokenization_text`
coordinate system.
