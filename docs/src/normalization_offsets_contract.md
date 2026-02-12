# Normalization and Offsets Contract

This section defines the integration contract used with `KeemenaPreprocessing`.

## Two Normalization Layers

1. Pipeline normalization (owned by `KeemenaPreprocessing`):
   general cleaning and preprocessing that produces `clean_text`.
2. Tokenizer intrinsic normalization (owned by `KeemenaSubwords`):
   tokenizer-defined normalization that produces `tokenization_text`.

Use:

```julia
tokenization_text = normalize(tokenizer, clean_text)
```

For tokenizers without intrinsic normalization, `normalize(tokenizer, text)` returns `text` unchanged.
For `HuggingFaceJSONTokenizer`, `normalize` applies the tokenizer.json normalizer pipeline.

## Canonical Integration Flow

```julia
using KeemenaSubwords

clean_text = "Hello WORLD"
tokenization_text = tokenization_view(tokenizer, clean_text)

result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    return_offsets=true,
    return_masks=true,
    add_special_tokens=true,
)
```

`assume_normalized=true` means `encode_result` must not re-run tokenizer intrinsic normalization.

## Offset Semantics

- Coordinate unit: UTF-8 codeunit indices (`offsets_coordinate_system() == :utf8_codeunits`)
- Span convention: half-open `[start, stop)`
- Special tokens inserted by post-processing use sentinel offsets `(0, 0)`
- `special_tokens_mask` marks special tokens with `1`

When `assume_normalized=true`, offsets are relative to the exact `text` passed to `encode_result`.
This is the expected mode for KeemenaPreprocessing alignment (`clean_text -> tokenization_text -> encode_result(...; assume_normalized=true)`).
