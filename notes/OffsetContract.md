# Normalization and Offsets Contract

This document is the canonical contract for normalization and offsets behavior in
KeemenaSubwords when integrated with KeemenaPreprocessing.

## Normalization Ownership

1. Pipeline normalization (KeemenaPreprocessing):
   produces `clean_text`.
2. Tokenizer intrinsic normalization (KeemenaSubwords):
   produces `tokenization_text`.

Use:

```julia
tokenization_text = tokenization_view(tokenizer, clean_text)
```

Then call:

```julia
result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    return_offsets=true,
    return_masks=true,
    add_special_tokens=true,
)
```

When `assume_normalized=true`, KeemenaSubwords must not re-run tokenizer
intrinsic normalization.

## Offset Convention

- coordinate unit: UTF-8 codeunits
- index base: 1-based
- span style: half-open `[start, stop)`
- valid bounds for spanful offsets:
  - `1 <= start <= stop <= ncodeunits(text) + 1`

Programmatic helpers:

- `offsets_coordinate_system() == :utf8_codeunits`
- `offsets_index_base() == 1`
- `offsets_span_style() == :half_open`
- `offsets_sentinel() == (0, 0)`
- `has_span(offset)` is true iff `offset != (0, 0)`
- `has_nonempty_span(offset)` is true iff the offset is spanful and `stop > start`
- `span_ncodeunits(offset)` returns span length in codeunits (`0` for sentinel/empty)

## Sentinel and Special Tokens

Sentinel:

- `(0, 0)` means "no source-text span".
- In a 1-based scheme, `(0, 0)` is out-of-range and unambiguous.

Special token semantics:

- Inserted special tokens (TemplateProcessing/post-processor inserted):
  - `special_tokens_mask[i] == 1`
  - `offsets[i] == (0, 0)`
- Special tokens matched from user text as added tokens:
  - `special_tokens_mask[i] == 1`
  - `offsets[i]` is a real span into the input text (`offsets[i] != (0, 0)`)

Important:

- `special_tokens_mask` marks special-token identity.
- Span participation is determined by offsets/sentinel, not by mask alone.

## Alignment Rule for KeemenaPreprocessing

Canonical coordinate system is `tokenization_text`.

KeemenaPreprocessing should:

1. Produce `clean_text` via pipeline normalization.
2. Produce `tokenization_text = tokenization_view(tokenizer, clean_text)`.
3. Call `encode_result(tokenizer, tokenization_text; assume_normalized=true, ...)`.
4. Compute both word offsets and subword offsets on `tokenization_text`.
5. Ignore sentinel offsets `(0, 0)` during span alignment.
6. Do not drop all `mask==1` tokens blindly; present-in-text special tokens may
   have real spans.

Recommended span participation policy:

- Participate in span alignment iff `has_nonempty_span(offset)`.

## Downstream-Safe Span Inspection

Offsets are codeunit spans. Do not assume they are always valid Julia string
slicing boundaries, especially for byte-level tokenizers on multibyte text.

Use these helpers for robust downstream handling:

- `span_codeunits(text, offset)`:
  - returns `UInt8[]` for sentinel/empty spans,
  - returns the exact byte slice for non-empty spans.
- `try_span_substring(text, offset)`:
  - returns `""` for sentinel/empty spans,
  - returns `String` only when both boundaries are valid Julia string boundaries,
  - returns `nothing` otherwise.
- `is_valid_string_boundary(text, idx)` can be used to inspect boundary validity.
- `offsets_are_nonoverlapping(offsets; ignore_sentinel=true, ignore_empty=true)`
  validates a downstream non-overlap invariant.
