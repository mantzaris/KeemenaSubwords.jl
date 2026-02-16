# Concepts

This page is a first-hour guide to the concepts you need for reliable tokenization and alignment workflows in KeemenaSubwords.

## Where This Fits

Typical Julia LLM preprocessing split:

- `KeemenaPreprocessing`: produces normalized `clean_text`.
- `KeemenaSubwords`: turns text into token pieces and 1-based token ids.

Recommended integration flow:

1. `clean_text = ...` from your preprocessing pipeline.
2. `tokenization_text = tokenization_view(tokenizer, clean_text)`.
3. `encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, return_masks=true, ...)`.

## Token Pieces Vs Token Ids

- `tokenize(tok, text)` returns token pieces (`Vector{String}`).
- `encode(tok, text; ...)` returns token ids (`Vector{Int}`).
- `decode(tok, ids)` maps ids back to text.

```@example concepts_core_bpe
using KeemenaSubwords

tok = load_tokenizer(:core_bpe_en)
text = "Hello world"

pieces = tokenize(tok, text)
ids = encode(tok, text; add_special_tokens=true)
decoded = decode(tok, ids)

(; pieces, ids, decoded)
```

KeemenaSubwords uses **1-based token ids**.

Convert to 0-based ids only when you need parity with external tooling:

```julia
ids_zero_based = ids .- 1
ids_julia = ids_zero_based .+ 1
```

## Tokenizer Families Supported

| Family | Typical format symbols | Typically byte-level | Offset implication |
| --- | --- | --- | --- |
| BPE (classic) | `:bpe` | No | Spanful offsets are expected to be string-safe in normal usage. |
| ByteBPE | `:bytebpe` | Yes | Offsets are valid codeunit spans, but may not always be safe string slice boundaries on multibyte text. |
| WordPiece | `:wordpiece`, `:wordpiece_vocab` | No | Spanful offsets are expected to be string-safe in normal usage. |
| Unigram TSV | `:unigram`, `:unigram_tsv` | No | Spanful offsets are expected to be string-safe in normal usage. |
| SentencePiece model | `:sentencepiece_model` | Usually no | Spanful offsets are expected to be string-safe for standard SentencePiece pipelines. |
| tiktoken | `:tiktoken` | Yes | Same byte-level caveat as ByteBPE. |
| HF tokenizer.json | `:hf_tokenizer_json` | Depends on pipeline | If ByteLevel components are present, use byte-level caveats for offsets. |

## Special Tokens And `add_special_tokens`

Inspect special token mappings and common ids:

```@example concepts_core_wordpiece_specials
using KeemenaSubwords

tok = load_tokenizer(:core_wordpiece_en)

specials = special_tokens(tok)
ids = (
    bos = try bos_id(tok) catch; nothing end,
    eos = try eos_id(tok) catch; nothing end,
    pad = try pad_id(tok) catch; nothing end,
    unk = try unk_id(tok) catch; nothing end,
)

(; specials, ids)
```

`add_special_tokens=true` asks the tokenizer/post-processor to insert framework specials (for example BOS/EOS or CLS/SEP).

Offset behavior:

- Inserted specials: `special_tokens_mask[i] == 1` and `offsets[i] == (0, 0)`.
- Specials that appear in the input text as matched added tokens: `special_tokens_mask[i] == 1`, but offsets can still be real spans into the input text.

## Structured Encoding And Offsets

Use `encode_result` when you need ids plus offsets and masks in one object (`TokenizationResult`).

```@example concepts_core_sentencepiece_structured
using KeemenaSubwords

tok = load_tokenizer(:core_sentencepiece_unigram_en)
clean_text = "Hello world"
tokenization_text = tokenization_view(tok, clean_text)

result = encode_result(
    tok,
    tokenization_text;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

(
    ids = result.ids,
    tokens = result.tokens,
    offsets = result.offsets,
    attention_mask = result.attention_mask,
    special_tokens_mask = result.special_tokens_mask,
    metadata = result.metadata,
)
```

High-level offset contract:

- Coordinate system: UTF-8 codeunits.
- Index base: 1.
- Span style: half-open `[start, stop)`.
- Sentinel for no-span tokens: `(0, 0)`.

For the full contract and helper APIs, see [Normalization and Offsets Contract](normalization_offsets_contract.md).

Recommended KeemenaPreprocessing alignment pattern:

```julia
tokenization_text = tokenization_view(tok, clean_text)
result = encode_result(
    tok,
    tokenization_text;
    assume_normalized=true,
    return_offsets=true,
    return_masks=true,
    add_special_tokens=true,
)
```

## Model Registry And Caching

Use the registry APIs to discover models and the cache APIs to avoid reloading tokenizers repeatedly:

```julia
available_models(shipped=true)
describe_model(:core_bpe_en)
prefetch_models([:core_bpe_en, :core_wordpiece_en, :core_sentencepiece_unigram_en])

tok = get_tokenizer_cached(:core_bpe_en)

# Clear long-lived cached tokenizer instances when needed
# (for example to release memory or force a fresh reload).
clear_tokenizer_cache!()
```

## Loading And Exporting

Pointers:

- [Loading Tokenizers](loading.md)
- [Loading Tokenizers From Local Paths](loading_local.md)
- [Tokenizer Formats and Required Files](formats.md)

Export APIs:

- `export_tokenizer(tokenizer, out_dir; format=...)`
- `save_tokenizer(tokenizer, out_dir; format=...)`

If you export with `format=:hf_tokenizer_json`, KeemenaSubwords writes `tokenizer.json` for HF-compatible fast tokenizer loading. Current scope details (for example companion config files) are documented in [Tokenizer Formats and Required Files](formats.md).

Placeholder examples below require local paths or gated access and are intentionally non-executable in docs:

```julia
# local path placeholder (non-executable)
tok = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)

# gated install placeholder (non-executable)
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
```
