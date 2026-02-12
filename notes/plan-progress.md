# KeemenaSubwords normalization/offsets contract progress

## Completed

1. Public tokenizer intrinsic normalization API
- Added `normalize(tokenizer::AbstractSubwordTokenizer, text)::String` (identity by default).
- Added HF specialization so `normalize(::HuggingFaceJSONTokenizer, text)` applies tokenizer.json normalizer pipeline.
- Added integration helpers:
  - `tokenization_view(tokenizer, clean_text)`
  - `requires_tokenizer_normalization(tokenizer)`
  - `offsets_coordinate_system()` -> `:utf8_codeunits`

2. Tokenization bypass switch in structured encode
- Extended `encode_result(...; assume_normalized::Bool=false, ...)`.
- Contract behavior implemented:
  - `assume_normalized=true` skips intrinsic normalization during encoding.
  - HF encode path now supports `assume_normalized=true` and bypasses normalizer.

3. Offsets + masks invariants hardening
- `TokenizationResult` docs now define:
  - UTF-8 codeunit offsets
  - half-open `[start, stop)` spans
  - special-token sentinel `(0, 0)` with `special_tokens_mask == 1`
- `encode_result` metadata now records:
  - `assume_normalized`
  - `offsets_coordinates`
  - `offsets_reference`
- Added offsets support in structured outputs for:
  - WordPiece
  - BPE
  - ByteBPE
  - Unigram TSV
  - SentencePiece wrapper
  - tiktoken
  - HF tokenizer.json (supported pretokenizer paths including whitespace, byte-level prefix-space, and metaspace contract fixtures)

4. Documentation
- Added dedicated docs page:
  - `docs/src/normalization_offsets_contract.md`
- Updated docs navigation and references:
  - `docs/make.jl`
  - `docs/src/index.md`
  - `docs/src/integration.md`
  - `docs/src/loading.md`
  - `docs/src/api.md`

5. Regression tests for contract
- Updated structured output expectations for HF offsets/masks in `test/runtests.jl`.
- Added new contract test section covering:
  - normalization bypass equivalence (`normalize` + `assume_normalized=true`)
  - offsets monotonicity/bounds on normalized text
  - TemplateProcessing special mask + sentinel offsets
  - family smoke coverage for `assume_normalized=true` + offsets/masks

## Validation executed

- `julia --project=. -e 'using Pkg; Pkg.test()'` -> pass
- `julia --project=docs docs/make.jl` -> pass
