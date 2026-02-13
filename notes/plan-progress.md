# KeemenaSubwords Plan Progress

## Iteration 23: Offsets robustness and downstream-safe span utilities

### Summary
- Added downstream-safe span helper APIs without changing the existing offset contract.
- Added regression tests for empty vs sentinel vs non-empty span participation, non-overlap checks, and byte-level multibyte safety.
- Updated canonical contract docs with explicit guidance for downstream consumers (especially KeemenaPreprocessing) on safe span handling.

### Code changes

1. `src/normalization.jl`
- Added and documented:
  - `has_nonempty_span(offset::Tuple{Int,Int})::Bool`
  - `span_ncodeunits(offset::Tuple{Int,Int})::Int`
  - `span_codeunits(text::AbstractString, offset::Tuple{Int,Int})::Vector{UInt8}`
  - `is_valid_string_boundary(text::AbstractString, idx::Int)::Bool`
  - `try_span_substring(text::AbstractString, offset::Tuple{Int,Int})::Union{Nothing,String}`
  - `offsets_are_nonoverlapping(offsets::Vector{Tuple{Int,Int}}; ignore_sentinel::Bool=true, ignore_empty::Bool=true)::Bool`
- Behavior guarantees:
  - sentinel `(0,0)` and empty spans remain non-participating for `has_nonempty_span`.
  - `try_span_substring` is non-throwing and returns:
    - `""` for sentinel/empty spans,
    - `String` when boundaries are valid,
    - `nothing` when boundaries are not valid.
  - `span_codeunits` returns exact byte slices for non-empty spans and `UInt8[]` for sentinel/empty.

2. `src/KeemenaSubwords.jl`
- Exported new helpers:
  - `has_nonempty_span`
  - `span_ncodeunits`
  - `span_codeunits`
  - `is_valid_string_boundary`
  - `try_span_substring`
  - `offsets_are_nonoverlapping`

3. `test/runtests.jl`
- Added new testset:
  - `Section 23 offsets robustness and downstream-safe span utilities`
- Coverage added:
  - helper semantics for sentinel/empty/non-empty offsets
  - boundary validity checks (`is_valid_string_boundary`)
  - non-throwing behavior for `try_span_substring`
  - overlap regression checks using `offsets_are_nonoverlapping` across representative families:
    - WordPiece
    - SentencePiece Unigram
    - HF tokenizer.json WordPiece fixture
    - classic BPE
    - ByteBPE
    - tiktoken fixture
  - multibyte byte-level safety checks on:
    - `"cafe\u0301"`
    - `"é"`
    - `"🙂"`
    - `"a🙂b"`
  - for ByteBPE and HF ByteLevel fixture:
    - `span_codeunits` length equals `stop - start` for non-empty spans,
    - `try_span_substring` never throws,
    - when `try_span_substring` returns `String`, its codeunits match `span_codeunits`.

4. `notes/OffsetContract.md`
- Added explicit downstream-safe section:
  - offsets are codeunit spans and may not always be valid Julia string slicing boundaries.
  - recommended alignment participation rule:
    - participate iff `has_nonempty_span(offset)`.
  - documented safe inspection helpers:
    - `span_codeunits`
    - `try_span_substring`
    - `is_valid_string_boundary`
    - `offsets_are_nonoverlapping`

5. `docs/src/normalization_offsets_contract.md`
- Synced from canonical notes via sync tool.

6. `docs/src/api.md`
- Added new helper APIs to public API surface list.

### Validation
- `julia --project=. -e 'using Pkg; Pkg.test()'` -> pass
- `julia --project=docs docs/make.jl` -> pass
- `julia --project=. tools/sync_offset_contract.jl --check` -> pass
