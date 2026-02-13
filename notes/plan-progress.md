# KeemenaSubwords plan progress

## Iteration 24

### Boundary-valid offsets guarantees by tokenizer family
- Added a new contract section to `notes/OffsetContract.md` documenting boundary validity expectations by tokenizer family:
  - non-byte-level tokenizers are expected to produce spanful offsets on valid Julia string boundaries,
  - byte-level tokenizers may produce non-boundary spans on multibyte Unicode,
  - downstream should treat `try_span_substring(...) == nothing` as expected for those byte-level multibyte cases and use `span_codeunits(...)`.
- Synced canonical contract to docs with `tools/sync_offset_contract.jl` so `docs/src/normalization_offsets_contract.md` stays consistent.

### Tests added (Section 24)
- Added `Section 24 boundary-valid offsets guarantees by tokenizer family` in `test/runtests.jl` with two testsets:
  - `Section 24 boundary-valid offsets for string-level tokenizers`:
    - tokenizers: WordPiece fixture, SentencePiece unigram, Unigram TSV, HF WordPiece JSON, HF Unigram JSON (metaspace), classic BPE.
    - multibyte inputs: `"cafe\u0301"`, `"é"`, `"🙂"`, `"a🙂b"`.
    - assertions for each non-empty span:
      - start/stop are valid Julia boundaries via `is_valid_string_boundary`,
      - `try_span_substring` returns `String` (never `nothing`),
      - substring codeunits match `span_codeunits`.
  - `Section 24 boundary-valid offsets for byte-level tokenizers on ASCII`:
    - tokenizers: ByteBPE fixture and HF ByteLevel fixture.
    - ASCII inputs: `"hello"`, `"hello world"`, `" hello world"`, `"hello  world"`, `"hello\tworld"`, `"hello\nworld"`.
    - assertions for each non-empty span:
      - `try_span_substring` returns `String`,
      - substring codeunits match `span_codeunits`.

### Validation
- `julia --project=. -e 'using Pkg; Pkg.test()'` -> pass
- `julia --project=docs docs/make.jl` -> pass
- `julia --project=. tools/sync_offset_contract.jl --check` -> pass
