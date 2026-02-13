# KeemenaSubwords plan progress

## Iteration 25

### Minor offsets polish: family coverage + strict validators

### Section 24 coverage expansion
- Expanded `Section 24 boundary-valid offsets for string-level tokenizers` in `test/runtests.jl` to include SentencePiece BPE compatibility:
  - `load_sentencepiece(fixture("sentencepiece", "toy_bpe.model"))`
- Expanded `Section 24 boundary-valid offsets for byte-level tokenizers on ASCII` to include `tiktoken` fixture coverage:
  - `load_tiktoken(fixture("tiktoken_model", "tokenizer.model"))`
  - used tokenizer-supported ASCII-safe strings for deterministic offline checks.

### Strict validation helpers
- Added strict offset contract validators in `src/normalization.jl`:
  - `validate_offsets_contract(text, offsets; require_string_boundaries=false)::Bool`
  - `assert_offsets_contract(text, offsets; require_string_boundaries=false)::Nothing`
- Validators check:
  - sentinel semantics (`(0,0)` is the only zero-index sentinel),
  - bounds and monotonic span shape (`stop >= start`, `stop <= ncodeunits(text)+1`),
  - optional string-boundary requirements for non-empty spans when `require_string_boundaries=true`.
- Exported new helpers in `src/KeemenaSubwords.jl`.
- Added API listing entry in `docs/src/api.md`.

### Validator tests
- Added `Section 25 strict offset validators` testset in `test/runtests.jl`:
  - valid sentinel/empty span vectors -> `validate==true`, `assert` non-throwing,
  - known-good tokenizer offsets (WordPiece fixture) -> strict validation passes,
  - intentionally invalid offsets -> `validate==false`, `assert` throws `ArgumentError`,
  - explicit `require_string_boundaries` behavior on a non-boundary multibyte span.

### Docs
- Updated canonical contract doc `notes/OffsetContract.md` with a short maintainer/debugging section for strict validators.
- Synced docs copy via existing sync workflow (`tools/sync_offset_contract.jl`).

### Validation
- `julia --project=. -e 'using Pkg; Pkg.test()'` -> pass
- `julia --project=docs docs/make.jl` -> pass
- `julia --project=. tools/sync_offset_contract.jl --check` -> pass
