# KeemenaSubwords Plan Progress

## 2026-02-07 - Iteration 01

### Direction chosen
Establish a stable Phase-0 baseline from the repository plan before adding more algorithms:
- scaffold core package files and API contracts,
- implement one fully working tokenizer path (WordPiece),
- add a tiny built-in model and tests to validate end-to-end behavior.

### Implemented in this iteration
- Created modular source layout and wired exports/includes:
  - `src/types.jl`
  - `src/vocab.jl`
  - `src/normalization.jl`
  - `src/models.jl`
  - `src/wordpiece.jl`
  - `src/io.jl`
  - updated `src/KeemenaSubwords.jl`
- Added foundational contracts:
  - `AbstractSubwordTokenizer <: Function`
  - `TokenizerMetadata`, `SubwordVocabulary`
  - `keemena_callable`, `level_key`
  - shared vocab helpers and normalization hook
- Added built-in model registry and first model entry:
  - `available_models`, `describe_model`, `model_path`
  - built-in tiny model: `:core_wordpiece_en`
  - model file: `models/core_wordpiece_en/vocab.txt`
- Implemented functional WordPiece path:
  - `load_wordpiece`
  - `tokenize`, callable overload, `encode`, `decode`
  - tokenizer-level vocab/metadata helpers
- Added loader dispatch and format detection:
  - `load_tokenizer(::Symbol)` and `load_tokenizer(::AbstractString; format=:auto)`
  - placeholder stubs for `save_tokenizer` / `export_tokenizer`
- Added initial artifacts placeholder file:
  - `Artifacts.toml`
- Updated package documentation and quick-start content:
  - `README.md`
  - `docs/src/index.md`
- Replaced placeholder tests with iteration-01 tests:
  - `test/runtests.jl`

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Result: all tests passing (`18/18`).

### Notes
- Core model storage is currently in-repo under `models/`; artifact-backed model delivery is reserved for a later iteration.
- This baseline is ready for follow-up iterations that add BPE/ByteBPE/Unigram and richer I/O support.

## 2026-02-07 - Iteration 02

### Objective
Complete all deliverables in plan sections `## 1)` through `## 3)`.

### Completed coverage for section 1 (scope goals)
- Implemented major tokenizer families with callable Julia-native types:
  - `BPETokenizer`
  - `ByteBPETokenizer`
  - `WordPieceTokenizer`
  - `UnigramTokenizer`
  - `SentencePieceTokenizer`
- KeemenaPreprocessing-compatible callable contract remains in place:
  - `AbstractSubwordTokenizer <: Function`
  - `keemena_callable`
  - `level_key`
- Preserved first-milestone non-goals (no full HF parity, no tiktoken parity, no dropout sampling).

### Completed coverage for section 2 (algorithms)
- Added classic BPE engine + merges loader (`vocab.txt` + `merges.txt`).
- Added byte-level BPE engine using GPT-2-style byte-to-unicode tables.
- Kept and validated WordPiece greedy longest-match path.
- Added Unigram tokenizer with deterministic Viterbi/DP segmentation.
- Added SentencePiece compatibility layer for `.model` files (lightweight format), supporting:
  - SentencePiece-Unigram wrapping Unigram engine,
  - SentencePiece-BPE wrapping BPE engine,
  - whitespace marker behavior (`▁`).

### Completed coverage for section 3 (built-in models + model loading strategy)
- `Artifacts.toml` present at repo root.
- Implemented in-package core model assets (no external download required):
  - `:core_bpe_en` -> `models/core_bpe_en/`
  - `:core_wordpiece_en` -> `models/core_wordpiece_en/vocab.txt`
  - `:core_sentencepiece_unigram_en` -> `models/core_sentencepiece_unigram_en/spm.model`
- Expanded model registry APIs:
  - `available_models()`
  - `describe_model(name)`
  - `model_path(name)`
- Implemented required loading patterns:
  - `load_tokenizer("/path/to/model_dir")`
  - `load_tokenizer("/path/to/spm.model")`
  - `load_tokenizer((vocab_path, merges_path))`
  - plus `load_tokenizer(spec::NamedTuple)` for explicit format/path specs.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `40/40` tests passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped as expected outside CI).

### Notes
- SentencePiece `.model` support in this iteration uses a lightweight text model format to keep the package dependency-light while providing the required compatibility wrapper behavior.
- Core models are shipped in-repo under `models/`; `Artifacts.toml` is in place for future content-addressed artifact migration.

## 2026-02-07 - Iteration 03

### Objective
Complete plan section `## 4)` (API surface overview), including:
- KeemenaPreprocessing integration-facing helpers,
- direct end-user API completeness for loading/tokenization/encoding/vocab metadata,
- save/export API behavior,
- future-facing training entrypoint surface.

### Completed coverage for section 4.1 (integration surface)
- Preserved callable tokenizer design (`AbstractSubwordTokenizer <: Function`) across all tokenizer families.
- Expanded integration helper behavior:
  - `keemena_callable(tokenizer::AbstractSubwordTokenizer)` returns tokenizer directly.
  - `keemena_callable(tokenizer::Function)` supported directly.
  - `keemena_callable(tokenizer)` now wraps non-Function tokenizers when a `tokenize` method exists.
- Added `level_key(tokenizer::Function)` in addition to tokenizer-type overload.
- Added explicit docs/examples showing `level_key(tokenizer)` usage with KeemenaPreprocessing bundles.

### Completed coverage for section 4.2 (direct user API)
- API surface present and tested for all concrete tokenizers:
  - `load_tokenizer(::Symbol)`
  - `load_tokenizer(::AbstractString; format=:auto)`
  - `load_tokenizer(::Tuple{vocab,merges}; format=...)`
  - `load_tokenizer(::NamedTuple)`
  - `tokenize`, callable overloads, `encode`, `decode`
  - `token_to_id`, `id_to_token`, `vocab_size`, `special_tokens`, `unk_id`, `model_info`
- Updated `model_info` to stable named-tuple output:
  - `(format, model_name, version, normalizer)`
- Implemented save/export behavior:
  - `save_tokenizer(tokenizer, outdir; format=:internal)`
  - `export_tokenizer(tokenizer, outdir; format=...)`
  - supported formats: `:internal`, `:bpe`, `:bpe_gpt2`, `:wordpiece_vocab`, `:unigram_tsv`, `:sentencepiece_model`
  - implemented deterministic writers for vocab/merges/unigram/sentencepiece model outputs.
- Added future-facing training entrypoint APIs as requested by section 4 plan surface:
  - `train_bpe(...)`
  - `train_unigram(...)`
  - `train_wordpiece(...)`
  - currently explicit stubs that throw `ArgumentError` (documented as not implemented yet).

### Files updated for section 4
- `src/types.jl`
- `src/io.jl`
- `src/training.jl` (new)
- `src/KeemenaSubwords.jl`
- `src/bpe.jl`
- `src/bytebpe.jl`
- `src/wordpiece.jl`
- `src/unigram.jl`
- `src/sentencepiece.jl`
- `README.md`
- `docs/src/index.md`
- `test/runtests.jl`

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `50/50` tests passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped as expected outside CI).

### Notes
- Training APIs are intentionally stubs in this iteration to complete the API contract from section 4 without claiming algorithm-training implementation readiness.
- Export support is format-focused and intentionally minimal/explicit to keep behavior deterministic for round-tripping in tests.

## 2026-02-07 - Iteration 04

### Objective
Complete plan section `## 5)` (package layout updates).

### Completed coverage for section 5
- `Artifacts.toml` remains present at repository root.
- Kept a single public module in `src/KeemenaSubwords.jl` with flat includes for algorithm isolation.
- Confirmed split source layout is in place:
  - `src/types.jl`
  - `src/vocab.jl`
  - `src/normalization.jl`
  - `src/models.jl`
  - `src/io.jl`
  - `src/bpe.jl`
  - `src/bytebpe.jl`
  - `src/wordpiece.jl`
  - `src/unigram.jl`
  - `src/sentencepiece.jl`
  - `src/training.jl`
- Strengthened model registry centralization with artifact-aware resolution in `src/models.jl`:
  - registry entries now carry `artifact_name`, `artifact_subpath`, and in-repo fallback paths,
  - `model_path` now resolves artifact paths when present, otherwise falls back to in-repo model assets.
- Added deterministic test fixtures (as recommended `test/golden`/`test/data` equivalent):
  - `test/fixtures/bpe/vocab.txt`
  - `test/fixtures/bpe/merges.txt`
  - `test/fixtures/wordpiece/vocab.txt`
  - `test/fixtures/unigram/unigram.tsv`
  - `test/fixtures/sentencepiece/toy_bpe.model`
- Updated tests to consume these fixtures directly for deterministic coverage.

### Additional repo setup adjustment
- Added `Pkg` stdlib to `Project.toml` dependencies to support runtime artifact-resolution utilities used in `src/models.jl`.
- Resolved and refreshed `Manifest.toml` accordingly.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `53/53` tests passed (`KeemenaSubwords sections 1-5`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped as expected outside CI).

### Notes
- Artifact wiring is now layout-ready while still using in-repo model fallbacks, keeping behavior stable without forcing external artifact fetches.
- Deterministic fixtures now cover BPE/WordPiece/Unigram/SentencePiece test inputs explicitly in-repo.

## 2026-02-07 - Iteration 05

### Objective
Complete plan section `## 6)` (proposed file/module tree + signature contracts alignment).

### Completed coverage for section 6
- Aligned module tree with section-6 training split:
  - added `src/bpe_train.jl`
  - added `src/unigram_train.jl`
  - wired both into module includes in `src/KeemenaSubwords.jl`.
- Updated `src/training.jl` to act as high-level API front-end and delegate to train backends:
  - `train_bpe(...)` now routes to `_train_bpe_impl(...)`
  - `train_unigram(...)` now routes to `_train_unigram_impl(...)`
  - defaults aligned with section-6 contract style (`<UNK>`, `<PAD>`).
- Preserved explicit, deterministic not-yet-implemented behavior for training backends via `ArgumentError` while keeping contract-ready signatures.
- Extended `src/io.jl` contract handling to match section-6 detection expectations:
  - auto-detect and load GPT-2 style `vocab.json + merges.txt` as `:bpe_gpt2`,
  - detect `tokenizer.json` and fail with clear unsupported-format error,
  - kept existing path/symbol/spec loaders stable.
- Added deterministic section-6 fixture assets:
  - `test/fixtures/bpe_gpt2/vocab.json`
  - `test/fixtures/bpe_gpt2/merges.txt`
  - `test/fixtures/internal/tokenizer.json`
- Expanded tests with explicit section-6 coverage:
  - GPT-2 format auto/explicit loading path,
  - unsupported internal `tokenizer.json` detection path.

### Files updated for section 6
- `src/KeemenaSubwords.jl`
- `src/training.jl`
- `src/io.jl`
- `src/bpe_train.jl` (new)
- `src/unigram_train.jl` (new)
- `test/runtests.jl`
- `test/fixtures/bpe_gpt2/vocab.json` (new)
- `test/fixtures/bpe_gpt2/merges.txt` (new)
- `test/fixtures/internal/tokenizer.json` (new)

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `57/57` tests passed (`KeemenaSubwords sections 1-6`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped as expected outside CI).

### Notes
- `bpe_train.jl` and `unigram_train.jl` are now structure-complete placeholders with stable contracts and clear pending-implementation errors, matching section-6 intent without over-claiming algorithm-training completion.

## 2026-02-07 - Iteration 06

### Objective
Complete plan section `## 7)` (implementation-order phase completion), focusing on remaining unfinished items.

### Completed section-7 items
- Implemented **Phase 3 step 11** (`BPE training`) with a deterministic baseline trainer:
  - corpus word counting,
  - iterative pair-frequency merges with `min_frequency` stopping,
  - vocabulary assembly with special-token handling,
  - returns `BPETokenizer`.
- Implemented **Phase 3 step 12** (`Unigram training`) with a deterministic baseline trainer:
  - candidate extraction (chars + frequent substrings + words),
  - bounded seed selection,
  - vocab construction with special-token handling,
  - log-prob assignment and return of `UnigramTokenizer`.
- Kept **Phase 3 step 13** (`save/export`) validated against trained model round-trips.
- Added **Phase 2 step 10 docs pages** explicitly requested by section 7:
  - `docs/src/integration.md`
  - `docs/src/loading.md`
  - `docs/src/models.md`
  - wired in `docs/make.jl` pages list.

### Structural/code updates
- `src/training.jl`
  - added corpus/special-token normalization helpers,
  - `train_bpe` and `train_unigram` now delegate to backend implementations.
- `src/bpe_train.jl`
  - replaced placeholder with baseline implementation.
- `src/unigram_train.jl`
  - replaced placeholder with baseline implementation.
- `src/bpe.jl`
  - improved unk-token auto-resolution (`<unk>`, `<UNK>`, `[UNK]`) for loader robustness.

### Test updates
- Replaced training-stub expectations with real training behavior checks in `test/runtests.jl`:
  - trains BPE/Unigram,
  - verifies tokenize/encode non-empty behavior,
  - validates save/load round-trips for trained tokenizers,
  - keeps `train_wordpiece` as explicit not-yet-implemented error.
- Test suite label updated to `KeemenaSubwords sections 1-7`.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `63/63` tests passed (`KeemenaSubwords sections 1-7`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped as expected outside CI).

### Notes
- Phase 4 items in section 7 include performance/advanced features and optional extensions; this iteration focused on closing concrete missing deliverables from phases 2-3 in the recommended order.

## 2026-02-07 - Iteration 07

### Objective
Close the remaining section-7 contract gap by implementing real Unigram EM + pruning behavior and aligning docs/tests with actual training status.

### Completed section-7 refinements
- Upgraded `train_unigram` backend (`src/unigram_train.jl`) from heuristic scoring to a deterministic Unigram LM training flow with:
  - seed vocabulary generation from character coverage + frequent substrings,
  - EM expectation/maximization loop via forward-backward over segmentations,
  - explicit pruning to target non-special vocabulary size,
  - post-pruning refinement pass for final probabilities.
- Added a strict training guard:
  - throws informative `ArgumentError` when `vocab_size` is too small to preserve required specials plus character coverage.
- Updated docs home page training status in `docs/src/index.md`:
  - now reflects `train_bpe` and `train_unigram` as implemented,
  - keeps `train_wordpiece` as intentionally unimplemented.
- Expanded section-7 tests in `test/runtests.jl`:
  - added vocabulary-size assertions for trained BPE/Unigram models,
  - added explicit `train_unigram` pruning-path coverage (`seed_size` larger than target vocab).

### Files updated in iteration 07
- `src/unigram_train.jl`
- `test/runtests.jl`
- `docs/src/index.md`
- `notes/plan-progress.md`

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: all tests passed (`KeemenaSubwords sections 1-7`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped outside CI as expected).

## 2026-02-08 - Iteration 08

### Objective
Implement plan section `## 9)` by adding downloadable public pretrained model keys (tiktoken + GPT-2 + BERT WordPiece + T5 SentencePiece), wiring them into the existing built-in registry/loader path, and validating end-to-end behavior.

### Completed section-9 implementation
- Extended the existing registry in `src/models.jl` (no parallel registry) with new keys:
  - `:tiktoken_o200k_base`
  - `:tiktoken_cl100k_base`
  - `:tiktoken_r50k_base`
  - `:tiktoken_p50k_base`
  - `:openai_gpt2_bpe`
  - `:bert_base_uncased_wordpiece`
  - `:t5_small_sentencepiece_unigram`
- Added `prefetch_models(keys=available_models(); force=false)` and integrated symbol-loading prefetch path:
  - `load_tokenizer(name::Symbol)` now optionally prefetches and then resolves through `describe_model`.
- Enhanced `describe_model` output with:
  - resolved primary path,
  - resolved required file list (`files`),
  - upstream provenance metadata (`upstream_files`, `provenance_urls`).
- Added full tiktoken support (`src/tiktoken.jl`):
  - loader for `.tiktoken`,
  - encode/tokenize/decode APIs,
  - sparse-rank handling (real upstream files are not strictly contiguous in rank IDs).
- Extended IO dispatch (`src/io.jl`) to support:
  - `:tiktoken` format and auto-detection,
  - GPT-2 upstream filename pair (`encoder.json` + `vocab.bpe`) in addition to `vocab.json` + `merges.txt`.
- Upgraded SentencePiece loader (`src/sentencepiece.jl`) with protobuf-model parsing fallback:
  - supports real `spiece.model` binary protobuf Unigram models.

### Artifact/download tooling
- Added `tools/build_public_model_artifact.jl` to:
  1) download upstream assets,
  2) verify SHA-256 for known files,
  3) compute SHA-256 for Hugging Face files and write metadata TOML,
  4) create and bind artifact in `Artifacts.toml`,
  5) emit tarball + checksum + release-upload instructions.
- Ran the script successfully:
  - `git-tree-sha1`: `0293c813181ade29e0c88e0e48015f201d89ddeb`
  - tarball: `artifacts-build/keemena_public_tokenizer_assets_v1-0293c813181ade29e0c88e0e48015f201d89ddeb.tar.gz`
  - tarball sha256: `37a377b41e1e24a663de3ebf0bc9d0ca8e12a6b22d6b07fee5cecc5b99c365c4`

### Tests/docs updates
- Expanded section-9 smoke tests in `test/runtests.jl` for all new built-in keys.
- Updated README and docs pages with:
  - new built-in model keys,
  - `prefetch_models(...)`,
  - artifact build helper workflow.
- Added/updated fallback fixture paths under `models/` for local deterministic behavior when artifact assets are unavailable.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `98/98` tests passed (`KeemenaSubwords sections 1-9`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped outside CI as expected).

## 2026-02-09 - Iteration 09

### Objective
Complete stage-9 model inventory work by adding curated, pinned, artifact-backed built-in tokenizers with explicit provenance/license metadata, and align README/docs with the live registry.

### Completed work
- Extended the existing built-in registry path (`available_models`, `describe_model`, `model_path`, `load_tokenizer(::Symbol)`) with curated keys:
  - `:mistral_v1_sentencepiece`
  - `:mistral_v3_sentencepiece`
  - `:phi2_bpe`
  - `:roberta_base_bpe`
  - `:xlm_roberta_base_sentencepiece_bpe`
- Added immutable upstream refs and license strings to registry metadata (exposed via `describe_model`):
  - pinned Hugging Face commit URLs,
  - `license`,
  - `upstream_ref`,
  - `upstream_files` and resolved `files`.
- Kept package architecture simple:
  - no `tokenizer.json` interpreter added,
  - reused existing format loaders (`:sentencepiece_model`, `:bpe_gpt2`, `:tiktoken`).
- Updated SentencePiece loader to accept directory-based model resolution for:
  - `spm.model`,
  - `tokenizer.model`,
  - `tokenizer.model.v3`,
  - `sentencepiece.bpe.model`.
- Converted non-core model paths to artifact-only fallbacks:
  - removed non-core in-repo files under `models/`,
  - retained only tiny `:core_*` in-repo model assets.

### Artifact and tooling updates
- Added `tools/build_curated_model_artifacts.jl`:
  - downloads pinned tokenizer files from reputable upstream URLs,
  - verifies file SHA-256,
  - builds one artifact per curated key,
  - binds artifact hashes in `Artifacts.toml`,
  - emits release tarball + checksum metadata.
- Updated `Artifacts.toml` with:
  - `git-tree-sha1` for all curated artifacts,
  - tarball `download` stanzas + `sha256` checksums (release URL pattern prepared).

### README/docs/test updates
- README now reflects the full current registry key set and clarifies:
  - artifact-backed vs in-repo core models,
  - `prefetch_models(...)` offline behavior,
  - current training status (`train_bpe`/`train_unigram` baseline implemented; `train_wordpiece` pending).
- Updated docs pages:
  - `docs/src/index.md`
  - `docs/src/models.md`
  - `docs/src/loading.md`
  with curated key examples.
- Expanded tests with curated key smoke checks and prefetch existence assertion:
  - `prefetch_models([:mistral_v1_sentencepiece])` + `model_path` directory existence,
  - load/tokenize/encode/decode-callability checks across curated keys.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `128/128` tests passed (`KeemenaSubwords sections 1-9`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped outside CI as expected).

## 2026-02-09 - Iteration 10

### Objective
Complete plan section `## 10)` by tightening LLM-oriented model coverage UX and ensuring README/docs consistency with the implemented registry and artifact strategy.

### Completed section-10 work
- Added/kept high-impact artifact-backed built-ins in the existing registry path, including `:qwen2_5_bpe`.
- Kept package scope simple:
  - no full `tokenizer.json` interpreter was introduced,
  - loaders remain based on supported file families (`.tiktoken`, GPT2/RoBERTa `vocab.json+merges.txt` / `encoder.json+vocab.bpe`, WordPiece `vocab.txt`, SentencePiece model files).
- Added model inventory UX helpers in `src/models.jl`:
  - `available_models(; format=nothing, family=nothing)`,
  - `recommended_defaults_for_llms()`,
  - `register_external_model!(...)` for user-supplied assets (e.g., gated Llama tokenizers).
- Improved model provenance/source reporting:
  - `_resolve_model_source` now reports `:external` for user-registered existing local assets (instead of `:missing`).
- Explicitly handled Mistral Tekken as **supported external path usage** (not shipped as a built-in) pending clearly redistributable pinned assets.

### README/docs alignment updates
- Updated docs examples to match README and live API/registry behavior:
  - `docs/src/index.md`
  - `docs/src/models.md`
  - `docs/src/loading.md`
- Added/updated documentation coverage for:
  - `:qwen2_5_bpe`,
  - `available_models` filtering by `format`/`family`,
  - `recommended_defaults_for_llms()` prefetch workflow,
  - external-user-supplied Llama 2/Llama 3 and Tekken-style loading examples,
  - `register_external_model!` usage.

### Test coverage additions retained
- Registry/filter/defaults tests include:
  - `available_models(format=:bpe_gpt2)` contains `:qwen2_5_bpe`,
  - `available_models(family=:qwen) == [:qwen2_5_bpe]`,
  - `recommended_defaults_for_llms()` includes `:qwen2_5_bpe`.
- Added external model registration smoke test:
  - `register_external_model!` + `load_tokenizer(:external_test_bpe)` round-trip callability.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `141/141` tests passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful (local deploy skipped outside CI as expected).

## 2026-02-09 - Iteration 11

### Objective
Complete plan section `## 11)` by closing the remaining no-Python tokenization gaps: pure Julia `tokenizer.json` loading, LLaMA convenience APIs without redistribution, and expanded built-in registry coverage.

### Completed section-11 implementation
- Added pure Julia Hugging Face tokenizer JSON loader stack under `src/huggingface_json/`:
  - `hf_json_types.jl`
  - `hf_json_parse.jl`
  - `hf_json_pipeline.jl`
  - `hf_json_loader.jl`
- Added new public tokenizer type + loader:
  - `HuggingFaceJSONTokenizer`
  - `load_hf_tokenizer_json(path)`
- Integrated into existing loader path (no parallel registry):
  - `load_tokenizer(path; format=:hf_tokenizer_json)`
  - directory autodetection now prefers `tokenizer.json` when present.
- Implemented supported matrix with clear unsupported-component errors (including JSON path + workaround):
  - model types: BPE, WordPiece, Unigram
  - normalizers: Lowercase, NFKC, Sequence
  - pre-tokenizers: ByteLevel, Whitespace/WhitespaceSplit, Metaspace, Split(regex), Sequence
  - post-processors: TemplateProcessing, Sequence
  - decoders: ByteLevel, WordPiece, Metaspace, Sequence
- Preserved compatibility fallback for built-in HF entries:
  - if a model is marked `:hf_tokenizer_json` but `tokenizer.json` is absent and GPT2 files exist, loading falls back to `vocab.json + merges.txt`.

### Model registry / UX updates
- Extended model registry metadata and APIs:
  - `available_models(; format=nothing, family=nothing, shipped=nothing)`
  - `describe_model(key)` now includes `expected_files` and `shipped`
  - preserved existing `available_models`, `describe_model`, `model_path`, `prefetch_models`, `load_tokenizer(::Symbol)` flow.
- Added local user-model registry persistence in cache:
  - `register_local_model!(...)`
  - persists to `local_models.toml` under cache root (`KEEMENA_SUBWORDS_CACHE_DIR` override supported).
- Added opt-in HF download helper for user-managed/gated tokenizers:
  - `download_hf_files(repo_id, filenames; revision, outdir, token, force)`
- Added built-in key:
  - `:bert_base_multilingual_cased_wordpiece` (artifact-backed intent, optional availability in tests).

### Qwen section-11 alignment
- Kept `:qwen2_5_bpe` in the main built-in registry.
- Updated registry format to `:hf_tokenizer_json` with GPT2 fallback behavior.
- Extended upstream metadata entries to include:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  in addition to `vocab.json` + `merges.txt`.

### Test/docs updates
- Added deterministic HF JSON fixtures and tests:
  - `test/fixtures/hf_json_wordpiece/tokenizer.json`
  - `test/fixtures/hf_json_unsupported/tokenizer.json`
- Added section-11 test coverage:
  - `tokenizer.json` load + tokenize/encode/decode smoke behavior
  - TemplateProcessing special token insertion checks
  - unsupported component error message check
  - local registry flow via `register_local_model!` + symbol loading
  - HF download helper smoke (`filenames=[]`) with optional network-gated test branch
- Updated README and docs pages with:
  - `tokenizer.json` loading examples
  - `register_local_model!` and `download_hf_files(...)`
  - `available_models(...; shipped=...)`
  - multilingual WordPiece built-in mention.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.resolve()'`
  - Result: dependency graph updated for JSON parser support.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `157/157` tests passed (`KeemenaSubwords sections 1-11`).
