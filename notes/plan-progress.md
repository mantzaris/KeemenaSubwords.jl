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


## 12) Eliminate README/docs drift, finish WordPiece artifact coverage, and add LLaMA "one command install" convenience (without redistributing gated files)

### 12.1 Objective
Make the package feel finished and trustworthy by:
- Ensuring README and Documenter docs are generated from the same source of truth (no drift).
- Ensuring WordPiece coverage is fully real (artifact is actually downloadable and prefetchable), not just present in docs.
- Providing LLaMA-family convenience that feels like built-ins (automatic download + key-based loading) while respecting gated licensing and not redistributing files.

### 12.2 What "done" looks like
- The README built-in key list exactly matches `available_models()` output and includes WordPiece multilingual key.
- Docs examples use the correct function name(s) for local registry (`register_local_model!` if that is the canonical API).
- WordPiece built-ins:
  - `:bert_base_uncased_wordpiece`
  - `:bert_base_multilingual_cased_wordpiece`
  are both artifact-backed and `prefetch_models([...])` succeeds.
- LLaMA convenience:
  - Users can run one command like `install_model!(:llama3_8b_instruct; token=ENV["HF_TOKEN"])`
  - After install: `load_tokenizer(:llama3_8b_instruct)` works without specifying file paths.
  - The package does not ship LLaMA files in Artifacts.toml and does not redistribute them.

### 12.3 Tasks

#### A) Single source of truth for model inventory (stop drift)
1) Create one canonical registry table in code (if not already):
- Keep all model keys and metadata in one place (example file: `src/model_registry.jl`).
- Include fields used everywhere:
  - `key`, `format`, `family`, `license`, `description`
  - `expected_files`
  - `distribution` (one of: `:shipped`, `:artifact_public`, `:installable_gated`, `:user_local`)
  - `upstream_repo`, `upstream_ref`, `upstream_files`

2) Auto-generate README models section
- Add markers to README.md:
  - `<!-- KEEMENA_MODELS_START -->`
  - `<!-- KEEMENA_MODELS_END -->`
- Add `tools/sync_readme_models.jl` that:
  - loads KeemenaSubwords
  - queries registry and renders a markdown table grouped by format/family
  - replaces the marked section
- Add a CI check (or pre-commit script) that fails if the README is out of date.

3) Auto-generate docs "Built-In Models" content
- Either:
  - render the same table in Documenter by calling a function that returns markdown, or
  - include a generated markdown file produced by a doc build step.
Goal: docs and README are always consistent.

#### B) Fix naming drift in external/local registry API
1) Decide canonical API name:
- If `register_local_model!` is the intended public API, keep it and:
  - add `register_external_model!` as a deprecated alias (with a deprecation warning)
  - update docs examples to use `register_local_model!`
- If `register_external_model!` is the intended name, then:
  - update notes/docs to stop mentioning `register_local_model!`
  - ensure it persists to cache if that is implemented

2) Update docs and README examples to use the canonical name.

#### C) Ensure WordPiece artifact coverage is actually complete
1) Make `:bert_base_multilingual_cased_wordpiece` a first-class built-in everywhere:
- Add it to the README built-in list (via the auto-generation above).
- Ensure `describe_model(:bert_base_multilingual_cased_wordpiece)` works and prints correct provenance.

2) Ensure artifact binding exists and is stable
- Confirm `Artifacts.toml` includes the multilingual vocab artifact with:
  - pinned upstream revision (avoid floating "main" if possible)
  - sha256 verification
- Ensure `prefetch_models([:bert_base_multilingual_cased_wordpiece])` succeeds.

3) Optional: add one more WordPiece for completeness (only if you want)
- `:bert_base_cased_wordpiece` (English cased) as a small, cheap addition.
- Do not add more WordPiece beyond this; it becomes redundant noise.

4) Tests
- Add a test that prefetches multilingual WordPiece when network is allowed (ENV gated),
  and always tests that the registry entry exists and expected_files are correct.

#### D) LLaMA convenience without redistributing gated assets
Important constraint:
- Do not include Meta LLaMA tokenizer files as public package artifacts, and do not host them yourself.
- Provide an install workflow that uses the user's credentials or signed URL, stores into cache, and registers locally.

1) Add "installable gated models" registry entries
- Add entries with `distribution=:installable_gated`, for example:
  - `:llama2_tokenizer` (SentencePiece)
  - `:llama3_8b_tokenizer` (prefer tokenizer.json if available, otherwise documented alternatives)
- These are not "shipped" and not in Artifacts.toml, but they appear in:
  - `available_models(distribution=:installable_gated)` or similar.

2) Implement installer API
- `install_model!(key::Symbol; token=nothing, revision="main", force=false)`
  - Looks up registry entry
  - Calls `download_hf_files(...)` (already exists) for the listed filenames
  - Stores them under `KEEMENA_SUBWORDS_CACHE_DIR/install/<key>/...`
  - Calls `register_local_model!(key, installed_dir; format=..., family=:llama, description=...)`
- Optional: add `install_llama2_tokenizer!` and `install_llama3_tokenizer!` wrappers for discoverability.

3) Loading behavior after install
- `load_tokenizer(:llama3_8b_tokenizer)` should:
  - first check user local registry key (installed)
  - then fall back to regular built-ins
  - never silently attempt gated downloads without explicit `install_model!`

4) Docs
- Add a clear docs section:
  - "LLaMA tokenizers are gated; you must accept Meta license and have access"
  - "Use install_model! with HF token or use manual path loading"

5) Tests
- Add a unit test for install flow without network:
  - when token is missing, ensure a clear error message that explains how to proceed
- Add optional network test (ENV gated):
  - attempts to download only tokenizer files for a gated model when token is provided

### 12.4 Suggested implementation order
1) Auto-generate README and docs models tables from registry (kills drift fast).
2) Fix registry helper naming drift (register_local_model! vs register_external_model!).
3) Make multilingual WordPiece artifact fully solid + tests.
4) Add LLaMA install_model! workflow + docs.
5) Optional: add :bert_base_cased_wordpiece.


## 2026-02-09 - Iteration 12

### Objective
Complete plan section `## 12)` by eliminating README/docs model drift, hardening WordPiece artifact coverage, and adding one-command gated LLaMA tokenizer install flow without redistributing gated assets.

### Completed section-12 implementation
- Extended model registry metadata in `src/models.jl` so each entry now carries:
  - `format`, `family`, `distribution`, `license`, `description`
  - `upstream_repo`, `upstream_ref`
  - existing artifact/fallback path fields
- Added distribution model taxonomy and filtering:
  - `:shipped`, `:artifact_public`, `:installable_gated`, `:user_local`
  - `available_models(; format, family, distribution, shipped)`
- Added installable gated model entries (no artifact redistribution):
  - `:llama2_tokenizer` (`:sentencepiece_model`)
  - `:llama3_8b_tokenizer` (`:hf_tokenizer_json`)
- Implemented one-command install API for gated models:
  - `install_model!(key; token, revision="main", force=false)`
  - convenience wrappers:
    - `install_llama2_tokenizer!`
    - `install_llama3_8b_tokenizer!`
- Added guarded UX for missing gated assets:
  - `load_tokenizer(:llama*_...)` and `model_path(:llama*_...)` now return actionable install guidance when files are not present.
- Fixed local/external helper naming drift:
  - canonical API remains `register_local_model!`
  - `register_external_model!` now behaves as a deprecated compatibility alias (deprecation warning).
- Strengthened local registry persistence schema:
  - persisted metadata now includes `distribution` and `upstream_repo` in addition to format/path/description/family/license/upstream_ref.

### README/docs drift elimination
- Added generated inventory markers to both:
  - `README.md`
  - `docs/src/models.md`
- Added generator/check script:
  - `tools/sync_readme_models.jl`
  - renders grouped markdown table (format/family) from registry data
  - updates both README and docs models page from the same generated content
  - supports `--check` mode for CI drift detection
- Added CI enforcement step in `.github/workflows/CI.yml`:
  - `julia --project=. tools/sync_readme_models.jl --check`
- Updated docs/loading/index examples for section-12 workflow:
  - canonical local registration API (`register_local_model!`)
  - explicit gated install usage with `install_model!`
  - distribution-aware model discovery examples.

### WordPiece artifact coverage status
- Confirmed multilingual WordPiece remains first-class and artifact-backed:
  - key: `:bert_base_multilingual_cased_wordpiece`
  - artifact binding present in `Artifacts.toml` (lazy, sha-verified)
  - provenance visible in `describe_model(...)`
- Tests now always validate registry/metadata expectations for multilingual WordPiece and keep network-sensitive prefetch assertions optional.

### Tests/docs updates
- Updated `test/runtests.jl` to section label `sections 1-12`.
- Added/extended section-12 tests for:
  - distribution filters and gated key visibility
  - gated install flow missing-token error messaging
  - multilingual WordPiece metadata coverage
- Kept deterministic behavior for CI by gating network-sensitive checks via environment variables.

### Verification
- Ran: `julia --project=. tools/sync_readme_models.jl`
  - Result: README/docs model inventory sections regenerated from registry.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: sync check passes.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `169/169` tests passed (`KeemenaSubwords sections 1-12`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful.

### Notes
- LLaMA tokenizer files are not shipped in `Artifacts.toml` and are not redistributed by this repository.
- Installation is explicit/opt-in via `install_model!` and requires user-provided access credentials.

## 2026-02-09 - Iteration 13

### Objective
Complete plan section `## 13)` by consolidating docs, slimming README, and formalizing path-based loader contracts with robust format detection and clearer validation errors across tokenizer families.

### Completed section-13 implementation

#### Documentation consolidation
- Slimmed `README.md` to a quickstart-focused entry point:
  - concise overview,
  - install,
  - minimal usage examples,
  - featured model list,
  - links to docs pages.
- Added new docs pages:
  - `docs/src/formats.md`
  - `docs/src/loading_local.md`
  - `docs/src/llm_cookbook.md`
  - `docs/src/gated_models.md`
  - `docs/src/troubleshooting.md`
- Updated docs nav in `docs/make.jl` to include the new pages.
- Kept the full model inventory table in docs only (`docs/src/models.md`) and preserved generation from registry metadata via `tools/sync_readme_models.jl` (docs marker section only).

#### Explicit loader contracts and convenience APIs
- Added/exported explicit public loaders:
  - `load_bpe_gpt2(vocab_json, merges_txt; byte_level=true, ...)`
  - `load_bpe_encoder(encoder_json, vocab_bpe; byte_level=true, ...)`
  - plus explicit exports for `load_wordpiece`, `load_sentencepiece`, `load_tiktoken`, and existing `load_hf_tokenizer_json`.
- Added/exported detection helpers:
  - `detect_tokenizer_files(dir)`
  - `detect_tokenizer_format(path)`
- Updated `load_tokenizer(path; format=...)` dispatch to support:
  - `format=:bpe_encoder`
  - `format=:sentencepiece_model`
  - explicit path-pair routing for GPT-2/BPE encoder variants.

#### Robust auto-detection and `.model` sniffing
- Implemented directory/file detection logic in `src/io.jl` with explicit precedence:
  1) `tokenizer.json`
  2) `vocab.json + merges.txt`
  3) `encoder.json + vocab.bpe`
  4) SentencePiece model filenames
  5) `.tiktoken`
  6) fallback classic BPE / WordPiece / Unigram indicators
- Added `.model` sniffing behavior:
  - tiktoken-like text payload -> `:tiktoken`
  - SentencePiece-like/binary payload -> `:sentencepiece_model`
- Preserved override behavior (`format=` always wins).

#### Loader validation and error clarity
- Improved validation/error messaging with expected-file guidance and example calls in:
  - GPT-2/BPE encoder loaders,
  - WordPiece path resolution,
  - SentencePiece path/type loading,
  - tiktoken path/file parsing,
  - HF tokenizer.json path resolution.

#### Local model registry ergonomics
- Extended `register_local_model!` to support:
  - path registration with `format=:auto` default,
  - explicit `NamedTuple` specs (path, vocab/merges, encoder/vocab.bpe, tokenizer.json, model_file, encoding_file).
- Added spec materialization for explicit file-set registration into cache-backed local spec dirs.
- Persisted richer local metadata in cache TOML:
  - format, distribution, upstream repo/ref,
  - resolved files,
  - optional notes.
- Added richer `describe_model(...)` fields for local entries:
  - `resolved_files`
  - `notes`.

### Tests added/updated for section 13
- Expanded test suite label to `KeemenaSubwords sections 1-13`.
- Added fixtures:
  - `test/fixtures/bpe_encoder/{encoder.json,vocab.bpe}`
  - `test/fixtures/tiktoken_model/tokenizer.model`
  - `test/fixtures/sentencepiece/binary_stub.model`
- Added section-13 tests for:
  - `detect_tokenizer_format` and `detect_tokenizer_files` behavior,
  - `.model` tiktoken vs sentencepiece sniffing,
  - explicit loaders (`load_bpe_gpt2`, `load_bpe_encoder`, `load_tiktoken`),
  - error message quality for missing required files,
  - `register_local_model!` with auto path and explicit NamedTuple spec,
  - format override behavior.

### Verification
- Ran: `julia --project=. tools/sync_readme_models.jl`
  - Result: docs inventory section regenerated.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: docs inventory sync check passed.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `194/194` tests passed (`KeemenaSubwords sections 1-13`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful.

### Notes
- Full model inventory remains generated from registry metadata on docs side.
- README is intentionally concise and links to docs for operational detail.
- Existing public APIs remain available; explicit convenience loaders were added without removing previous entry points.

## 2026-02-10 - Iteration 14

### Objective
Complete plan section `## 14)` by slimming README, polishing docs consistency, improving API discoverability, and adding CI guardrails to prevent docs/README drift and bad format examples.

### Completed section-14 implementation

#### README slimming and docs-first flow
- Rewrote `README.md` as quickstart-only content:
  - install,
  - 3 minimal examples (built-in load, local format override, gated install),
  - short featured models section,
  - links to canonical docs pages.
- Added featured-model marker block:
  - `<!-- KEEMENA_FEATURED_MODELS_START -->`
  - `<!-- KEEMENA_FEATURED_MODELS_END -->`

#### Generated inventory split (README summary + docs full table)
- Extended `tools/sync_readme_models.jl` to render and sync two outputs from the same registry source:
  - README featured list block,
  - full docs model table block in `docs/src/models.md`.
- `--check` mode now verifies both generated blocks and fails when stale.

#### Docs consistency and naming canonicalization
- Added API docs page:
  - `docs/src/api.md`
  - explicit loader list via `@docs`,
  - registry/install/discovery API list,
  - full exported API via `@autodocs`.
- Added API nav entry in `docs/make.jl` and linked index content in `docs/src/index.md`.
- Canonicalized local-loading and format-reference docs:
  - `docs/src/formats.md`
  - `docs/src/loading_local.md`
  - `docs/src/loading.md`
  - `docs/src/gated_models.md`
- Ensured examples consistently use:
  - GPT2/RoBERTa BPE -> `vocab.json + merges.txt` with `vocab_json`, `merges_txt`,
  - encoder variant -> `encoder.json + vocab.bpe` with `encoder_json`, `vocab_bpe`,
  - WordPiece -> `vocab.txt` with `vocab_txt`,
  - SentencePiece -> `model_file`,
  - tiktoken -> `encoding_file`,
  - HF JSON -> `tokenizer_json`.
- Kept `register_local_model!` as canonical in examples; `register_external_model!` only appears as a deprecation note.

#### API discoverability polish
- Added/updated docstrings with example calls for explicit loader APIs:
  - `load_bpe_gpt2`, `load_bpe_encoder`,
  - `load_wordpiece`, `load_sentencepiece`,
  - `load_tiktoken`, `load_hf_tokenizer_json`,
  - `detect_tokenizer_format`, `detect_tokenizer_files`.
- Added docstrings for gated convenience wrappers:
  - `install_llama2_tokenizer!`
  - `install_llama3_8b_tokenizer!`

#### Guardrails and CI checks
- Added docs example consistency checker:
  - `tools/check_docs_examples.jl`
  - validates common wrong format/file pairings (for example GPT2+BPE with `vocab.txt`),
  - enforces canonical API usage in docs examples (`register_local_model!`).
- Updated CI workflow (`.github/workflows/CI.yml`) to run:
  - `julia --project=. tools/sync_readme_models.jl --check`
  - `julia --project=. tools/check_docs_examples.jl`
  - package tests.
- Enabled docs doctest execution in `docs/make.jl` (`doctest=true`).

### Verification
- Ran: `julia --project=. tools/sync_readme_models.jl`
  - Result: generated README featured block and docs model inventory updated.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: README/docs generated blocks in sync.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: docs examples passed consistency checks.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `194/194` tests passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build successful with doctest pass; deploy skipped locally as expected.

### Notes
- README now stays intentionally brief; full inventory/details remain docs-authoritative and generated from registry metadata.
- Section-14 guardrails are now automated in CI to reduce future docs/API drift.

## 2026-02-10 - Iteration 15

### Objective
Complete plan section `## 15)` with final docs/API consistency polish and deeper Hugging Face `tokenizer.json` compatibility so common modern HF pipelines work in pure Julia.

### Completed section-15 implementation

#### Docs marker leakage and generation flow
- Removed visible marker-line strategy from generated sections.
- Updated `tools/sync_readme_models.jl` to sync by stable anchor blocks instead:
  - README block between `## Featured Models` and `## Documentation`.
  - docs models block between the generated-table intro line and provenance line.
- Regenerated:
  - `README.md` featured models section.
  - `docs/src/models.md` full registry table.
- Result: no `KEEMENA_*` marker artifacts in rendered docs output.

#### Canonical examples and API discoverability
- Kept explicit loader APIs exported and prominently documented in `docs/src/api.md`:
  - `load_bpe_gpt2`
  - `load_bpe_encoder`
  - `load_wordpiece`
  - `load_sentencepiece`
  - `load_tiktoken`
  - `load_hf_tokenizer_json`
  - plus `load_tokenizer`, `detect_tokenizer_format`, `detect_tokenizer_files`.
- Standardized local spec/doc examples around canonical named keys:
  - `vocab_json`, `merges_txt`
  - `encoder_json`, `vocab_bpe`
  - `vocab_txt`
  - `model_file`
  - `encoding_file`
  - `tokenizer_json`
- Clarified SentencePiece support consistently as:
  - standard binary `.model` / `.model.v3`,
  - or Keemena text-exported SentencePiece `.model` payloads.

#### Docs guardrails and consistency linting
- Expanded `tools/check_docs_examples.jl` checks:
  - rejects GPT2+BPE mismatches (`format=:bpe_gpt2` with `vocab.txt`),
  - rejects encoder-variant mismatches,
  - rejects WordPiece usage/specs paired with BPE files (`vocab.json`, `merges.txt`, `encoder.json`, `vocab.bpe`).
- Preserved deprecation hygiene:
  - `register_external_model!` appears only as compatibility note (primary docs use `register_local_model!`).

#### Hugging Face tokenizer.json compliance expansion
- Expanded schema/model/pipeline support across:
  - top-level parsing (`model`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, `added_tokens`, plus `truncation`/`padding` metadata parse),
  - model specs:
    - BPE (`continuing_subword_prefix`, `fuse_unk`, `byte_fallback`, `dropout` parse+validation),
    - WordPiece (`max_input_chars_per_word`),
    - Unigram (`byte_fallback`),
  - added token runtime handling (`special`, `single_word`, `lstrip`, `rstrip`, `normalized`),
  - normalizers:
    - `StripAccents`, `Replace`, `Prepend`, `NFD`, plus `Sequence`,
  - pre-tokenizers:
    - `Digits`, `Punctuation`, plus existing `ByteLevel`, `Metaspace`, `Whitespace*`, `Split`, `Sequence`,
  - post-processors:
    - `BertProcessing`, `RobertaProcessing`, `TemplateProcessing`, `Sequence`,
  - decoders:
    - `BPE`, `WordPiece`, `ByteLevel`, `Metaspace`, `Sequence`.
- Added/updated HF JSON fixtures and section-15 tests for:
  - byte-level BPE patterns,
  - metaspace unigram patterns,
  - added token behaviors and template processing,
  - unsupported component errors with JSON path + workaround guidance.

#### WordPiece behavior hardening
- Extended `WordPieceTokenizer` with `max_input_chars_per_word`.
- Updated loading and tokenization logic to honor max-length behavior for WordPiece segmentation.

#### Longer-term roadmap documentation (no training scope)
- Added roadmap section to `docs/src/llm_cookbook.md` covering:
  - incremental HF component coverage,
  - richer optional encode outputs,
  - curated additional flagship tokenizers,
  - performance hardening priorities.
- Updated docs home map link text in `docs/src/index.md`.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `207/207` tests passed (`KeemenaSubwords sections 1-15`).
- Ran: `julia --project=. tools/sync_readme_models.jl`
  - Result: generated README/docs model sections updated.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: README/docs generated sections in sync.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: docs example consistency checks passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed; deploy skipped locally as expected.

### Notes
- Marker leakage on docs models page is resolved by removing marker lines from generated page content.
- Full inventory remains generated from registry metadata; README stays concise and docs-first.


## 2026-02-10 - Iteration 16

### Objective
Complete plan section `## 16)` by stabilizing CI/version policy, aligning local-loading docs contracts with the real API, and locking the behavior with regression tests.

### Completed section-16 implementation

#### CI and version baseline cleanup
- Rewrote `.github/workflows/CI.yml` in plain ASCII and normalized formatting.
- Aligned CI matrix with package compat:
  - kept Julia `1.10` and `pre`,
  - removed Julia `1.6` (package already requires Julia `1.10` in `Project.toml`).
- Added/kept sync and docs-contract checks in CI test job:
  - `tools/sync_readme_models.jl --check`
  - `tools/check_docs_examples.jl`
- Documented the minimum version in `README.md` (`Julia 1.10+`).

#### Named-spec contract unification (+ aliases)
- Made `path` the canonical single-file named-spec key in docs and examples.
- Kept backward-compatible aliases in `load_tokenizer(spec::NamedTuple)`:
  - WordPiece: `path` + `vocab_txt`,
  - HF tokenizer.json: `path` + `tokenizer_json`,
  - Unigram: `path` + `unigram_tsv`,
  - existing aliases retained for SentencePiece/tiktoken (`model_file`, `encoding_file`).
- Updated `src/io.jl` docstring examples and error contract text accordingly.

#### Public loader coherence decision
- Chosen direction: keep classic loaders public.
- Exported in `src/KeemenaSubwords.jl`:
  - `load_bpe`
  - `load_bytebpe`
  - `load_unigram`
- Added them to API docs in `docs/src/api.md`.

#### Docs drift fixes and byte-level UX clarification
- Updated `docs/src/formats.md`:
  - canonical named-spec keys now reflect `path` for single-file formats and alias notes,
  - added a new **Byte-level behavior** subsection clarifying:
    - which formats are byte-level vs non-byte-level,
    - practical encode/decode expectations,
    - HF `tokenizer.json` ByteLevel interaction.
- Updated `docs/src/loading_local.md` examples to use canonical `path`-based specs with alias note.

#### Regression guardrails
- Expanded `tools/check_docs_examples.jl` with section-16 contract checks:
  - validates that `docs/src/formats.md` named-spec keys are supported by `load_tokenizer(spec)` parsing,
  - fails when docs code blocks reference non-exported Keemena APIs (unless module-qualified),
  - keeps prior wrong-pairing checks (for example GPT2 BPE with `vocab.txt`).
- Added section-16 tests in `test/runtests.jl`:
  - named-spec alias equivalence tests (`path` vs alias keys),
  - format-detection precedence test (`tokenizer.json` preferred over co-present `vocab.json` + `merges.txt`),
  - override test (`format=:bpe_gpt2` still routes correctly),
  - smoke tests for public classic loaders (`load_bpe`, `load_bytebpe`, `load_unigram`).
- Updated test suite label to `KeemenaSubwords sections 1-16`.

### Verification
- Ran: `LC_ALL=C grep -nP "[^\\x00-\\x7F]" .github/workflows/CI.yml`
  - Result: no non-ASCII characters found.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: README/docs generated sections in sync.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: docs contract checks passed.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `222/222` tests passed (`KeemenaSubwords sections 1-16`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed; deploy skipped locally as expected.

### Notes
- This iteration explicitly formalizes Julia `1.10+` as the supported baseline in both compat and CI.
- Docs and API contracts now use a stable canonical spec pattern while remaining backward-compatible for existing user code.

## 2026-02-10 - Iteration 17

### Objective
Implement section `## 17)` hardening work for conformance, structured outputs, and robust bring-your-own-files workflows while keeping runtime Python-free and preserving existing APIs.

### Completed section-17 implementation

#### Conformance golden framework
- Added committed golden conformance assets under `test/golden/`:
  - `test/golden/core_bpe_en.json`
  - `test/golden/core_wordpiece_en.json`
  - `test/golden/core_sentencepiece_unigram_en.json`
  - `test/golden/fixture_bpe_gpt2.json`
  - `test/golden/fixture_hf_json_wordpiece.json`
  - `test/golden/fixture_tiktoken_model.json`
  - shared string corpus file `test/golden/strings.txt`
- Added Julia golden runner `test/test_goldens.jl`:
  - loads tokenizer via public APIs (symbol key or local path + format),
  - compares `encode` ids exactly,
  - optionally checks tokens and decode outputs,
  - emits first-mismatch diagnostics with local windows.
- Integrated runner into main test suite (`test/runtests.jl`) as `Section 17 golden conformance`.

#### Maintainer-only golden generator (optional Python tooling)
- Added optional tooling under `tools/generate_goldens/`:
  - `tools/generate_goldens/requirements.txt`
  - `tools/generate_goldens/generate_goldens.py`
  - `tools/generate_goldens/README.md`
- This tooling is not used by runtime or CI tests; committed JSON goldens remain the source of truth for test execution.

#### Structured encode output API (non-breaking)
- Added public `TokenizationResult` struct in `src/types.jl`.
- Added public APIs in `src/io.jl`:
  - `encode_result(tokenizer, text; add_special_tokens=true, return_offsets=false, return_masks=false)`
  - `encode_batch_result(tokenizer, texts; kwargs...)`
- Kept existing `encode(tokenizer, text; ...) -> Vector{Int}` unchanged.
- Added best-effort normalized-coordinate offsets for:
  - `WordPieceTokenizer`
  - `BPETokenizer`
  - `UnigramTokenizer` (when whitespace marker is empty)
- For ambiguous pipelines (for example HF JSON/byte-level), offsets return `nothing` by design.
- Added masks in structured output when requested:
  - `attention_mask`
  - `token_type_ids`
  - `special_tokens_mask`

#### BYO files hardening and internal file-spec standardization
- Added internal/public `FilesSpec` struct (`src/types.jl`) to represent single-file and multi-file tokenizer layouts.
- Added `load_tokenizer(spec::FilesSpec)` (`src/io.jl`) by converting to canonical NamedTuple contract.
- Added `register_local_model!(key, spec::FilesSpec; ...)` (`src/models.jl`) delegating to existing persisted local registry flow.
- Preserved backward compatibility with existing NamedTuple specs and aliases.

#### Public API and docs alignment
- Exported new API symbols in `src/KeemenaSubwords.jl`:
  - `TokenizationResult`
  - `FilesSpec`
  - `encode_result`
  - `encode_batch_result`
- Updated API docs discoverability (`docs/src/api.md`) and loading examples:
  - structured output example in `docs/src/loading.md`
  - `FilesSpec` usage example in `docs/src/loading_local.md`

#### Additional section-17 regression tests
- Added `Section 17 structured outputs and file specs` in `test/runtests.jl` covering:
  - `encode_result` ids/tokens/offsets/masks behavior,
  - `encode_batch_result` behavior,
  - `FilesSpec` load/register flows,
  - additional detection precedence checks for mixed-format directories.
- Updated top-level test label to `KeemenaSubwords sections 1-17`.

### Verification
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: docs contract checks passed.
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `332/332` tests passed (`KeemenaSubwords sections 1-17`).
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed (deploy skipped locally as expected).

### Notes
- HF tokenizer.json core compliance expansions from section 15 are now backed by additional conformance goldens and section-17 regression coverage.
- Python remains optional and maintainer-only for golden regeneration; runtime/test execution remains Julia-only.


## 2026-02-10 - Iteration 18

### Objective
Implement section `## 18)` to stabilize CI around artifact-backed models, fix registry/test contract mismatches, and make artifact/download diagnostics robust.

### Completed section-18 implementation

#### A) GPT-2 registry/test mismatch resolved
- Updated the model registry test invariant in `test/runtests.jl`:
  - replaced brittle `length(gpt2_info.files) == 2` with format-contract assertions on `expected_files` for GPT-2 (`vocab.json + merges.txt` and `encoder.json + vocab.bpe`).
- This keeps registry correctness checks independent from artifact installation state.

#### B) Artifact robustness + diagnostics
- Added artifact verification tooling:
  - `tools/verify_artifacts.jl`
  - verifies artifact binding/hash, runs `ensure_artifact_installed`, prints attempted URLs and precise errors.
- Hardened artifact installation behavior in `src/models.jl`:
  - artifact installation attempts are memoized per artifact in-process to reduce repeated retries.
  - failure logs now include artifact name, candidate URLs, and exception message.

#### C) Public-model cache fallback (artifact-first, cache fallback)
- Refactored artifact-public model fallback paths in `src/models.jl`:
  - artifact-backed entries now resolve fallback path under user cache (`KEEMENA_SUBWORDS_CACHE_DIR` / depot cache), not package-local placeholders under `models/_artifacts_only`.
- Added public fallback installer path in `prefetch_models`:
  - artifacts are still preferred.
  - if artifact install is unavailable, public upstream files are downloaded to cache with optional SHA-256 verification from registry metadata.
- This preserves the built-in key UX while removing dependence on package-tree placeholder directories.

#### D) Deterministic tests + reduced download spam
- Updated section-9 tests in `test/runtests.jl`:
  - split registry metadata checks from download integration checks.
  - network/download integration runs only when `KEEMENA_TEST_DOWNLOADS=1` (legacy `KEEMENA_RUN_NETWORK_TESTS` still honored).
  - avoids repeated prefetch attempts in default CI.
  - avoids unhandled exceptions by keeping model load assertions in explicit guarded test paths.
- Updated test suite label to `KeemenaSubwords sections 1-18`.

#### E) CI defaults and dependency coherence
- Added `KEEMENA_TEST_DOWNLOADS: "0"` in `.github/workflows/CI.yml` test job for deterministic default CI behavior.
- Added stdlib dependency declaration for `SHA` in `Project.toml` because the cache verifier now computes SHA-256 for downloaded files.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `323/323` tests passed (`KeemenaSubwords sections 1-18`) with download integration disabled by default.
- Ran: `KEEMENA_TEST_DOWNLOADS=1 julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `334/334` tests passed (`KeemenaSubwords sections 1-18`) with download integration enabled.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed (local deploy skipped as expected).
- Ran: `julia --project=. tools/verify_artifacts.jl`
  - Result: all configured tokenizer artifacts verified successfully in current environment.

### Notes
- Section 18 now enforces a stable offline/CI baseline while still supporting explicit download integration testing.
- Artifact diagnostics are now actionable if a future hosting/checksum issue appears in CI.

## 2026-02-10 - Iteration 19

### Objective
Implement section `## 19)` to harden built-in asset UX, fix SentencePiece fallback discovery gaps, and add CI download verification that exercises real model fetch/load behavior.

### Completed section-19 implementation

#### A) Quiet artifact failures when fallback succeeds + richer prefetch status
- Added `prefetch_models_status(keys; force=false)` in `src/models.jl`:
  - returns `Dict{Symbol,NamedTuple}` with:
    - `available::Bool`
    - `method::Symbol` (`:artifact`, `:fallback_download`, `:already_present`, `:failed`)
    - `path::Union{Nothing,String}`
    - `error::Union{Nothing,String}`
- Kept `prefetch_models` backward-compatible:
  - now delegates to `prefetch_models_status` and returns `Dict{Symbol,Bool}`.
- Changed prefetch flow:
  - artifact install is attempted first,
  - if artifact fails, fallback file downloads are attempted,
  - warning is emitted only if neither artifact nor fallback yields usable files.
- Added maintainer debug mode:
  - `KEEMENA_SUBWORDS_ASSET_DEBUG=1`
  - logs artifact failure details, URLs, download attempts, and fallback outcomes.

#### B) SentencePiece fallback discovery fix (`spiece.model` + single `*.model` heuristic)
- Extended SentencePiece candidate discovery used by model resolution:
  - added `spiece.model` to `src/models.jl::_sentencepiece_candidates`.
- Added safe directory heuristic in both model discovery paths:
  - if no canonical filename is found,
  - and no `tokenizer.json` is present,
  - and directory contains exactly one `*.model`/`*.model.v3`, treat as SentencePiece.
- Applied this consistently in:
  - `src/models.jl::_sentencepiece_candidates`
  - `src/io.jl::detect_tokenizer_files`
  - `src/sentencepiece.jl::_resolve_sentencepiece_model_path`

#### C) Retry/backoff download helper wired into fallback and HF download paths
- Added `_download_with_retries(url, dest; retries=4, initial_backoff=0.5)` in `src/models.jl`:
  - exponential backoff,
  - URL-aware error message with last exception,
  - optional debug logs when `KEEMENA_SUBWORDS_ASSET_DEBUG=1`.
- Integrated helper into:
  - fallback public asset downloads (`_ensure_public_model_cached!`)
  - `download_hf_files(...)` for installable/local workflows.

#### D) Tests for section 19 contracts and smoke behavior
- Updated top-level testset label to `KeemenaSubwords sections 1-19`.
- Refined section-9 tests to be registry-shape checks only (offline deterministic).
- Added new `Section 19 asset UX and sentencepiece fallback` testset in `test/runtests.jl`:
  - validates `prefetch_models_status` shape and compatibility with `prefetch_models`,
  - regression test for `spiece.model` detection/loading,
  - regression test for directory with exactly one `custom.model`,
  - download smoke (gated by `KEEMENA_TEST_DOWNLOADS=1`) for minimal representative keys:
    - `:tiktoken_o200k_base`
    - `:openai_gpt2_bpe`
    - `:bert_base_uncased_wordpiece`
    - `:t5_small_sentencepiece_unigram`
    - `:qwen2_5_bpe`

#### E) CI download-smoke workflow
- Updated `.github/workflows/CI.yml`:
  - added scheduled trigger (nightly) for proactive upstream/download drift checks,
  - added dedicated `download-smoke` job (non-PR events) with `KEEMENA_TEST_DOWNLOADS=1`,
  - added cache for `~/.julia/keemena_subwords` alongside existing Julia cache,
  - keeps default PR test job deterministic and fast.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `339/339` tests passed (`KeemenaSubwords sections 1-19`) in default/offline mode.
- Ran: `KEEMENA_TEST_DOWNLOADS=1 julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `366/366` tests passed (`KeemenaSubwords sections 1-19`) with download smoke enabled.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: README/docs inventory blocks still in sync.

### Notes
- User-facing logs are now quiet for successful fallback recovery paths.
- Maintainer diagnostics remain available via debug env var and CI download-smoke coverage.

## 2026-02-10 - Iteration 20

### Objective
Implement section `## 20)` to harden end-user asset UX and concurrency behavior, add high-level E2E workflow tests, introduce one-call convenience APIs with caching, and document deferred goals in `futurework.md`.

### Completed section-20 implementation

#### A) Informative artifact-vs-fallback decisions (quiet by default)
- Added new status API in `src/models.jl`:
  - `prefetch_models_status(keys; force=false)` returns per-key:
    - `available::Bool`
    - `method::Symbol` (`:artifact`, `:fallback_download`, `:already_present`, `:failed`)
    - `path::Union{Nothing,String}`
    - `error::Union{Nothing,String}`
- Kept `prefetch_models` backward-compatible:
  - still returns `Dict{Symbol,Bool}` via the new status API.
- Added convenience query helpers:
  - `asset_status(key; force=false)`
  - `print_asset_status(key; force=false)`
- Added one-time per-session fallback info messaging:
  - if artifact fails and fallback succeeds, emit concise info once per key:
    - "Model <key>: artifact unavailable, used fallback download, cached at <path>"
- Added debug-mode diagnostics (`KEEMENA_SUBWORDS_ASSET_DEBUG=1`):
  - artifact URLs
  - artifact error
  - captured artifact installer output
  - fallback download details
- Captured artifact installer stdout/stderr in `_ensure_artifact_installed`:
  - suppressed when fallback succeeds
  - included in failure summaries when both artifact and fallback fail.

#### B) Concurrency-safe fallback/cache downloads
- Added per-model lock helper in `src/models.jl`:
  - `_with_model_lock(key) do ... end`
  - lock file under `<cache>/locks/<key>.lock`
  - timeout + polling backoff
  - `finally` cleanup release
- Added atomic download commit helper:
  - `_download_to_file_atomic(url, dest; expected_sha256=...)`
  - unique temp filename (`.download.<pid>.<thread>.<time>.<rand>`)
  - checksum validation on temp file
  - atomic rename into destination
  - stale temp cleanup
- Added retry helper with exponential backoff:
  - `_download_with_retries(url, dest; retries=4, initial_backoff=0.5)`
- Wired hardening into:
  - fallback public model cache install (`_ensure_public_model_cached!`)
  - `download_hf_files(...)` (wrapped with lock + atomic writes)

#### C) E2E user workflow tests
- Added new file `test/e2e_user_workflows.jl` and included it from `test/runtests.jl`.
- Added readable high-level workflows:
  - HF tokenizer.json directory autodetect end-to-end
  - register local model then load by symbolic key
  - WordPiece vocab.txt path loading end-to-end
  - SentencePiece discovery via `spiece.model` and single `custom.model` heuristic
  - gated network E2E workflow set when `KEEMENA_TEST_DOWNLOADS=1` for representative public keys
- Updated top-level test label to `KeemenaSubwords sections 1-20`.

#### D) One-call convenience APIs + session tokenizer cache
- Added one-call overloads in `src/io.jl` for key/path usage without explicit `load_tokenizer`:
  - `tokenize(key_or_path, text; format=nothing, prefetch=true)`
  - `encode(key_or_path, text; format=nothing, prefetch=true, kwargs...)`
  - `encode_result(key_or_path, text; format=nothing, prefetch=true, kwargs...)`
  - `decode(key_or_path, ids; format=nothing, prefetch=true)`
- Added in-session tokenizer cache with lock protection:
  - `get_tokenizer_cached(...)`
  - `clear_tokenizer_cache!()`
  - `cached_tokenizers()`
- Exported new APIs in `src/KeemenaSubwords.jl`:
  - `prefetch_models_status`, `asset_status`, `print_asset_status`
  - `get_tokenizer_cached`, `clear_tokenizer_cache!`, `cached_tokenizers`

#### E) Deferred roadmap file
- Added root file `futurework.md` with deferred sections:
  - distribution/reproducibility (including gated checksum persistence)
  - training roadmap
  - HF tokenizer.json parity expansion
  - offsets/structured outputs
  - performance roadmap
  - expanded conformance goldens

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `366/366` tests passed (`KeemenaSubwords sections 1-20`) in default lane.
- Ran: `KEEMENA_TEST_DOWNLOADS=1 julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `438/438` tests passed (`KeemenaSubwords sections 1-20`) in download-enabled lane.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: passed.
- Ran: `julia --project=docs docs/make.jl`
  - Result: docs build/doctests passed.
- Ran: `julia --project=. tools/sync_readme_models.jl --check`
  - Result: README/docs model inventory blocks are in sync.

### Notes
- Normal users now see concise success messages for fallback recovery instead of noisy artifact failure output.
- Maintainers still have full diagnostics available through status APIs and debug env logging.

## 2026-02-11 - Iteration 21

### Objective
Implement section `## 21)` by expanding real-world test coverage with a shared edge-case corpus, cross-tokenizer E2E workflows, stronger HF tokenizer.json assertions, minimal pinned expected-id checks, and more realistic gated download-lane usage.

### Completed section-21 implementation

#### A) Shared edge-case corpus (single source for broad text coverage)
- Added `test/corpus/tokenization_edge_cases.toml` with 100+ curated strings across categories:
  - ASCII baseline
  - whitespace/newline/tab-heavy inputs
  - Unicode normalization-sensitive inputs
  - emoji/symbol sequences
  - mixed scripts
  - RTL examples
  - special-token-like strings
- Added generated long-input definitions (8k/16k/32k) in the corpus file.
- Added `test/helpers/corpus.jl`:
  - `load_edge_case_corpus_categories(; include_long=true)`
  - `load_edge_case_corpus(; include_long=true)`
  - `edge_case_corpus_subset(n; include_long=false, nonempty_only=true)`

#### B) Stronger HF tokenizer.json fixture coverage
- Added new realistic HF fixture:
  - `test/fixtures/hf_json_realistic_pipeline/tokenizer.json`
  - includes normalizer/pretokenizer sequences, TemplateProcessing, decoder, and overlapping added tokens.
- Added byte-fallback-focused HF fixture:
  - `test/fixtures/hf_json_byte_fallback/tokenizer.json`
  - exercises BPE byte fallback with explicit byte tokens.
- Expanded `Section 15 Hugging Face compliance expansion` tests in `test/runtests.jl`:
  - added-token longest match
  - `single_word` boundary behavior
  - special-token bypass behavior
  - template special token insertion checks
  - byte-fallback assertions for unseen-byte inputs
  - corpus-subset robustness loop (encode/decode smoke over many diverse strings)

#### C) Cross-tokenizer E2E user workflows (clearly separated)
- Added new file `test/e2e_user_workflows_extended.jl` and included it from `test/runtests.jl`.
- Added top-level testset `Section 21 E2E user workflows extended` with:
  - one-call API workflows over shared corpus subsets across families:
    - BPE, WordPiece, SentencePiece, HF tokenizer.json, tiktoken fixture
  - local registration workflow over corpus subset
  - long-input smoke tests
  - realistic download-enabled gated workflow (`KEEMENA_TEST_DOWNLOADS=1`) using corpus subsets for:
    - `:tiktoken_o200k_base`
    - `:openai_gpt2_bpe`
    - `:bert_base_uncased_wordpiece`
    - `:t5_small_sentencepiece_unigram`
    - `:qwen2_5_bpe`
  - cache reuse checks via `cached_tokenizers()`.

#### D) Minimal pinned expected-id regressions
- Added `test/golden/minimal_expected_ids.toml` with exact expected IDs for representative models:
  - `core_bpe_en`
  - `core_wordpiece_en`
  - `core_sentencepiece_unigram_en`
  - fixture HF tokenizer.json WordPiece
  - fixture tiktoken model
- Added `test/test_minimal_expected_ids.jl` to validate exact IDs against this TOML.

#### E) Test harness wiring updates
- Updated `test/runtests.jl`:
  - includes corpus helper (`include("helpers/corpus.jl")`)
  - includes new extended E2E file
  - includes minimal expected-ids test file
  - adds shared corpus sanity testset
  - top-level label updated to `KeemenaSubwords sections 1-21`.

### Verification
- Ran: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `1743/1743` tests passed (`KeemenaSubwords sections 1-21`) in default/offline lane.
- Ran: `KEEMENA_TEST_DOWNLOADS=1 julia --project=. -e 'using Pkg; Pkg.test()'`
  - Result: `1962/1962` tests passed in download-enabled lane.
- Ran: `julia --project=. tools/check_docs_examples.jl`
  - Result: docs example consistency checks passed.

### Notes
- Section 21 adds broader user-story coverage while keeping default CI deterministic and offline-stable.
- Download-lane tests now exercise a realistic text subset and one-call cache flow, not only single-string smoke cases.

