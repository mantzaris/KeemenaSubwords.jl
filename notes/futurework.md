# Future Work

This file tracks deferred work that is important for long-term robustness and ecosystem parity, but is intentionally not required for the current v1-ish functionality.

## 1) Full Hugging Face tokenizer.json parity (long tail)
Goal: Load and execute a much larger fraction of real-world Hugging Face `tokenizer.json` pipelines in pure Julia.

Why:
- Many modern HF tokenizers use complex normalizer/pre-tokenizer/post-processor/decoder pipelines.
- Current support covers common core components; parity work closes remaining gaps.

Tasks:
- Expand HF JSON model parity:
  - BPE: handle additional flags and clarify unsupported behavior (for example dropout semantics).
  - Unigram: improve edge-case handling and byte-fallback parity.
  - WordPiece: verify full parity for `max_input_chars_per_word` and continuation prefix rules.
- Expand HF JSON normalizers:
  - continue with `StripAccents`, `Replace`, `Prepend`
  - add additional encountered variants (including NFC/NFD family behaviors).
- Expand HF JSON pre-tokenizers:
  - `Punctuation`, `Digits`, Unicode script boundary splitting, and stronger regex split parity.
  - verify multi-stage `Sequence` behavior.
- Expand HF JSON post-processors:
  - robust `BertProcessing` and `RobertaProcessing` parity.
  - improve pair-sequence special-token insertion parity.
- Expand HF JSON decoders:
  - strengthen `ByteLevel`, `Metaspace`, `WordPiece`, and `BPE` decoder pipeline parity.
- Improve unsupported-component errors:
  - include JSON path, component type, and actionable workaround/export guidance.
- Add more realistic fixtures:
  - multi-stage pipelines and tricky `added_tokens` configurations.

Acceptance signals:
- A broader set of public `tokenizer.json` files load and produce stable IDs.
- Unsupported components fail fast with actionable diagnostics.

## 2) Wider conformance against reference implementations
Goal: Verify encoding parity against upstream tokenizers for selected flagship models over a shared edge-case corpus.

Why:
- Existing goldens and E2E tests cover representative behavior.
- Systematic conformance catches subtle drift in normalization, special tokens, byte fallback, and pipeline semantics.

Tasks:
- Maintain a curated reference set (public, redistributable):
  - tiktoken: `cl100k_base` and/or `o200k_base`
  - GPT-2 byte-level BPE
  - BERT uncased WordPiece
  - T5-small SentencePiece Unigram
  - one HF `tokenizer.json`-first model (for example Qwen2.5).
- Continue expanding shared corpus usage with category-driven subsets.
- Maintain maintainer-only Python golden generation tooling:
  - use `tiktoken`, `tokenizers`, `sentencepiece`, `transformers`
  - generate committed goldens; Julia tests remain Python-free at runtime.
- Expand Julia conformance assertions:
  - exact ID match
  - targeted checks for `added_tokens`, byte fallback, and special-token insertion.
- Add periodic parity maintenance workflow:
  - nightly/manual regeneration and drift reporting.

Acceptance signals:
- Exact ID parity for selected reference tokenizers on shared corpus subsets.
- Clear mismatch diff output to speed investigation.

## 3) Structured outputs, offsets, and alignment semantics
Goal: Provide consistent structured encode outputs across tokenizer families.

Why:
- Downstream annotation, span extraction, and alignment workflows require reliable offsets and masks.
- Users coming from Python tokenizers expect structured output beyond IDs.

Tasks:
- Define and document a stable offsets contract:
  - normalized-text coordinates by default
  - optional original-text coordinate mapping later.
- Improve `encode_result` parity across families:
  - Byte-level offsets
  - BPE/WordPiece subpiece offsets
  - SentencePiece offset semantics clarity.
- Standardize masks where applicable:
  - `attention_mask`
  - `token_type_ids` (especially pair flows)
  - `special_tokens_mask`.
- Add offset invariant tests:
  - monotonic ranges
  - overlap/non-overlap expectations
  - whitespace and special-token handling.
- Document tradeoffs explicitly where normalization changes text length.

Acceptance signals:
- `encode_result` offsets and masks are consistent for common pipelines.
- Documentation clearly states offset semantics by tokenizer family.

## 4) Offsets/alignment split with KeemenaPreprocessing.jl
Decision guidance:
- KeemenaSubwords should own token-level offsets (token to span in a documented coordinate system).
- KeemenaPreprocessing should own higher-level alignment across representations (char/byte/word/sentence) and persist it in `ProcessBundle`.

Recommended split:
- In KeemenaSubwords (future):
  - keep `encode_result` focused on token IDs, tokens, token offsets, and tokenizer-level masks.
- In KeemenaPreprocessing (current/future):
  - build richer alignment projections and multi-stage span tracking on top of tokenizer offsets.

Timing:
- Do not block current stability for full multi-layer alignment semantics.
- Implement incremental offset improvements in KeemenaSubwords first, then compose in KeemenaPreprocessing.

Why this split works:
- Keeps KeemenaSubwords focused and lightweight.
- Avoids duplicating orchestration logic already handled in KeemenaPreprocessing.
- Enables unified cross-stage alignment where it belongs conceptually.

## 5) Distribution and reproducibility
- Persist computed SHA-256 checksums for gated installs after `install_model!` in local registry metadata.
- Keep artifact-hosted public assets pinned to immutable release URLs with verification tooling.
- Continue improving fallback download provenance reporting.

## 6) Tokenizer training (deferred)
- Extend and harden BPE, Unigram, and WordPiece training workflows.
- Add reproducibility and export/import compatibility tests.

## 7) Performance roadmap
- Improve merge/tokenizer hot-path caches.
- Add trie acceleration for WordPiece and optimized DP for Unigram.
- Continue reducing allocations in high-throughput encode/decode paths.
