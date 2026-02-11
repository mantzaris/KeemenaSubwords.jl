# Future Work

## Distribution and Reproducibility
- Persist computed SHA-256 checksums for gated installs after `install_model!` and store them in the local model registry metadata.
- Keep artifact-hosted public assets reproducible with immutable release URLs and verification tooling.
- Continue improving fallback download provenance reporting for long-term auditability.

## Tokenizer Training (Deferred)
- Extend and harden built-in training workflows for BPE, Unigram, and WordPiece.
- Add end-to-end training reproducibility tests and export/import compatibility checks.

## Hugging Face tokenizer.json Parity
- Expand support for additional normalizers, pre-tokenizers, post-processors, and decoders used in the wild.
- Improve added-token edge-case parity and structured output metadata coverage.

## Offsets and Structured Outputs
- Add original-text coordinate offsets where feasible, alongside normalized-coordinate offsets.
- Expand optional encode outputs for downstream pipeline interoperability.

## Performance Roadmap
- Improve hot-path caches for merges and frequently used tokenizers.
- Add trie-based acceleration for WordPiece and optimized DP for Unigram.
- Continue reducing allocation overhead in high-throughput encode/decode workloads.

## Conformance Goldens
- Expand curated golden vectors against upstream reference implementations.
- Add automated maintainer workflows for golden regeneration and drift reporting.
