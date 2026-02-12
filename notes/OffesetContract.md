## Architecture decision: Normalization + offsets contract for KeemenaPreprocessing <-> KeemenaSubwords

### Context
KeemenaPreprocessing offers user-configurable normalization/cleaning and must maintain robust alignments (char/byte/word/sentence). KeemenaSubwords performs tokenizer-specific normalization (as defined by the tokenizer) and subword tokenization, returning ids and token-level offsets.

Without a clear split, there is a risk of:
- double-normalization (eg lowercase or NFKC applied twice),
- token ids drifting from pretrained/reference behavior,
- offsets computed on one text view while tokenization happens on another, breaking alignment.

### Principle
KeemenaPreprocessing should orchestrate normalization and define canonical text views, but must not re-implement tokenizer-specific normalization logic. Tokenizer intrinsic normalization must remain authoritative and implemented in KeemenaSubwords.

### Definitions
There are two distinct kinds of normalization:

1) **Pipeline normalization (KeemenaPreprocessing)**
- Purpose: general cleaning and user-configurable preprocessing that supports the overall pipeline (segmentation, cleaning, shaping the canonical pre-tokenization text).
- Examples: trimming, whitespace policy, basic text cleanup, user-selected Unicode normalization if used consistently across downstream steps.
- Output: `clean_text`.

2) **Tokenizer intrinsic normalization (KeemenaSubwords)**
- Purpose: match the tokenizer’s expected normalization exactly (eg HF tokenizer.json normalizer pipeline, WordPiece casing rules).
- This normalization is part of the tokenizer definition and is required for correct ids.
- Output: `tokenization_text` (aka tokenizer-normalized view).

### Policy when subwords are enabled
When subwords are enabled, KeemenaPreprocessing normalization is split into two buckets:

- **Bucket A: Pipeline normalization (Preprocessing)**
  - Applied to raw input to produce `clean_text`.
  - Must be stable and well-defined because it affects all later alignments.

- **Bucket B: Tokenizer intrinsic normalization (Subwords)**
  - Applied to `clean_text` using KeemenaSubwords’ tokenizer normalization to produce `tokenization_text`.
  - Preprocessing must not replicate tokenizer normalization rules.

### Canonical offsets rule (must-have integration invariant)
All word and subword offsets used for alignment are computed on the same canonical string:

- `tokenization_text`

This means:
- Word spans (for word-level tokenization in KeemenaPreprocessing) are computed on `tokenization_text`.
- Subword token offsets returned by KeemenaSubwords are also relative to `tokenization_text`.
- Alignment tables (word <-> subword, sentence <-> subword) are built on this shared coordinate system.

### Does "Preprocessing normalizes first, then Subwords normalizes" work?
Yes, this is the simplest robust default, provided the following rule is enforced:

- Preprocessing applies pipeline normalization first (Bucket A).
- Subwords applies tokenizer intrinsic normalization second (Bucket B).
- The same tokenizer normalization must not be applied twice.

To ensure this, KeemenaSubwords must expose a switch to bypass intrinsic normalization during tokenization:

- `encode_result(tokenizer, text; assume_normalized=true, ...)`

Meaning: "text is already tokenizer-normalized; do not run tokenizer normalizer again".

### Canonical integration flow (the contract)
KeemenaPreprocessing orchestrates:

1) `clean_text = preprocessing_normalize(raw_text)`  
   (Pipeline normalization: cleaning, user-configured preprocessing)

2) `tokenization_text = KeemenaSubwords.normalize(tokenizer, clean_text)`  
   (Tokenizer intrinsic normalization, implemented in KeemenaSubwords)

3) `subword_result = KeemenaSubwords.encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, return_masks=true, ...)`  
   (Tokenization on canonical tokenization_text, with offsets relative to that same string)

4) Word offsets are computed on `tokenization_text`  
   (so word and subword offsets share a coordinate system)

5) KeemenaPreprocessing builds alignment maps in ProcessBundle using:
- word offsets on `tokenization_text`
- subword offsets on `tokenization_text`
- special_tokens_mask to exclude special tokens from span alignment

### Required outputs from KeemenaSubwords for alignment
For robust integration, KeemenaSubwords must provide:
- `ids` (token ids)
- `tokens` (optional but recommended for debugging)
- `offsets` per token relative to `tokenization_text`
- `special_tokens_mask` (1 for special tokens, else 0)
- (optional) `token_type_ids` for pair sequences when supported

### Notes on user configuration and overlap
If a user configures normalization in KeemenaPreprocessing that overlaps with tokenizer intrinsic normalization, Preprocessing should:
- still produce `clean_text` using pipeline normalization,
- still produce `tokenization_text` via KeemenaSubwords.normalize,
- optionally warn if the pipeline normalization likely changes token ids (advanced UX decision).
