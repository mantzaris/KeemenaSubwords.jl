# KeemenaSubwords plan progress

## Iteration 26

### Training scaffolding reorganization (non-breaking)

Summary:
- Reorganized training implementation into dedicated `src/training/` scaffolding.
- Separated training tests into `test/training_runtests.jl`.
- Kept pretrained tokenizer workflows and offset contract untouched.
- Preserved existing training API behavior for `train_bpe`, `train_unigram`, `train_wordpiece`.

Source layout changes:
- Moved:
  - `src/training.jl` -> `src/training/training_api.jl`
  - `src/bpe_train.jl` -> `src/training/bpe_train.jl`
  - `src/unigram_train.jl` -> `src/training/unigram_train.jl`
- Added:
  - `src/training/training.jl` include hub
  - `src/training/wordpiece_train.jl` stub
  - `src/training/sentencepiece_train.jl` stub

Module wiring:
- Updated `src/KeemenaSubwords.jl` to use a single include:
  - `include("training/training.jl")`
- Removed direct includes of old training file paths.
- Added `train_sentencepiece` export as a safe stub entrypoint that throws a clear
  `ArgumentError` (matching current non-implemented training behavior pattern).

Training API behavior:
- `train_bpe` unchanged.
- `train_unigram` unchanged.
- `train_wordpiece` unchanged semantically: still throws the documented
  "not implemented yet" `ArgumentError`.
- Added `train_sentencepiece` stub entrypoint that throws:
  - `"train_sentencepiece is not implemented yet. Use load_tokenizer(...) with an existing model for now."`

Testing separation:
- Moved "Section 7 training implementations" out of `test/runtests.jl` into:
  - `test/training_runtests.jl`
- In `test/runtests.jl`, replaced inline training section with:
  - `include("training_runtests.jl")`
- Kept training tests deterministic and fast:
  - BPE training basic train/tokenize/encode/decode/save-reload checks.
  - Unigram training basic train/tokenize/encode/decode/save-reload checks.
  - WordPiece not-implemented error check preserved.
  - Added SentencePiece training stub not-implemented error check.
- Reserved future heavy-test gate pattern:
  - `ENV["KEEMENA_TEST_TRAINING_HEAVY"] == "1"` scaffold block.

Docs updates:
- Added `docs/src/training.md` (experimental training stub page).
- Added docs navigation entry in `docs/make.jl`:
  - `"Training" => "training.md"`
- Added docs home map link in `docs/src/index.md`.
- Kept README minimal and added a single docs link:
  - `Training (Experimental)` under Documentation links.

Validation:
- `julia --project=. -e 'using Pkg; Pkg.test()'` -> pass
- `julia --project=docs docs/make.jl` -> pass
