# Quick Guide Recipes

This page is a choose-your-path entry point for the most common KeemenaSubwords workflows.
Each recipe is written as a guided walkthrough and links to deeper docs when you need detail.

Core invariants:

- Token ids in KeemenaSubwords are 1-based.
- Offsets are 1-based UTF-8 codeunit half-open spans `[start, stop)`.
- `(0, 0)` is the no-span sentinel.

Recommended pipeline contract:

`clean_text` comes from preprocessing, then
`tokenization_text = tokenization_view(tokenizer, clean_text)`, then
`encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, return_masks=true, ...)`.

## How to use this page

Use this 3-step mental model:

1. Choose a tokenizer source:
   shipped registry key, local files, or gated install.
2. Choose output shape:
   single input with `encode_result`, or many inputs with `encode_batch_result` plus collation.
3. Choose metadata:
   offsets for alignment tasks, masks for training tasks, or both.

## Choose a recipe by goal

- I just want token ids from text -> [P1](#p1-load-a-shipped-tokenizer-and-encode-or-decode)
- I need offsets for alignment -> [P3](#p3-get-ids-plus-offsets-plus-masks-for-alignment)
- I have many texts and want a batch -> [P5](#p5-batch-encode-multiple-sequences-no-padding-yet)
- I need training-ready tensors and causal labels -> [P6](#p6-padding-plus-labels-for-training-pointer-recipe)
- I need Python interop -> [P7](#p7-export-to-hugging-face-tokenizerjson-for-python)
- I want to load local files -> [P8](#p8-load-from-a-local-path-auto-detect-plus-override)
- I need a gated tokenizer -> [P9](#p9-install-and-load-a-gated-model)
- I want to train a tokenizer -> [T1](#t1-train-a-tiny-wordpiece-tokenizer-save-reload-and-encode)

## Quick Handlers (1-2 line workflows)

Pick the one matching your goal. You do not need all of these.

- One text -> `quick_tokenize`
- Many texts, no padding -> `quick_encode_batch`
- Many texts, padded + causal labels -> `quick_causal_lm_batch`
- Train a tokenizer bundle locally -> `quick_train_bundle`

### quick_tokenize

When to use: you have one text and want pieces, ids, decoded text, and optional alignment metadata in one call.

```@example quick_handler_quick_tokenize
using KeemenaSubwords
quick_tokenize(:core_bpe_en, "hello world")
```

What you get back:
- `token_ids`
- `token_pieces`
- `decoded_text`

Common options:
- `add_special_tokens=true` (default)
- `return_offsets=true` and `return_masks=true` are enabled by default and safe to ignore when not needed
- `apply_tokenization_view=true` (default)

### quick_encode_batch

When to use: you have many texts and want per-sequence structured outputs without padding yet.

```@example quick_handler_quick_encode_batch
using KeemenaSubwords
quick_encode_batch(:core_wordpiece_en, ["hello world", "hello"]).sequence_lengths
```

What you get back:
- `sequence_lengths` preview in this snippet
- full per-sequence structured outputs at `.results`

Common options:
- `add_special_tokens=true` (default)
- `return_offsets=true` and `return_masks=true` (both default to true)
- `apply_tokenization_view=true` (default)

### quick_causal_lm_batch

When to use: you want training-ready padded tensors and next-token labels in one step.

```@example quick_handler_quick_causal_lm_batch
using KeemenaSubwords
(let out = quick_causal_lm_batch(:core_wordpiece_en, ["hello world", "hello"])
    (ids_size=size(out.ids), labels_size=size(out.labels), pad_token_id=out.pad_token_id)
end)
```

What you get back:
- padded `(seq_len, batch)` `ids`
- padded `(seq_len, batch)` `attention_mask`
- shifted `(seq_len, batch)` causal `labels`

Common options:
- `ignore_index=-100` (default)
- `zero_based=false` (set true only when external tooling expects 0-based labels)
- `pad_to_multiple_of=nothing` (set to align sequence lengths for kernels)

### quick_train_bundle

When to use: you want a local one-call training round-trip (train, save bundle, reload, sanity encode/decode).

```@example quick_handler_quick_train_bundle
using KeemenaSubwords
quick_train_bundle(:wordpiece, ["hello world", "hello tokenizer", "world tokenizer"]; vocab_size=48, min_frequency=1).bundle_files
```

What you get back:
- `bundle_files` preview in this snippet
- returned object also includes `tokenizer`, `training_summary`, `sanity_encoded_ids`, and `sanity_decoded_text`

Common options:
- `vocab_size` and `min_frequency` for training
- `quick_train_bundle(corpus; ...)` defaults to `:wordpiece`
- `sanity_text="hello world"` (default)
- `overwrite=true` and `export_format=:auto` (defaults)

## Pretrained tokenizer recipes (common)

### P1: Load a shipped tokenizer and encode or decode

- **You have:** `text::String`, for example `"hello world"`.
- **You want:** token pieces (`Vector{String}`), token ids (`Vector{Int}`), and a decoded string.
- **Objective:** quickly verify end-to-end tokenization behavior on a built-in model, with the option to inspect offsets and masks in the same call.
- **Steps:**
  1. Call `quick_tokenize(:core_bpe_en, text)` for a one-call output bundle.
  2. Read `token_pieces`, `token_ids`, and `decoded_text`.
  3. If needed, pass options like `add_special_tokens=false` or `return_offsets=false`.

```@example quick_guide_p1
using KeemenaSubwords

text = "hello world"
quick_output = quick_tokenize(
    :core_bpe_en,
    text;
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

(
    token_pieces=quick_output.token_pieces,
    token_ids=quick_output.token_ids,
    decoded_text=quick_output.decoded_text,
    offsets_reference=quick_output.metadata.offsets_reference,
)
```

- **What you should see:**
  - `token_pieces` is a vector of strings.
  - `token_ids` is a vector of integers.
  - `decoded_text` is a string and should be close to input text for covered vocabulary.
- **Concerns and setup notes:**
  - `tokenize` returns readable pieces, `encode` returns integer ids, and `decode` maps ids back to text.
  - `quick_tokenize` uses `encode_result` internally, so offsets and masks are available by default.
  - `add_special_tokens=true` includes model specials (useful for model input); set `false` for raw spans.
  - Ids are always 1-based in this package.
- **Next:** if you need model selection, go to [P2](#p2-discover-models-and-inspect-metadata). If you need offsets, go to [P3](#p3-get-ids-plus-offsets-plus-masks-for-alignment).

#### What `quick_tokenize` does under the hood

```@example quick_guide_p1_under_hood
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)
text = "hello world"

tokenization_text = tokenization_view(tokenizer, text)
result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

(
    token_pieces=result.tokens,
    token_ids=result.ids,
    decoded_text=decode(tokenizer, result.ids),
)
```

### P2: Discover models and inspect metadata

- **You have:** no tokenizer picked yet.
- **You want:** a shortlist of candidates with provenance and defaults.
- **Objective:** choose a safe model key and understand where it comes from.
- **Steps:**
  1. Call `available_models(shipped=true)` for built-in keys.
  2. Call `recommended_defaults_for_llms()` for practical default candidates.
  3. Call `describe_model(key)` for provenance, distribution, and file expectations.

```@example quick_guide_p2
using KeemenaSubwords

shipped_model_keys = available_models(shipped=true)
recommended_keys = recommended_defaults_for_llms()
core_wordpiece_info = describe_model(:core_wordpiece_en)

preview_count = min(length(shipped_model_keys), 8)
shipped_preview = shipped_model_keys[1:preview_count]

(
    shipped_preview=shipped_preview,
    recommended_keys=recommended_keys,
    core_wordpiece=(
        format=core_wordpiece_info.format,
        distribution=core_wordpiece_info.distribution,
        description=core_wordpiece_info.description,
    ),
)
```

- **What you should see:**
  - `shipped_preview` contains local-ready model keys.
  - `recommended_keys` contains practical LLM defaults.
  - `describe_model` returns structured metadata (format, distribution, description, upstream info).
- **Concerns and setup notes:**
  - `shipped` means tiny models included in the repository.
  - `artifact_public` means downloadable public artifacts.
  - `installable_gated` means install flow requires credentials and license acceptance.
  - This recipe is offline-safe because it does not download.
- **Next:** model inventory details are in [Built-In Models](models.md). For install flows, go to [P9](#p9-install-and-load-a-gated-model).

### P3: Get ids plus offsets plus masks for alignment

- **You have:** `clean_text::String` from preprocessing.
- **You want:** a `TokenizationResult` with ids, offsets, and masks.
- **Objective:** get alignment-ready metadata in one call.
- **Steps:**
  1. Call `load_tokenizer(...)`.
  2. Call `tokenization_view(tokenizer, clean_text)` to get tokenizer-coordinate text.
  3. Call `encode_result(...; assume_normalized=true, return_offsets=true, return_masks=true)`.
  4. Read `result.metadata.offsets_reference` to confirm what string offsets are relative to.

```@example quick_guide_p3
using KeemenaSubwords

tokenizer = load_tokenizer(:core_sentencepiece_unigram_en)
clean_text = "Hello, world! Offsets demo."
tokenization_text = tokenization_view(tokenizer, clean_text)

result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

@assert result.offsets !== nothing
@assert result.special_tokens_mask !== nothing

preview_rows = [
    (
        token_index=i,
        token=result.tokens[i],
        offset=result.offsets[i],
        is_special=result.special_tokens_mask[i] == 1,
    )
    for i in 1:min(length(result.ids), 12)
]

(
    offsets_reference=result.metadata.offsets_reference,
    token_count=length(result.ids),
    preview_rows=preview_rows,
)
```

- **What you should see:**
  - `result.offsets` and `result.special_tokens_mask` are present.
  - `offsets_reference` is `:input_text` because `assume_normalized=true` and you passed `tokenization_text`.
  - Preview rows include offsets and special-token flags per token.
- **Concerns and setup notes:**
  - Use `assume_normalized=true` only when the input text is already `tokenization_view(...)` output.
  - Offsets are relative to whatever `offsets_reference` says.
  - Inserted specials often have `(0, 0)` sentinel offsets.
- **Next:** for offset semantics go to [Normalization and Offsets Contract](normalization_offsets_contract.md). For alignment algorithms go to [Offsets Alignment Examples](offset_alignment_examples.md).

### P4: Span text extraction from offsets (safe slicing)

- **You have:** token offsets from `TokenizationResult`.
- **You want:** readable text snippets per token span.
- **Objective:** debug and inspect token alignment quickly.
- **Steps:**
  1. Build `tokenization_text` with `tokenization_view`.
  2. Produce offsets with `encode_result(...; return_offsets=true)`.
  3. Call `try_span_substring(tokenization_text, offset)` for each offset.

```@example quick_guide_p4
using KeemenaSubwords

tokenizer = load_tokenizer(:core_sentencepiece_unigram_en)
tokenization_text = tokenization_view(tokenizer, "Hello, world! Offsets demo.")
result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

@assert result.offsets !== nothing

span_preview = [
    (
        token_index=i,
        token=result.tokens[i],
        offset=result.offsets[i],
        span_text=try_span_substring(tokenization_text, result.offsets[i]),
    )
    for i in 1:min(length(result.ids), 12)
]

span_preview
```

- **What you should see:**
  - Most spanful offsets produce `span_text::String`.
  - Sentinel or empty spans produce `""`.
  - In byte-level cases, `try_span_substring` may return `nothing` for non-boundary spans.
- **Concerns and setup notes:**
  - `try_span_substring` returns `nothing` when boundaries are not valid Julia string boundaries.
  - If you need bytes regardless of boundaries, use `span_codeunits(tokenization_text, offset)`.
  - Keep extraction text and offset coordinate text consistent (`tokenization_text`).
- **Next:** go to [Offsets Alignment Examples](offset_alignment_examples.md) for overlap mapping and span-label workflows.

### P5: Batch encode multiple sequences (no padding yet)

- **You have:** many texts (`Vector{String}`).
- **You want:** `Vector{TokenizationResult}` with one structured output per input sequence.
- **Objective:** prepare data for later collation while preserving per-sequence metadata.
- **Steps:**
  1. Normalize each input with `tokenization_view`.
  2. Call `encode_batch_result(...)` with offsets and masks enabled.
  3. Inspect per-sequence lengths before padding.

```@example quick_guide_p5
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
clean_texts = ["hello world", "hello", "world hello world"]
tokenization_texts = [tokenization_view(tokenizer, clean_text) for clean_text in clean_texts]

batch_results = encode_batch_result(
    tokenizer,
    tokenization_texts;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=true,
    return_masks=true,
)

sequence_lengths = [length(result.ids) for result in batch_results]

(
    sequence_lengths=sequence_lengths,
    has_variable_lengths=length(unique(sequence_lengths)) > 1,
)
```

- **What you should see:**
  - `batch_results` is a vector, not a matrix.
  - Sequence lengths can differ.
  - Each element still has its own ids, masks, and optional offsets.
- **Concerns and setup notes:**
  - No padding is applied automatically.
  - This is intentional: you can choose task-specific collation later.
- **Next:** for padding and training tensors, go to [P6](#p6-padding-plus-labels-for-training-pointer-recipe) and [Structured Outputs and Batching](structured_outputs_and_batching.md).

### P6: Padding plus labels for training (pointer recipe)

- **You have:** many input texts (`Vector{String}`), or precomputed `Vector{TokenizationResult}` if you split steps yourself.
- **You want:** padded `(seq_len, batch)` matrices and causal LM labels.
- **Objective:** build training-ready tensors with explicit padding and masking behavior.
- **Steps:**
  1. Call `quick_causal_lm_batch(...)` to run encode, collate, and label-shift in one call.
  2. Read `ids`, `attention_mask`, and `labels`.
  3. Use `zero_based=true` only for external consumers that expect 0-based labels.

```@example quick_guide_p6
using KeemenaSubwords

training_batch = quick_causal_lm_batch(
    :core_wordpiece_en,
    ["hello world", "hello"];
    add_special_tokens=true,
    return_offsets=false,
    ignore_index=-100,
    zero_based=false,
)

@assert size(training_batch.ids) == size(training_batch.attention_mask)
@assert size(training_batch.labels) == size(training_batch.ids)

(
    ids_size=size(training_batch.ids),
    labels_size=size(training_batch.labels),
    ignore_index_count=count(==(-100), training_batch.labels),
    pad_token_id=training_batch.pad_token_id,
)
```

- **What you should see:**
  - `ids`, `attention_mask`, and `labels` all share the same matrix shape.
  - `ignore_index_count` is positive (padding and final-token masking).
  - `pad_token_id` is inferred from tokenizer `pad_id` or `eos_id`.
- **Concerns and setup notes:**
  - KeemenaSubwords ids are 1-based.
  - `ignore_index=-100` is the common causal LM training convention.
  - Final valid token in each sequence should remain ignored.
- **Next:** go to [Structured Outputs and Batching](structured_outputs_and_batching.md) for fuller collation, causal labels, and block packing.

#### What `quick_causal_lm_batch` does under the hood

```@example quick_guide_p6_under_hood
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
input_texts = ["hello world", "hello"]

batch_encoding = quick_encode_batch(
    tokenizer,
    input_texts;
    add_special_tokens=true,
    return_offsets=false,
    return_masks=true,
)

collated = collate_padded_batch(batch_encoding.results; tokenizer=tokenizer)
labels = causal_lm_labels(collated.ids, collated.attention_mask; ignore_index=-100, zero_based=false)

(
    sequence_lengths=batch_encoding.sequence_lengths,
    ids_size=size(collated.ids),
    labels_size=size(labels),
)
```

### P7: Export to Hugging Face tokenizer.json for Python

- **You have:** a tokenizer loaded in Julia.
- **You want:** a `tokenizer.json` file that Python can load.
- **Objective:** share identical tokenization rules across Julia and Python.
- **Steps:**
  1. Load or train a tokenizer in Julia.
  2. Call `export_tokenizer(...; format=:hf_tokenizer_json)`.
  3. Load the emitted `tokenizer.json` in Python using `PreTrainedTokenizerFast`.

```@example quick_guide_p7
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
output_directory = mktempdir()
export_tokenizer(tokenizer, output_directory; format=:hf_tokenizer_json)

isfile(joinpath(output_directory, "tokenizer.json"))
```

Python usage (non-executable in Documenter):

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="out_tokenizer/tokenizer.json")
```

- **What you should see:**
  - Julia writes `tokenizer.json` to the output directory.
  - Python loads that same file with `PreTrainedTokenizerFast`.
- **Concerns and setup notes:**
  - Exported file captures tokenizer pipeline behavior supported by the package.
  - Keep format contracts in mind when sharing across runtimes.
- **Next:** details are in [Tokenizer Formats and Required Files](formats.md) and [LLM Cookbook](llm_cookbook.md).

### P8: Load from a local path (auto-detect plus override)

- **You have:** local tokenizer files on disk.
- **You want:** a loaded tokenizer without guessing format details manually.
- **Objective:** use auto-detection when it works, and explicit overrides when needed.
- **Steps:**
  1. Try `load_tokenizer("/path/to/model_dir")` first.
  2. If format is ambiguous, set `format=:...` explicitly.
  3. Validate with a tiny `encode` or `tokenize` call.

Decision tree:

- If auto-detect works, use `load_tokenizer(path)`.
- If ambiguous or incorrect, use `load_tokenizer(path; format=:...)`.

```julia
# non-executable path placeholders
tokenizer_auto = load_tokenizer("/path/to/model_dir")
tokenizer_tiktoken = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
tokenizer_sentencepiece = load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```

- **What you should see:**
  - Auto-detect succeeds for common directory layouts.
  - Explicit override resolves ambiguous `.model` cases.
- **Concerns and setup notes:**
  - `format` selects a file contract (required filenames and parsing rules).
  - Placeholder paths here are non-executable in docs.
- **Next:** go to [Loading Tokenizers From Local Paths](loading_local.md) and [Tokenizer Formats and Required Files](formats.md).

### P9: Install and load a gated model

- **You have:** model access credentials and accepted upstream license terms.
- **You want:** installed local assets for a gated model key.
- **Objective:** run a reproducible install-then-load workflow.
- **Steps:**
  1. Set an HF token (for example in `ENV["HF_TOKEN"]`).
  2. Call `install_model!(...; token=ENV["HF_TOKEN"])`.
  3. Call `load_tokenizer(:model_key)` after install.

```julia
# non-executable gated workflow
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
tokenizer = load_tokenizer(:llama3_8b_tokenizer)
```

- **What you should see:**
  - Install step fetches and stores assets for the key.
  - Load step then resolves locally by model key.
- **Concerns and setup notes:**
  - You must accept upstream license terms before access is granted.
  - Keep secrets in environment variables, not source files.
- **Next:** go to [Installable Gated Models](gated_models.md) and [LLM Cookbook](llm_cookbook.md).

## Training recipes (experimental)

Training is usually appropriate when you need domain adaptation, research control over vocabulary behavior, or constrained deployments that need custom tokenizer assets.
Training APIs are experimental and may evolve faster than pretrained loading and encoding APIs.

### T1: Train a tiny WordPiece tokenizer, save, reload, and encode

- **You have:** a small in-memory corpus (`Vector{String}`).
- **You want:** a trained tokenizer bundle you can reload reproducibly.
- **Objective:** run a fully local training round trip without network access.
- **Steps:**
  1. Call `quick_train_bundle(:wordpiece, corpus; ...)`.
  2. Read `bundle_files` and `tokenizer`.
  3. Confirm sanity ids and decoded text from the returned fields.

```@example quick_guide_t1
using KeemenaSubwords

training_corpus = [
    "hello world",
    "hello tokenizer",
    "world of subwords",
]

training_output = quick_train_bundle(
    :wordpiece,
    training_corpus;
    vocab_size=64,
    min_frequency=1,
    sanity_text="hello world",
)

(
    bundle_directory=training_output.bundle_directory,
    bundle_files=training_output.bundle_files,
    sanity_encoded_ids=training_output.sanity_encoded_ids,
    sanity_decoded_text=training_output.sanity_decoded_text,
)
```

- **What you should see:**
  - `bundle_files` includes tokenizer exports and the training manifest.
  - Returned tokenizer has already been reloaded from bundle.
  - Workflow is deterministic for the same corpus and config.
- **Concerns and setup notes:**
  - The bundle gives reproducible reload without remembering loader kwargs.
  - `quick_train_bundle` currently supports `:wordpiece` plus HF preset trainer symbols.
  - Avoid asserting exact token ids across different configs.
- **Next:** full API and preset coverage are in [Training (experimental)](training.md).

#### What `quick_train_bundle` does under the hood

```@example quick_guide_t1_under_hood
using KeemenaSubwords

training_corpus = ["hello world", "hello tokenizer", "world tokenizer"]
training_result = train_wordpiece_result(training_corpus; vocab_size=48, min_frequency=1)

bundle_directory = mktempdir()
save_training_bundle(training_result, bundle_directory; overwrite=true)
reloaded_tokenizer = load_training_bundle(bundle_directory)

encoded_ids = encode(reloaded_tokenizer, "hello world"; add_special_tokens=false)
decoded_text = decode(reloaded_tokenizer, encoded_ids)

(
    bundle_files=sort(readdir(bundle_directory)),
    encoded_ids=encoded_ids,
    decoded_text=decoded_text,
)
```

### T2: Train HF BERT WordPiece preset and export tokenizer.json

- **You have:** text suited to BERT-style tokenization behavior.
- **You want:** a BERT preset tokenizer and optional HF export.
- **Objective:** use familiar BERT normalization and pretokenization defaults.
- **Steps:**
  1. Call `train_hf_bert_wordpiece(corpus; ...)`.
  2. Export with `export_tokenizer(...; format=:hf_tokenizer_json)`.

```julia
# non-executable training preset sketch
training_corpus = ["Hello, world!", "Tokenizer training example"]
tokenizer = train_hf_bert_wordpiece(training_corpus; vocab_size=128, min_frequency=1)
export_tokenizer(tokenizer, "out_hf_bert"; format=:hf_tokenizer_json)
```

- **What you should see:**
  - Training returns a WordPiece tokenizer configured for BERT-style behavior.
  - Export creates `tokenizer.json` for HF interop.
- **Concerns and setup notes:**
  - Presets are convenient defaults, not strict replication of every upstream variant.
- **Next:** see [Training (experimental)](training.md) and [Tokenizer Formats and Required Files](formats.md).

### T3: Train HF RoBERTa or GPT-2 ByteBPE preset

- **You have:** corpus data for byte-level subword training.
- **You want:** a byte-level preset tokenizer in RoBERTa or GPT-2 style.
- **Objective:** use byte-level presets for robust coverage of arbitrary UTF-8 input.
- **Steps:**
  1. Call `train_hf_roberta_bytebpe(...)` or `train_hf_gpt2_bytebpe(...)`.
  2. Export with `export_tokenizer(...; format=:hf_tokenizer_json)` if needed.

```julia
# non-executable training preset sketch
training_corpus = ["hello world", "cafe costs 5"]
tokenizer = train_hf_roberta_bytebpe(training_corpus; vocab_size=384, min_frequency=1)
export_tokenizer(tokenizer, "out_hf_roberta"; format=:hf_tokenizer_json)
```

- **What you should see:**
  - Training returns a byte-level tokenizer preset.
  - Exported artifacts are reloadable in Julia and usable in compatible external tools.
- **Concerns and setup notes:**
  - Byte-level offsets still follow the package contract.
  - Span boundaries may not always be safe Julia string boundaries on multibyte text.
- **Next:** offset rules are in [Normalization and Offsets Contract](normalization_offsets_contract.md) and examples are in [Offsets Alignment Examples](offset_alignment_examples.md).

## Other options (short list)

- Cache tokenizers for repeated use with `get_tokenizer_cached(...)` and clear cache with `clear_tokenizer_cache!()`.
- Use explicit loaders when file contracts are known, for example `load_bpe_gpt2`, `load_sentencepiece`, and `load_tiktoken`.
- Convert to 0-based ids only when an external consumer requires it:
  `ids_zero_based = token_ids .- 1`.
