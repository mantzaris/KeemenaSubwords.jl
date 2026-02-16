# Quick Guide Recipes

This page is a choose-your-path entry point for the most common KeemenaSubwords workflows.
Each recipe is short and points to deeper docs when you need details.

Core invariants:

- Token ids in KeemenaSubwords are 1-based.
- Offsets are 1-based UTF-8 codeunit half-open spans `[start, stop)`.
- `(0, 0)` is the no-span sentinel.

Recommended pipeline contract:
`clean_text` comes from preprocessing, then
`tokenization_text = tokenization_view(tokenizer, clean_text)`, then
`encode_result(tokenizer, tokenization_text; assume_normalized=true, return_offsets=true, return_masks=true, ...)`.

## Pretrained tokenizer recipes (common)

### P1: Load a shipped tokenizer and encode or decode

When to use: you want the fastest path to tokenize text with a built-in model and verify round-trip behavior.

```@example quick_guide_p1
using KeemenaSubwords

tokenizer = load_tokenizer(:core_bpe_en)
text = "hello world"

token_pieces = tokenize(tokenizer, text)
token_ids = encode(tokenizer, text; add_special_tokens=true)
decoded_text = decode(tokenizer, token_ids)

(token_pieces=token_pieces, token_ids=token_ids, decoded_text=decoded_text)
```

Go deeper: [Concepts](concepts.md), [Loading Tokenizers](loading.md).

### P2: Discover models and inspect metadata

When to use: you want to pick a model key, inspect provenance, and list practical defaults.

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

Optional prefetch for offline-safe shipped models:

```julia
prefetch_models([:core_bpe_en, :core_wordpiece_en, :core_sentencepiece_unigram_en])
```

Go deeper: [Built-In Models](models.md), [LLM Cookbook](llm_cookbook.md).

### P3: Get ids plus offsets plus masks for alignment

When to use: you need token ids and reliable alignment metadata in one result object.

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

Go deeper: [Normalization and Offsets Contract](normalization_offsets_contract.md), [Offsets Alignment Examples](offset_alignment_examples.md), [Structured Outputs and Batching](structured_outputs_and_batching.md).

### P4: Span text extraction from offsets (safe slicing)

When to use: you want substring views for token spans without assuming all offsets are safe string boundaries.

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

Byte-level caveat: some byte-level tokenizers can emit non-boundary spans on multibyte text. Use `try_span_substring` first, then `span_codeunits` fallback when needed.

Go deeper: [Offsets Alignment Examples](offset_alignment_examples.md), [Normalization and Offsets Contract](normalization_offsets_contract.md).

### P5: Batch encode multiple sequences (no padding yet)

When to use: you want per-sequence structured outputs before collation.

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

Go deeper: [Structured Outputs and Batching](structured_outputs_and_batching.md).

### P6: Padding plus labels for training (pointer recipe)

When to use: you want minimal tensors for causal LM training with `ignore_index=-100` behavior.

```@example quick_guide_p6
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
clean_texts = ["hello world", "hello"]
tokenization_texts = [tokenization_view(tokenizer, clean_text) for clean_text in clean_texts]
batch_results = encode_batch_result(
    tokenizer,
    tokenization_texts;
    assume_normalized=true,
    add_special_tokens=true,
    return_offsets=false,
    return_masks=true,
)

function tiny_pad_batch(results::Vector{TokenizationResult}; pad_token_id::Int)
    batch_size = length(results)
    max_length = maximum(length(result.ids) for result in results)
    ids = fill(pad_token_id, max_length, batch_size)
    attention_mask = fill(0, max_length, batch_size)
    for (column_index, result) in pairs(results)
        sequence_length = length(result.ids)
        ids[1:sequence_length, column_index] = result.ids
        attention_mask[1:sequence_length, column_index] .= 1
    end
    return (ids=ids, attention_mask=attention_mask)
end

function tiny_causal_labels(ids::Matrix{Int}, attention_mask::Matrix{Int}; ignore_index::Int=-100)
    labels = fill(ignore_index, size(ids))
    for column_index in axes(ids, 2)
        valid_positions = findall(attention_mask[:, column_index] .== 1)
        for i in 1:(length(valid_positions) - 1)
            labels[valid_positions[i], column_index] = ids[valid_positions[i + 1], column_index]
        end
    end
    return labels
end

collated = tiny_pad_batch(batch_results; pad_token_id=pad_id(tokenizer))
labels = tiny_causal_labels(collated.ids, collated.attention_mask; ignore_index=-100)
labels_zero_based = map(label -> label == -100 ? -100 : label - 1, labels)

(
    ids_size=size(collated.ids),
    labels_size=size(labels),
    ignore_index_count=count(==(-100), labels),
    labels_zero_based=labels_zero_based,
)
```

Go deeper: [Structured Outputs and Batching](structured_outputs_and_batching.md) for full padding, causal labels, and block packing recipes.

### P7: Export to Hugging Face tokenizer.json for Python

When to use: you need interop with Python fast tokenizers.

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

Go deeper: [Tokenizer Formats and Required Files](formats.md), [LLM Cookbook](llm_cookbook.md).

### P8: Load from a local path (auto-detect plus override)

When to use: model files already exist locally and you want either detection or explicit format control.

```julia
# non-executable path placeholders
tokenizer_auto = load_tokenizer("/path/to/model_dir")
tokenizer_tiktoken = load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)
tokenizer_sentencepiece = load_tokenizer("/path/to/tokenizer.model"; format=:sentencepiece_model)
```

Go deeper: [Loading Tokenizers From Local Paths](loading_local.md), [Tokenizer Formats and Required Files](formats.md).

### P9: Install and load a gated model

When to use: you need a gated upstream tokenizer and have access credentials.

```julia
# non-executable gated workflow
install_model!(:llama3_8b_tokenizer; token=ENV["HF_TOKEN"])
tokenizer = load_tokenizer(:llama3_8b_tokenizer)
```

You must accept upstream license terms and have valid access before install.

Go deeper: [Installable Gated Models](gated_models.md), [LLM Cookbook](llm_cookbook.md).

## Training recipes (experimental)

Training APIs are experimental and may evolve faster than pretrained loading and encoding APIs.

### T1: Train a tiny WordPiece tokenizer, save, reload, and encode

When to use: you want a self-contained training round trip without network access.

```@example quick_guide_t1
using KeemenaSubwords

training_corpus = [
    "hello world",
    "hello tokenizer",
    "world of subwords",
]

training_result = train_wordpiece_result(
    training_corpus;
    vocab_size=64,
    min_frequency=1,
)

bundle_directory = mktempdir()
save_training_bundle(training_result, bundle_directory; overwrite=true)
reloaded_tokenizer = load_training_bundle(bundle_directory)

encoded_ids = encode(reloaded_tokenizer, "hello world"; add_special_tokens=false)
decoded_text = decode(reloaded_tokenizer, encoded_ids)
bundle_files = sort(readdir(bundle_directory))

(
    encoded_ids=encoded_ids,
    decoded_text=decoded_text,
    bundle_files=bundle_files,
)
```

### T2: Train HF BERT WordPiece preset and export tokenizer.json

When to use: you want a BERT-style preset with direct HF `tokenizer.json` export.

```julia
# non-executable training preset sketch
training_corpus = ["Hello, world!", "Tokenizer training example"]
tokenizer = train_hf_bert_wordpiece(training_corpus; vocab_size=128, min_frequency=1)
export_tokenizer(tokenizer, "out_hf_bert"; format=:hf_tokenizer_json)
```

### T3: Train HF RoBERTa or GPT-2 ByteBPE preset

When to use: you want byte-level preset behavior for GPT-2 or RoBERTa style pipelines.

```julia
# non-executable training preset sketch
training_corpus = ["hello world", "cafe costs 5"]
tokenizer = train_hf_roberta_bytebpe(training_corpus; vocab_size=384, min_frequency=1)
export_tokenizer(tokenizer, "out_hf_roberta"; format=:hf_tokenizer_json)
```

Byte-level reminder: offsets still follow the same contract, but span boundaries may not always be safe Julia string boundaries on multibyte text.

Go deeper:

- [Training (experimental)](training.md)
- [Tokenizer Formats and Required Files](formats.md)
- [Normalization and Offsets Contract](normalization_offsets_contract.md)

## Other options (short list)

- Cache tokenizers for repeated use with `get_tokenizer_cached(...)` and clear cache with `clear_tokenizer_cache!()`.
- Use explicit loaders when file contracts are known, for example `load_bpe_gpt2`, `load_sentencepiece`, and `load_tiktoken`.
- Convert to 0-based ids only when an external consumer requires it:
  `ids_zero_based = token_ids .- 1`.
