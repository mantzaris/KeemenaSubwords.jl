# Structured Outputs and Batching

This page shows practical recipes for turning tokenizer outputs into training-ready batch tensors and label alignments.

## `TokenizationResult` Fields And Invariants

`encode_result` and `encode_batch_result` return `TokenizationResult` values with:

- `ids`: token ids (`Vector{Int}`), using KeemenaSubwords 1-based id space.
- `tokens`: token strings (`Vector{String}`) aligned position-by-position with `ids`.
- `offsets`: `Union{Nothing, Vector{Tuple{Int,Int}}}` when `return_offsets=true`.
  Offsets are 1-based UTF-8 codeunit half-open spans `[start, stop)`.
  Sentinel `(0, 0)` means no source-text span.
- `attention_mask`: `Union{Nothing, Vector{Int}}` when `return_masks=true`.
  For a single unpadded `encode_result` call this is all ones.
- `token_type_ids`: `Union{Nothing, Vector{Int}}` when `return_masks=true`.
  Current default is zeros for single-sequence examples.
- `special_tokens_mask`: `Union{Nothing, Vector{Int}}` when `return_masks=true`.
  Marks special-token identity.
  A token can be special and still have a real span (for example `[UNK]` in WordPiece).
- `metadata`: named tuple with model and offsets metadata.
  `metadata.offsets_reference` is:
  - `:input_text` when `assume_normalized=true`
  - `:tokenizer_normalized_text` when `assume_normalized=false`

## Example 1: Inspect `encode_result` Output

```@example structured_outputs_batching
using KeemenaSubwords

tokenizer = load_tokenizer(:core_wordpiece_en)
clean_text = "hello world"
tokenization_text = tokenization_view(tokenizer, clean_text)

result = encode_result(
    tokenizer,
    tokenization_text;
    assume_normalized = true,
    add_special_tokens = true,
    return_offsets = true,
    return_masks = true,
)

result_default_reference = encode_result(
    tokenizer,
    clean_text;
    assume_normalized = false,
    add_special_tokens = true,
    return_offsets = true,
    return_masks = true,
)

@assert result.offsets !== nothing
@assert result.attention_mask !== nothing
@assert result.special_tokens_mask !== nothing

rows = [
    (
        token_index = i,
        token_id = result.ids[i],
        token = result.tokens[i],
        offset = result.offsets[i],
        special = result.special_tokens_mask[i],
        span_text = try_span_substring(tokenization_text, result.offsets[i]),
    )
    for i in eachindex(result.ids)
]

(
    ids = result.ids,
    tokens = result.tokens,
    attention_mask_unique_values = unique(result.attention_mask),
    offsets_reference = (
        assume_normalized_true = result.metadata.offsets_reference,
        assume_normalized_false = result_default_reference.metadata.offsets_reference,
    ),
    token_rows = rows,
)
```

What this shows:

- Inserted specials are typically sentinel spans `(0, 0)` with `special == 1`.
- `special_tokens_mask` marks token identity, not span participation.
- In this example, `[UNK]` is special but still has a spanful offset from source text.

## Example 2: `encode_batch_result` Returns Per-Sequence Outputs

```@example structured_outputs_batching
clean_texts = [
    "hello world",
    "world hello world",
    "offset demo",
]
tokenization_texts = [tokenization_view(tokenizer, text) for text in clean_texts]

batch_results = encode_batch_result(
    tokenizer,
    tokenization_texts;
    assume_normalized = true,
    add_special_tokens = true,
    return_offsets = true,
    return_masks = true,
)

batch_summary = [
    (
        sequence_index = i,
        n_ids = length(batch_results[i].ids),
        n_offsets = batch_results[i].offsets === nothing ? 0 : length(batch_results[i].offsets),
        attention_mask = batch_results[i].attention_mask,
    )
    for i in eachindex(batch_results)
]

batch_summary
```

`encode_batch_result` returns `Vector{TokenizationResult}`.
Padding is not applied automatically. Each sequence keeps its own length.

## Example 3: Padding Collator Recipe

Policy in this example:

- output layout is `(seq_len, batch)` (column-major friendly in Julia),
- `ids` pad with `pad_id`,
- padded `attention_mask` positions are `0`,
- padded `special_tokens_mask` positions are marked as `1` (`mark_pad_as_special=true`).

```@example structured_outputs_batching
function pad_batch(
    results::Vector{TokenizationResult};
    pad_id::Int,
    mark_pad_as_special::Bool = true,
)
    batch_size = length(results)
    max_len = maximum(length(r.ids) for r in results)

    ids = fill(pad_id, max_len, batch_size)
    attention_mask = fill(0, max_len, batch_size)
    special_tokens_mask = fill(mark_pad_as_special ? 1 : 0, max_len, batch_size)

    for (col, r) in pairs(results)
        seq_len = length(r.ids)
        ids[1:seq_len, col] = r.ids
        attention_mask[1:seq_len, col] .= 1
        if r.special_tokens_mask === nothing
            special_tokens_mask[1:seq_len, col] .= 0
        else
            special_tokens_mask[1:seq_len, col] = r.special_tokens_mask
        end
    end

    return (
        ids = ids,
        attention_mask = attention_mask,
        special_tokens_mask = special_tokens_mask,
    )
end

collated = pad_batch(batch_results; pad_id = pad_id(tokenizer))

(
    ids_size = size(collated.ids),
    attention_mask_size = size(collated.attention_mask),
    special_tokens_mask_size = size(collated.special_tokens_mask),
    ids = collated.ids,
    attention_mask = collated.attention_mask,
    special_tokens_mask = collated.special_tokens_mask,
)
```

## Example 4: Causal LM Labels (Next-Token Prediction)

Policy in this example:

- `ignore_index = -100` for all padded positions,
- `ignore_index = -100` for the final valid token in each sequence (no next token target),
- labels otherwise point to the next valid token id.

```@example structured_outputs_batching
function causal_lm_labels(
    ids::AbstractMatrix{<:Integer},
    attention_mask::AbstractMatrix{<:Integer};
    ignore_index::Int = -100,
)::Matrix{Int}
    size(ids) == size(attention_mask) || throw(ArgumentError("ids and attention_mask must have the same shape"))

    seq_len, batch_size = size(ids)
    labels = fill(ignore_index, seq_len, batch_size)

    for col in 1:batch_size
        valid_positions = findall(attention_mask[:, col] .== 1)
        isempty(valid_positions) && continue

        for i in 1:(length(valid_positions) - 1)
            pos = valid_positions[i]
            next_pos = valid_positions[i + 1]
            labels[pos, col] = ids[next_pos, col]
        end

        labels[last(valid_positions), col] = ignore_index
    end

    return labels
end

labels = causal_lm_labels(collated.ids, collated.attention_mask; ignore_index=-100)

# KeemenaSubwords ids are 1-based. Convert only if an external consumer expects 0-based ids.
labels_zero_based = map(labels) do label
    label == -100 ? -100 : label - 1
end

(
    ids = collated.ids,
    attention_mask = collated.attention_mask,
    labels = labels,
    labels_zero_based = labels_zero_based,
)
```

## Example 5: Pack Sequences Into Fixed-Length Blocks

This recipe concatenates sequence ids, optionally inserts a separator id between sequences, then chunks into fixed-length blocks.

```@example structured_outputs_batching
function pack_blocks(
    results::Vector{TokenizationResult};
    block_size::Int,
    pad_id::Int,
    eos_id::Union{Nothing,Int}=nothing,
)
    block_size > 0 || throw(ArgumentError("block_size must be > 0"))

    flat_ids = Int[]
    for (i, result) in pairs(results)
        append!(flat_ids, result.ids)
        if eos_id !== nothing && i < length(results)
            push!(flat_ids, eos_id)
        end
    end

    n_blocks = max(1, cld(length(flat_ids), block_size))
    ids_blocks = fill(pad_id, block_size, n_blocks)
    attention_mask_blocks = fill(0, block_size, n_blocks)

    for (flat_index, token_id) in pairs(flat_ids)
        block_index = cld(flat_index, block_size)
        row_index = flat_index - (block_index - 1) * block_size
        ids_blocks[row_index, block_index] = token_id
        attention_mask_blocks[row_index, block_index] = 1
    end

    return (
        ids_blocks = ids_blocks,
        attention_mask_blocks = attention_mask_blocks,
        total_tokens = length(flat_ids),
    )
end

separator_id = get(special_tokens(tokenizer), :sep, nothing)
packed = pack_blocks(
    batch_results;
    block_size=6,
    pad_id=pad_id(tokenizer),
    eos_id=separator_id,
)

(
    separator_id = separator_id,
    ids_blocks_size = size(packed.ids_blocks),
    attention_mask_blocks_size = size(packed.attention_mask_blocks),
    ids_blocks = packed.ids_blocks,
    attention_mask_blocks = packed.attention_mask_blocks,
)
```

For span-to-token label alignment (for example NER-style supervision), see [Offsets Alignment Examples](offset_alignment_examples.md).
