"""
    quick_tokenize(tokenizer_or_source, input_text; kwargs...) -> NamedTuple

High-level one-call wrapper for common single-text tokenization workflows.

This helper applies the recommended offsets pipeline by default:
1. `tokenization_text = tokenization_view(tokenizer, input_text)`
2. `encode_result(tokenizer, tokenization_text; assume_normalized=true, ...)`

Supported inputs:
- `quick_tokenize(tokenizer::AbstractSubwordTokenizer, input_text; ...)`
- `quick_tokenize(source::Symbol, input_text; format=nothing, prefetch=true, ...)`
- `quick_tokenize(source::AbstractString, input_text; format=nothing, prefetch=true, ...)`

Keyword arguments:
- `add_special_tokens::Bool=true`
- `apply_tokenization_view::Bool=true`
- `return_offsets::Bool=true`
- `return_masks::Bool=true`

Returns a `NamedTuple` with keys:
- `token_pieces`
- `token_ids`
- `decoded_text`
- `tokenization_text`
- `offsets`
- `attention_mask`
- `token_type_ids`
- `special_tokens_mask`
- `metadata`
"""
function quick_tokenize(
    tokenizer::AbstractSubwordTokenizer,
    input_text::AbstractString;
    add_special_tokens::Bool=true,
    apply_tokenization_view::Bool=true,
    return_offsets::Bool=true,
    return_masks::Bool=true,
)::NamedTuple
    tokenization_text = _prepare_tokenization_text(
        tokenizer,
        input_text;
        apply_tokenization_view=apply_tokenization_view,
    )

    result = encode_result(
        tokenizer,
        tokenization_text;
        add_special_tokens=add_special_tokens,
        assume_normalized=true,
        return_offsets=return_offsets,
        return_masks=return_masks,
    )

    return (
        token_pieces=result.tokens,
        token_ids=result.ids,
        decoded_text=decode(tokenizer, result.ids),
        tokenization_text=tokenization_text,
        offsets=result.offsets,
        attention_mask=result.attention_mask,
        token_type_ids=result.token_type_ids,
        special_tokens_mask=result.special_tokens_mask,
        metadata=result.metadata,
    )
end

function quick_tokenize(
    source::Symbol,
    input_text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_tokenize(tokenizer, input_text; kwargs...)
end

function quick_tokenize(
    source::AbstractString,
    input_text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_tokenize(tokenizer, input_text; kwargs...)
end

"""
    quick_encode_batch(tokenizer_or_source, input_texts; kwargs...) -> NamedTuple

High-level wrapper for batch structured encoding.

By default, each input text is first converted with `tokenization_view` so offsets
and alignment metadata are anchored to tokenizer-coordinate text.

Keyword arguments:
- `add_special_tokens::Bool=true`
- `apply_tokenization_view::Bool=true`
- `return_offsets::Bool=true`
- `return_masks::Bool=true`
- `format::Union{Nothing,Symbol}=nothing` (source overloads only)
- `prefetch::Bool=true` (source overloads only)

Returns a `NamedTuple` with keys:
- `tokenization_texts`
- `results`
- `sequence_lengths`
- `metadata`
"""
function quick_encode_batch(
    tokenizer::AbstractSubwordTokenizer,
    input_texts::AbstractVector{<:AbstractString};
    add_special_tokens::Bool=true,
    apply_tokenization_view::Bool=true,
    return_offsets::Bool=true,
    return_masks::Bool=true,
)::NamedTuple
    tokenization_texts = _prepare_tokenization_texts(
        tokenizer,
        input_texts;
        apply_tokenization_view=apply_tokenization_view,
    )

    results = encode_batch_result(
        tokenizer,
        tokenization_texts;
        add_special_tokens=add_special_tokens,
        assume_normalized=true,
        return_offsets=return_offsets,
        return_masks=return_masks,
    )
    sequence_lengths = [length(result.ids) for result in results]
    metadata = [result.metadata for result in results]

    return (
        tokenization_texts=tokenization_texts,
        results=results,
        sequence_lengths=sequence_lengths,
        metadata=metadata,
    )
end

function quick_encode_batch(
    source::Symbol,
    input_texts::AbstractVector{<:AbstractString};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_encode_batch(tokenizer, input_texts; kwargs...)
end

function quick_encode_batch(
    source::AbstractString,
    input_texts::AbstractVector{<:AbstractString};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_encode_batch(tokenizer, input_texts; kwargs...)
end

"""
    collate_padded_batch(results; tokenizer=nothing, pad_token_id=nothing, kwargs...) -> NamedTuple

Collate `Vector{TokenizationResult}` into dense `(sequence_length, batch_size)` matrices.

Returns:
- `ids::Matrix{Int}`
- `attention_mask::Matrix{Int}`
- `token_type_ids::Matrix{Int}`
- `special_tokens_mask::Matrix{Int}`
- `sequence_lengths::Vector{Int}`
- `pad_token_id::Int`
- `pad_side::Symbol`

Padding behavior:
- `ids` are filled with `pad_token_id`.
- `attention_mask` uses `1` for valid tokens and `0` for padding.
- `token_type_ids` defaults to `0` where missing.
- `special_tokens_mask` defaults to `0` on valid tokens where missing and uses
  `1` on padding positions.

Pad token selection:
- If `pad_token_id` is provided, it is used directly.
- Otherwise `pad_id(tokenizer)` is used when available.
- Otherwise `eos_id(tokenizer)` is used when available.
- If none are available, throws an `ArgumentError`.

Optional keyword arguments:
- `pad_to_multiple_of::Union{Nothing,Int}=nothing`
- `pad_side::Symbol=:right` (only right padding is currently supported)
"""
function collate_padded_batch(
    results::Vector{TokenizationResult};
    tokenizer::Union{Nothing,AbstractSubwordTokenizer}=nothing,
    pad_token_id::Union{Nothing,Int}=nothing,
    pad_to_multiple_of::Union{Nothing,Int}=nothing,
    pad_side::Symbol=:right,
)::NamedTuple
    isempty(results) && throw(ArgumentError("collate_padded_batch requires at least one sequence"))
    pad_side == :right || throw(ArgumentError("collate_padded_batch currently supports only pad_side=:right"))

    if pad_to_multiple_of !== nothing
        pad_to_multiple_of > 0 || throw(ArgumentError("pad_to_multiple_of must be > 0"))
    end

    resolved_pad_token_id = _resolve_pad_token_id(tokenizer, pad_token_id)

    sequence_lengths = [length(result.ids) for result in results]
    max_sequence_length = maximum(sequence_lengths)
    target_sequence_length = if pad_to_multiple_of === nothing
        max_sequence_length
    else
        cld(max_sequence_length, pad_to_multiple_of) * pad_to_multiple_of
    end

    batch_size = length(results)
    ids = fill(resolved_pad_token_id, target_sequence_length, batch_size)
    attention_mask = fill(0, target_sequence_length, batch_size)
    token_type_ids = fill(0, target_sequence_length, batch_size)
    special_tokens_mask = fill(1, target_sequence_length, batch_size)

    for (column_index, result) in pairs(results)
        sequence_length = length(result.ids)
        ids[1:sequence_length, column_index] = result.ids
        attention_mask[1:sequence_length, column_index] .= 1

        if result.token_type_ids !== nothing
            length(result.token_type_ids) == sequence_length || throw(ArgumentError(
                "token_type_ids length mismatch at sequence $column_index",
            ))
            token_type_ids[1:sequence_length, column_index] = result.token_type_ids
        end

        if result.special_tokens_mask === nothing
            special_tokens_mask[1:sequence_length, column_index] .= 0
        else
            length(result.special_tokens_mask) == sequence_length || throw(ArgumentError(
                "special_tokens_mask length mismatch at sequence $column_index",
            ))
            special_tokens_mask[1:sequence_length, column_index] = result.special_tokens_mask
        end
    end

    return (
        ids=ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        special_tokens_mask=special_tokens_mask,
        sequence_lengths=sequence_lengths,
        pad_token_id=resolved_pad_token_id,
        pad_side=pad_side,
    )
end

function collate_padded_batch(
    tokenizer::AbstractSubwordTokenizer,
    results::Vector{TokenizationResult};
    kwargs...,
)::NamedTuple
    return collate_padded_batch(results; tokenizer=tokenizer, kwargs...)
end

"""
    causal_lm_labels(ids, attention_mask; ignore_index=-100, zero_based=false) -> Matrix{Int}

Build next-token labels for causal language modeling from padded `ids` and
`attention_mask` matrices shaped `(sequence_length, batch_size)`.

For each sequence column:
- valid non-final positions receive the next valid token id,
- the final valid position receives `ignore_index`,
- padding positions receive `ignore_index`.

When `zero_based=true`, subtracts `1` from all non-ignored labels to support
consumers that expect 0-based ids.
"""
function causal_lm_labels(
    ids::Matrix{Int},
    attention_mask::Matrix{Int};
    ignore_index::Int=-100,
    zero_based::Bool=false,
)::Matrix{Int}
    size(ids) == size(attention_mask) || throw(ArgumentError(
        "ids and attention_mask must have the same shape",
    ))

    sequence_length, batch_size = size(ids)
    labels = fill(ignore_index, sequence_length, batch_size)

    for column_index in 1:batch_size
        valid_positions = findall(!iszero, view(attention_mask, :, column_index))
        n_valid = length(valid_positions)
        n_valid <= 1 && continue

        for i in 1:(n_valid - 1)
            current_position = valid_positions[i]
            next_position = valid_positions[i + 1]
            labels[current_position, column_index] = ids[next_position, column_index]
        end
    end

    if zero_based
        for index in eachindex(labels)
            labels[index] == ignore_index && continue
            labels[index] -= 1
        end
    end

    return labels
end

"""
    quick_causal_lm_batch(tokenizer_or_source, input_texts; kwargs...) -> NamedTuple

One-call helper for training-ready causal LM tensors.

Pipeline:
1. `quick_encode_batch(...; return_masks=true)`
2. `collate_padded_batch(...)`
3. `causal_lm_labels(...)`

Keyword arguments:
- `add_special_tokens::Bool=true`
- `apply_tokenization_view::Bool=true`
- `return_offsets::Bool=false`
- `pad_token_id::Union{Nothing,Int}=nothing`
- `pad_to_multiple_of::Union{Nothing,Int}=nothing`
- `pad_side::Symbol=:right`
- `ignore_index::Int=-100`
- `zero_based::Bool=false`
- `format::Union{Nothing,Symbol}=nothing` (source overloads only)
- `prefetch::Bool=true` (source overloads only)

Returns a `NamedTuple` with keys:
- `ids`
- `attention_mask`
- `labels`
- `token_type_ids`
- `special_tokens_mask`
- `tokenization_texts`
- `sequence_lengths`
- `pad_token_id`
- `ignore_index`
- `zero_based`
"""
function quick_causal_lm_batch(
    tokenizer::AbstractSubwordTokenizer,
    input_texts::AbstractVector{<:AbstractString};
    add_special_tokens::Bool=true,
    apply_tokenization_view::Bool=true,
    return_offsets::Bool=false,
    pad_token_id::Union{Nothing,Int}=nothing,
    pad_to_multiple_of::Union{Nothing,Int}=nothing,
    pad_side::Symbol=:right,
    ignore_index::Int=-100,
    zero_based::Bool=false,
)::NamedTuple
    isempty(input_texts) && throw(ArgumentError("quick_causal_lm_batch requires at least one input text"))

    batch_encoding = quick_encode_batch(
        tokenizer,
        input_texts;
        add_special_tokens=add_special_tokens,
        apply_tokenization_view=apply_tokenization_view,
        return_offsets=return_offsets,
        return_masks=true,
    )

    collated = collate_padded_batch(
        batch_encoding.results;
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_side=pad_side,
    )

    labels = causal_lm_labels(
        collated.ids,
        collated.attention_mask;
        ignore_index=ignore_index,
        zero_based=zero_based,
    )

    return (
        ids=collated.ids,
        attention_mask=collated.attention_mask,
        labels=labels,
        token_type_ids=collated.token_type_ids,
        special_tokens_mask=collated.special_tokens_mask,
        tokenization_texts=batch_encoding.tokenization_texts,
        sequence_lengths=batch_encoding.sequence_lengths,
        pad_token_id=collated.pad_token_id,
        ignore_index=ignore_index,
        zero_based=zero_based,
    )
end

function quick_causal_lm_batch(
    source::Symbol,
    input_texts::AbstractVector{<:AbstractString};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_causal_lm_batch(tokenizer, input_texts; kwargs...)
end

function quick_causal_lm_batch(
    source::AbstractString,
    input_texts::AbstractVector{<:AbstractString};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::NamedTuple
    tokenizer = _resolve_quick_tokenizer(source; format=format, prefetch=prefetch)
    return quick_causal_lm_batch(tokenizer, input_texts; kwargs...)
end

"""
    quick_train_bundle(trainer, corpus; kwargs...) -> NamedTuple

High-level training round-trip helper:
1. Train with a selected trainer.
2. Save a training bundle with `save_training_bundle`.
3. Reload with `load_training_bundle`.
4. Run a sanity encode/decode pass.

Supported trainer symbols:
- `:wordpiece`
- `:hf_bert_wordpiece`
- `:hf_roberta_bytebpe`
- `:hf_gpt2_bytebpe`

Convenience overload:
- `quick_train_bundle(corpus; kwargs...)` defaults to `trainer=:wordpiece`.

Keyword arguments:
- `bundle_directory::Union{Nothing,AbstractString}=nothing`
- `overwrite::Bool=true`
- `export_format::Symbol=:auto`
- `sanity_text::AbstractString="hello world"`
- plus trainer-specific keywords forwarded to the selected `train_*_result`.

Returns a `NamedTuple` with keys:
- `bundle_directory`
- `bundle_files`
- `training_summary`
- `tokenizer`
- `sanity_encoded_ids`
- `sanity_decoded_text`
"""
function quick_train_bundle(
    trainer::Symbol,
    corpus;
    bundle_directory::Union{Nothing,AbstractString}=nothing,
    overwrite::Bool=true,
    export_format::Symbol=:auto,
    sanity_text::AbstractString="hello world",
    kwargs...,
)::NamedTuple
    training_result = _run_quick_trainer(trainer, corpus; kwargs...)

    target_bundle_directory = bundle_directory === nothing ?
        mktempdir() :
        normpath(String(bundle_directory))

    save_training_bundle(
        training_result,
        target_bundle_directory;
        overwrite=overwrite,
        export_format=export_format,
    )

    tokenizer = load_training_bundle(target_bundle_directory)
    sanity_encoded_ids = encode(tokenizer, String(sanity_text); add_special_tokens=false)
    sanity_decoded_text = decode(tokenizer, sanity_encoded_ids)

    training_summary = (
        trainer=trainer,
        tokenizer_type=string(typeof(training_result.tokenizer)),
        config_type=string(typeof(training_result.config)),
        model_name=training_result.config.model_name,
        version=string(training_result.config.version),
        vocab_size=vocab_size(training_result.tokenizer),
    )

    return (
        bundle_directory=target_bundle_directory,
        bundle_files=sort(readdir(target_bundle_directory)),
        training_summary=training_summary,
        tokenizer=tokenizer,
        sanity_encoded_ids=sanity_encoded_ids,
        sanity_decoded_text=sanity_decoded_text,
    )
end

function quick_train_bundle(
    corpus::AbstractVector{<:AbstractString};
    kwargs...,
)::NamedTuple
    return quick_train_bundle(:wordpiece, corpus; kwargs...)
end

function _resolve_quick_tokenizer(
    source::Symbol;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::AbstractSubwordTokenizer
    return get_tokenizer_cached(source; format=format, prefetch=prefetch)
end

function _resolve_quick_tokenizer(
    source::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::AbstractSubwordTokenizer
    return get_tokenizer_cached(source; format=format, prefetch=prefetch)
end

function _prepare_tokenization_text(
    tokenizer::AbstractSubwordTokenizer,
    input_text::AbstractString;
    apply_tokenization_view::Bool,
)::String
    raw_text = String(input_text)
    return apply_tokenization_view ? tokenization_view(tokenizer, raw_text) : raw_text
end

function _prepare_tokenization_texts(
    tokenizer::AbstractSubwordTokenizer,
    input_texts::AbstractVector{<:AbstractString};
    apply_tokenization_view::Bool,
)::Vector{String}
    return [
        _prepare_tokenization_text(
            tokenizer,
            input_text;
            apply_tokenization_view=apply_tokenization_view,
        )
        for input_text in input_texts
    ]
end

function _resolve_pad_token_id(
    tokenizer::Union{Nothing,AbstractSubwordTokenizer},
    pad_token_id::Union{Nothing,Int},
)::Int
    pad_token_id !== nothing && return pad_token_id

    tokenizer === nothing && throw(ArgumentError(
        "pad_token_id was not provided and tokenizer is missing. Provide pad_token_id or tokenizer.",
    ))

    tokenizer_pad_id = pad_id(tokenizer)
    tokenizer_pad_id !== nothing && return tokenizer_pad_id

    tokenizer_eos_id = eos_id(tokenizer)
    tokenizer_eos_id !== nothing && return tokenizer_eos_id

    throw(ArgumentError(
        "Could not infer pad token id from tokenizer. Provide pad_token_id explicitly.",
    ))
end

function _run_quick_trainer(
    trainer::Symbol,
    corpus;
    kwargs...,
)
    if trainer == :wordpiece
        return train_wordpiece_result(corpus; kwargs...)
    elseif trainer == :hf_bert_wordpiece
        return train_hf_bert_wordpiece_result(corpus; kwargs...)
    elseif trainer == :hf_roberta_bytebpe
        return train_hf_roberta_bytebpe_result(corpus; kwargs...)
    elseif trainer == :hf_gpt2_bytebpe
        return train_hf_gpt2_bytebpe_result(corpus; kwargs...)
    end

    throw(ArgumentError(
        "Unsupported trainer=$trainer. Supported trainers are :wordpiece, :hf_bert_wordpiece, :hf_roberta_bytebpe, :hf_gpt2_bytebpe.",
    ))
end
