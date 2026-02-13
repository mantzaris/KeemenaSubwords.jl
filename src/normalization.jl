"""
Normalize text using an optional user-provided callable.
"""
function normalize_text(
    text::AbstractString;
    normalizer::Union{Nothing,Function}=nothing,
)::String
    if normalizer === nothing
        return String(text)
    end
    normalized = normalizer(text)
    normalized isa AbstractString || throw(ArgumentError("Normalizer must return an AbstractString"))
    return String(normalized)
end

"""
Return tokenizer intrinsic normalization output.

This does not perform pipeline-level preprocessing. Tokenizers without intrinsic
normalization return `text` unchanged.
"""
function normalize(
    tokenizer::AbstractSubwordTokenizer,
    text::AbstractString,
)::String
    _ = tokenizer
    return String(text)
end

"""
Canonical tokenizer text view used for subword offsets/alignment.
"""
tokenization_view(tokenizer::AbstractSubwordTokenizer, clean_text::AbstractString)::String =
    normalize(tokenizer, clean_text)

"""
Whether this tokenizer defines intrinsic normalization that can change text.
"""
requires_tokenizer_normalization(tokenizer::AbstractSubwordTokenizer)::Bool = false

"""
Offset coordinate system for `TokenizationResult.offsets`.

Offsets are UTF-8 codeunit indices with half-open span convention `[start, stop)`.
"""
offsets_coordinate_system()::Symbol = :utf8_codeunits

"""
Offset index base for `TokenizationResult.offsets`.

Offsets are 1-based codeunit indices.
"""
offsets_index_base()::Int = 1

"""
Offset span style.

`TokenizationResult.offsets` use half-open spans `[start, stop)`.
"""
offsets_span_style()::Symbol = :half_open

"""
Sentinel used for tokens without a source-text span.
"""
offsets_sentinel()::Tuple{Int,Int} = (0, 0)

"""
Return `true` when an offset carries a real source-text span.
"""
has_span(offset::Tuple{Int,Int})::Bool = offset != offsets_sentinel()

"""
Return `true` when an offset carries a non-empty source-text span.
"""
has_nonempty_span(offset::Tuple{Int,Int})::Bool = has_span(offset) && offset[2] > offset[1]

"""
Return span length measured in UTF-8 codeunits.

Sentinel and empty spans return `0`.
"""
function span_ncodeunits(offset::Tuple{Int,Int})::Int
    has_span(offset) || return 0
    start_idx, stop_idx = offset
    return max(0, stop_idx - start_idx)
end

"""
Return the offset span as UTF-8 codeunits.

Sentinel and empty spans return `UInt8[]`. Invalid or out-of-bounds spans also
return `UInt8[]` to keep this helper non-throwing for downstream inspection.
"""
function span_codeunits(
    text::AbstractString,
    offset::Tuple{Int,Int},
)::Vector{UInt8}
    has_nonempty_span(offset) || return UInt8[]
    start_idx, stop_idx = offset
    max_stop = ncodeunits(text) + offsets_index_base()
    if start_idx < offsets_index_base() || stop_idx > max_stop || stop_idx <= start_idx
        return UInt8[]
    end
    return Vector{UInt8}(codeunits(text)[start_idx:(stop_idx - 1)])
end

"""
Return whether `idx` is a valid Julia string boundary for `text`.

This includes the exclusive end boundary `ncodeunits(text) + 1`.
"""
function is_valid_string_boundary(
    text::AbstractString,
    idx::Int,
)::Bool
    idx >= offsets_index_base() || return false
    end_idx = ncodeunits(text) + offsets_index_base()
    idx <= end_idx || return false
    return idx == end_idx || isvalid(text, idx)
end

"""
Attempt to return a substring for a half-open codeunit span `[start, stop)`.

Sentinel and empty spans return `""`. If span boundaries are not valid Julia
string boundaries, this returns `nothing`. This helper never throws.
"""
function try_span_substring(
    text::AbstractString,
    offset::Tuple{Int,Int},
)::Union{Nothing,String}
    try
        has_span(offset) || return ""
        start_idx, stop_idx = offset
        stop_idx > start_idx || return ""
        is_valid_string_boundary(text, start_idx) || return nothing
        is_valid_string_boundary(text, stop_idx) || return nothing
        return String(SubString(text, start_idx, prevind(text, stop_idx)))
    catch
        return nothing
    end
end

"""
Return whether participating offsets are non-overlapping in sequence order.

Participating offsets satisfy:
- not sentinel when `ignore_sentinel=true`
- not empty when `ignore_empty=true`

For participating offsets, this enforces `next.start >= prev.stop`.
"""
function offsets_are_nonoverlapping(
    offsets::Vector{Tuple{Int,Int}};
    ignore_sentinel::Bool=true,
    ignore_empty::Bool=true,
)::Bool
    prev_stop = nothing
    for offset in offsets
        if !has_span(offset)
            ignore_sentinel && continue
            return false
        end

        start_idx, stop_idx = offset
        stop_idx >= start_idx || return false
        if stop_idx == start_idx && ignore_empty
            continue
        end

        if prev_stop !== nothing && start_idx < prev_stop
            return false
        end
        prev_stop = stop_idx
    end
    return true
end

function _offsets_contract_failure(
    text::AbstractString,
    offsets::Vector{Tuple{Int,Int}};
    require_string_boundaries::Bool,
)::Union{Nothing,String}
    base = offsets_index_base()
    max_stop = ncodeunits(text) + base
    sentinel = offsets_sentinel()

    for (idx, offset) in enumerate(offsets)
        start_idx, stop_idx = offset

        if offset == sentinel
            continue
        end

        if start_idx == 0 || stop_idx == 0
            return "offset[$idx]=$offset is invalid; only sentinel $(sentinel) may use zero indices"
        end
        if start_idx < base || stop_idx < base
            return "offset[$idx]=$offset is below index base $base"
        end
        if stop_idx < start_idx
            return "offset[$idx]=$offset has stop < start"
        end
        if stop_idx > max_stop
            return "offset[$idx]=$offset exceeds max stop $max_stop for text with ncodeunits=$(ncodeunits(text))"
        end

        if require_string_boundaries && has_nonempty_span(offset)
            if !is_valid_string_boundary(text, start_idx)
                return "offset[$idx]=$offset has non-boundary start index $start_idx"
            end
            if !is_valid_string_boundary(text, stop_idx)
                return "offset[$idx]=$offset has non-boundary stop index $stop_idx"
            end
        end
    end

    return nothing
end

"""
Validate offsets against the package offset contract.

Returns `true` when all offsets satisfy bounds/sentinel invariants. With
`require_string_boundaries=true`, non-empty spans must also start/end on valid
Julia string boundaries.
"""
function validate_offsets_contract(
    text::AbstractString,
    offsets::Vector{Tuple{Int,Int}};
    require_string_boundaries::Bool=false,
)::Bool
    failure = _offsets_contract_failure(
        text,
        offsets;
        require_string_boundaries=require_string_boundaries,
    )
    return failure === nothing
end

"""
Assert offsets satisfy the package offset contract.

Throws `ArgumentError` on first contract violation. With
`require_string_boundaries=true`, non-empty spans must start/end on valid Julia
string boundaries.
"""
function assert_offsets_contract(
    text::AbstractString,
    offsets::Vector{Tuple{Int,Int}};
    require_string_boundaries::Bool=false,
)::Nothing
    failure = _offsets_contract_failure(
        text,
        offsets;
        require_string_boundaries=require_string_boundaries,
    )
    failure === nothing || throw(ArgumentError("Offset contract violation: $failure"))
    return nothing
end
