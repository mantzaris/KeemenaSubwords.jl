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
