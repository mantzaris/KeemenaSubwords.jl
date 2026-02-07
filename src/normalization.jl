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
