const _SPECIAL_TOKEN_KEY_ALIASES = Dict{Symbol,Symbol}(
    :unknown => :unk,
    :padding => :pad,
    :begin => :bos,
    :end => :eos,
)

const _SPECIAL_TOKEN_ORDER = Dict{Symbol,Int}(
    :unk => 1,
    :pad => 2,
    :bos => 3,
    :eos => 4,
    :cls => 5,
    :sep => 6,
    :mask => 7,
)

function _canonical_special_token_key(symbol::Symbol)::Symbol
    lowered = Symbol(lowercase(String(symbol)))
    return get(_SPECIAL_TOKEN_KEY_ALIASES, lowered, lowered)
end

function _normalize_special_tokens(
    special_tokens::Dict{Symbol,String},
)::Dict{Symbol,String}
    normalized = Dict{Symbol,String}()
    entries = collect(special_tokens)
    sort!(entries; by = p -> String(p.first))

    for (symbol, token) in entries
        canonical = _canonical_special_token_key(symbol)
        token_string = String(token)
        existing = get(normalized, canonical, nothing)
        if existing !== nothing && existing != token_string
            throw(ArgumentError(
                "Conflicting values for special token key :$canonical " *
                "(\"$existing\" vs \"$token_string\")",
            ))
        end
        normalized[canonical] = token_string
    end

    return normalized
end

function _ordered_special_token_pairs(
    special_tokens::Dict{Symbol,String},
)::Vector{Pair{Symbol,String}}
    pairs = collect(special_tokens)
    sort!(pairs; by = p -> (get(_SPECIAL_TOKEN_ORDER, p.first, typemax(Int)), String(p.first)))
    return pairs
end

function _ordered_special_token_values(
    special_tokens::Dict{Symbol,String},
)::Vector{String}
    ordered = String[]
    for pair in _ordered_special_token_pairs(special_tokens)
        token = pair.second
        token in ordered || push!(ordered, token)
    end
    return ordered
end

function _collect_word_counts(
    corpus;
    pretokenizer::Union{Nothing,Function}=nothing,
)::Dict{String,Int}
    counts = Dict{String,Int}()

    for doc in corpus
        text = String(doc)
        pieces = pretokenizer === nothing ? eachsplit(text) : pretokenizer(text)
        pieces === nothing && continue

        for piece in pieces
            token = String(piece)
            isempty(token) && continue
            counts[token] = get(counts, token, 0) + 1
        end
    end

    return counts
end

function _validate_positive(value::Int, field_name::AbstractString)::Nothing
    value > 0 || throw(ArgumentError("$field_name must be positive"))
    return nothing
end

function _validate_nonempty(value::AbstractString, field_name::AbstractString)::Nothing
    isempty(value) && throw(ArgumentError("$field_name must be non-empty"))
    return nothing
end

function _validate_bpe_config(config::BPETrainingConfig)::Nothing
    _validate_positive(config.vocab_size, "vocab_size")
    _validate_positive(config.min_frequency, "min_frequency")
    _validate_nonempty(config.end_of_word_marker, "end_of_word_marker")
    _validate_nonempty(config.model_name, "model_name")
    haskey(config.special_tokens, :unk) || throw(ArgumentError("special_tokens must include :unk"))
    return nothing
end

function _validate_bytebpe_config(config::ByteBPETrainingConfig)::Nothing
    _validate_positive(config.vocab_size, "vocab_size")
    _validate_positive(config.min_frequency, "min_frequency")
    _validate_nonempty(config.end_of_word_marker, "end_of_word_marker")
    _validate_nonempty(config.model_name, "model_name")
    haskey(config.special_tokens, :unk) || throw(ArgumentError("special_tokens must include :unk"))
    return nothing
end

function _validate_required_vocab_capacity(
    vocab_size::Int,
    required_tokens::Vector{String},
)::Nothing
    min_required = length(required_tokens)
    vocab_size >= min_required || throw(ArgumentError(
        "vocab_size=$vocab_size is too small: need at least $min_required to keep required special/marker tokens",
    ))
    return nothing
end

function _push_unique_token!(
    tokens::Vector{String},
    token_set::Set{String},
    token::String,
)::Bool
    token in token_set && return false
    push!(tokens, token)
    push!(token_set, token)
    return true
end
