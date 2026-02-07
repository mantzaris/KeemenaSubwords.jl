"""
Vocabulary container with forward/reverse lookup and special token IDs.

IDs are 1-based.
"""
struct SubwordVocabulary
    id_to_token::Vector{String}
    token_to_id::Dict{String,Int}
    special_token_ids::Dict{Symbol,Int}

    function SubwordVocabulary(
        id_to_token::Vector{String},
        token_to_id::Dict{String,Int},
        special_token_ids::Dict{Symbol,Int}=Dict{Symbol,Int}(),
    )
        length(id_to_token) == length(token_to_id) || throw(ArgumentError("Vocabulary arrays are inconsistent"))
        new(id_to_token, token_to_id, special_token_ids)
    end
end

"""
Build a vocabulary from ordered tokens.
"""
function build_vocab(
    tokens::Vector{String};
    special_tokens::Dict{Symbol,String}=Dict{Symbol,String}(),
)::SubwordVocabulary
    token_to_id_map = Dict{String,Int}()
    for (i, token) in enumerate(tokens)
        haskey(token_to_id_map, token) && throw(ArgumentError("Duplicate token in vocabulary: $token"))
        token_to_id_map[token] = i
    end

    special_ids = Dict{Symbol,Int}()
    for (symbol, token) in special_tokens
        haskey(token_to_id_map, token) && (special_ids[symbol] = token_to_id_map[token])
    end

    return SubwordVocabulary(copy(tokens), token_to_id_map, special_ids)
end

"""
Infer common special token symbols from token strings.
"""
function detect_special_tokens(tokens::Vector{String}, unk_token::String)::Dict{Symbol,String}
    specials = Dict{Symbol,String}()

    if unk_token in tokens
        specials[:unk] = unk_token
    end

    for (sym, token) in (
        (:pad, "[PAD]"),
        (:cls, "[CLS]"),
        (:sep, "[SEP]"),
        (:bos, "<s>"),
        (:eos, "</s>"),
    )
        token in tokens && (specials[sym] = token)
    end

    for (sym, token) in (
        (:pad, "<pad>"),
        (:bos, "[BOS]"),
        (:eos, "[EOS]"),
    )
        token in tokens && !haskey(specials, sym) && (specials[sym] = token)
    end

    return specials
end

"""
Return unknown-token ID.
"""
function unk_id(vocab::SubwordVocabulary)::Int
    haskey(vocab.special_token_ids, :unk) || throw(ArgumentError("Vocabulary has no :unk special token"))
    return vocab.special_token_ids[:unk]
end

"""
Return padding-token ID if available.
"""
pad_id(vocab::SubwordVocabulary)::Union{Int,Nothing} = get(vocab.special_token_ids, :pad, nothing)

"""
Return beginning-of-sequence token ID if available.
"""
bos_id(vocab::SubwordVocabulary)::Union{Int,Nothing} = get(vocab.special_token_ids, :bos, nothing)

"""
Return end-of-sequence token ID if available.
"""
eos_id(vocab::SubwordVocabulary)::Union{Int,Nothing} = get(vocab.special_token_ids, :eos, nothing)

"""
Map token string to ID, falling back to :unk.
"""
function token_to_id(vocab::SubwordVocabulary, token::AbstractString)::Int
    token_string = String(token)
    if haskey(vocab.token_to_id, token_string)
        return vocab.token_to_id[token_string]
    end
    return unk_id(vocab)
end

"""
Map ID to token string.
"""
function id_to_token(vocab::SubwordVocabulary, id::Int)::String
    (1 <= id <= length(vocab.id_to_token)) || throw(BoundsError(vocab.id_to_token, id))
    return vocab.id_to_token[id]
end

"""
Vocabulary size.
"""
vocab_size(vocab::SubwordVocabulary)::Int = length(vocab.id_to_token)

"""
Whether an ID is a non-content special token that should be omitted in decode.
"""
function is_skippable_special_id(vocab::SubwordVocabulary, id::Int)::Bool
    for symbol in (:pad, :cls, :sep, :bos, :eos)
        if get(vocab.special_token_ids, symbol, 0) == id
            return true
        end
    end
    return false
end

"""
Read one token per line from a text file.
"""
function read_token_file(path::AbstractString)::Vector{String}
    tokens = String[]
    open(path, "r") do io
        for raw_line in eachline(io)
            token = strip(raw_line)
            isempty(token) && continue
            push!(tokens, token)
        end
    end
    isempty(tokens) && throw(ArgumentError("Token file is empty: $path"))
    return tokens
end
