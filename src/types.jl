"""
Abstract parent type for all subword tokenizers.

Tokenizers are callable and support:
`tokenizer(text::AbstractString) -> Vector{String}`.
"""
abstract type AbstractSubwordTokenizer <: Function end

"""
Common metadata for tokenizer instances.
"""
struct TokenizerMetadata
    format::Symbol
    model_name::String
    version::VersionNumber
    normalizer::Symbol
end

"""
Convert `TokenizerMetadata` into a stable API-facing named tuple.
"""
metadata_namedtuple(metadata::TokenizerMetadata)::NamedTuple = (
    format = metadata.format,
    model_name = metadata.model_name,
    version = metadata.version,
    normalizer = metadata.normalizer,
)

"""
Return a function compatible with KeemenaPreprocessing's callable tokenizer contract.
"""
keemena_callable(tokenizer::AbstractSubwordTokenizer)::Function = tokenizer
keemena_callable(tokenizer::Function)::Function = tokenizer

function keemena_callable(tokenizer)::Function
    if hasmethod(tokenize, Tuple{typeof(tokenizer),AbstractString})
        return (text::AbstractString) -> tokenize(tokenizer, text)
    end
    throw(ArgumentError("Cannot produce a callable tokenizer from $(typeof(tokenizer))"))
end

"""
Level key used by KeemenaPreprocessing for callable tokenizers.
"""
level_key(tokenizer::AbstractSubwordTokenizer)::Symbol = Symbol(typeof(tokenizer))
level_key(tokenizer::Function)::Symbol = Symbol(typeof(tokenizer))

"""
Tokenize text into subword pieces.
"""
function tokenize(tokenizer::AbstractSubwordTokenizer, text::AbstractString)
    throw(MethodError(tokenize, (tokenizer, text)))
end

"""
Encode text into token IDs.
"""
function encode(tokenizer::AbstractSubwordTokenizer, text::AbstractString; add_special_tokens::Bool=false)
    throw(MethodError(encode, (tokenizer, text, add_special_tokens)))
end

"""
Decode token IDs into text.
"""
function decode(tokenizer::AbstractSubwordTokenizer, ids::AbstractVector{Int})
    throw(MethodError(decode, (tokenizer, ids)))
end

"""
Forward token lookup.
"""
function token_to_id(tokenizer::AbstractSubwordTokenizer, token::AbstractString)
    throw(MethodError(token_to_id, (tokenizer, token)))
end

"""
Reverse token lookup.
"""
function id_to_token(tokenizer::AbstractSubwordTokenizer, id::Int)
    throw(MethodError(id_to_token, (tokenizer, id)))
end

"""
Vocabulary size.
"""
function vocab_size(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(vocab_size, (tokenizer,)))
end

"""
Return model metadata.
"""
function model_info(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(model_info, (tokenizer,)))
end

"""
Return special token IDs keyed by symbol.
"""
function special_tokens(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(special_tokens, (tokenizer,)))
end

"""
Unknown token ID.
"""
function unk_id(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(unk_id, (tokenizer,)))
end

"""
Padding token ID if available.
"""
function pad_id(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(pad_id, (tokenizer,)))
end

"""
BOS token ID if available.
"""
function bos_id(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(bos_id, (tokenizer,)))
end

"""
EOS token ID if available.
"""
function eos_id(tokenizer::AbstractSubwordTokenizer)
    throw(MethodError(eos_id, (tokenizer,)))
end
