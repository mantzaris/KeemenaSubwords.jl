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
Structured file specification for local tokenizer loading/registration.

Use `path` for single-file formats and explicit pairs for multi-file formats.
"""
struct FilesSpec
    format::Symbol
    path::Union{Nothing,String}
    vocab::Union{Nothing,String}
    merges::Union{Nothing,String}
    vocab_json::Union{Nothing,String}
    merges_txt::Union{Nothing,String}
    encoder_json::Union{Nothing,String}
    vocab_bpe::Union{Nothing,String}
    vocab_txt::Union{Nothing,String}
    unigram_tsv::Union{Nothing,String}
    tokenizer_json::Union{Nothing,String}
    model_file::Union{Nothing,String}
    encoding_file::Union{Nothing,String}
end

function FilesSpec(;
    format::Symbol,
    path=nothing,
    vocab=nothing,
    merges=nothing,
    vocab_json=nothing,
    merges_txt=nothing,
    encoder_json=nothing,
    vocab_bpe=nothing,
    vocab_txt=nothing,
    unigram_tsv=nothing,
    tokenizer_json=nothing,
    model_file=nothing,
    encoding_file=nothing,
)::FilesSpec
    _as_opt_string(x) = x === nothing ? nothing : String(x)
    return FilesSpec(
        format,
        _as_opt_string(path),
        _as_opt_string(vocab),
        _as_opt_string(merges),
        _as_opt_string(vocab_json),
        _as_opt_string(merges_txt),
        _as_opt_string(encoder_json),
        _as_opt_string(vocab_bpe),
        _as_opt_string(vocab_txt),
        _as_opt_string(unigram_tsv),
        _as_opt_string(tokenizer_json),
        _as_opt_string(model_file),
        _as_opt_string(encoding_file),
    )
end

"""
Structured tokenization output for downstream pipelines.

Offset contract:
- coordinate unit: UTF-8 codeunits.
- index base: 1.
- span style: half-open `[start, stop)`.
- valid bounds for spanful tokens: `1 <= start <= stop <= ncodeunits(text) + 1`.
- sentinel for tokens without source-text spans: `(0, 0)`.
- inserted post-processor specials use sentinel offsets.
- present-in-text special added tokens keep real spans, and may still have
  `special_tokens_mask[i] == 1`.
- `special_tokens_mask` marks special-token identity; `offsets` determine span
  participation.
"""
struct TokenizationResult
    ids::Vector{Int}
    tokens::Vector{String}
    offsets::Union{Nothing,Vector{Tuple{Int,Int}}}
    attention_mask::Union{Nothing,Vector{Int}}
    token_type_ids::Union{Nothing,Vector{Int}}
    special_tokens_mask::Union{Nothing,Vector{Int}}
    metadata::NamedTuple
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
Encode text and return a structured `TokenizationResult`.

Key keyword arguments:
- `assume_normalized::Bool=false`: when `true`, tokenizer intrinsic normalization
  is skipped and offsets are computed against the exact provided `text`.
- `return_offsets::Bool=false`: include token-level offsets when available.
- `return_masks::Bool=false`: include attention/token-type/special-token masks.

Offset note:
- Offsets use the package-wide 1-based UTF-8 codeunit half-open convention.
- `assume_normalized` changes whether intrinsic normalization runs; it does not
  change the offset coordinate system.
"""
function encode_result end

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
