abstract type HFJSONModelSpec end

struct HFBPEModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    merges::Vector{Tuple{String,String}}
    unk_token::String
    byte_level::Bool
    end_of_word_suffix::Union{Nothing,String}
end

struct HFWordPieceModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    unk_token::String
    continuation_prefix::String
end

struct HFUnigramModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    scores::Vector{Float64}
    unk_id::Int
end

abstract type HFJSONNormalizer end
struct HFNoopNormalizer <: HFJSONNormalizer end
struct HFLowercaseNormalizer <: HFJSONNormalizer end
struct HFNFCNormalizer <: HFJSONNormalizer end
struct HFNFKCNormalizer <: HFJSONNormalizer end
struct HFSequenceNormalizer <: HFJSONNormalizer
    items::Vector{HFJSONNormalizer}
end

abstract type HFJSONPreTokenizer end
struct HFNoopPreTokenizer <: HFJSONPreTokenizer end
struct HFByteLevelPreTokenizer <: HFJSONPreTokenizer
    add_prefix_space::Bool
end
struct HFWhitespacePreTokenizer <: HFJSONPreTokenizer end
struct HFMetaspacePreTokenizer <: HFJSONPreTokenizer
    replacement::String
    add_prefix_space::Bool
end
struct HFSplitPreTokenizer <: HFJSONPreTokenizer
    pattern::Regex
    behavior::Symbol
end
struct HFSequencePreTokenizer <: HFJSONPreTokenizer
    items::Vector{HFJSONPreTokenizer}
end

abstract type HFJSONPostProcessor end
struct HFNoopPostProcessor <: HFJSONPostProcessor end
struct HFByteLevelPostProcessor <: HFJSONPostProcessor end
struct HFTemplateProcessingPostProcessor <: HFJSONPostProcessor
    single::Vector{String}
    pair::Vector{String}
    special_tokens::Dict{String,Int}
end
struct HFSequencePostProcessor <: HFJSONPostProcessor
    items::Vector{HFJSONPostProcessor}
end

abstract type HFJSONDecoder end
struct HFNoopDecoder <: HFJSONDecoder end
struct HFByteLevelDecoder <: HFJSONDecoder end
struct HFWordPieceDecoder <: HFJSONDecoder
    prefix::String
end
struct HFMetaspaceDecoder <: HFJSONDecoder
    replacement::String
end
struct HFSequenceDecoder <: HFJSONDecoder
    items::Vector{HFJSONDecoder}
end

struct HFJSONSpec
    model::HFJSONModelSpec
    normalizer::HFJSONNormalizer
    pretokenizer::HFJSONPreTokenizer
    postprocessor::HFJSONPostProcessor
    decoder::HFJSONDecoder
    added_token_ids::Dict{String,Int}
    special_token_ids::Dict{String,Int}
    source_path::String
end

struct HuggingFaceJSONTokenizer <: AbstractSubwordTokenizer
    base::AbstractSubwordTokenizer
    normalizer::HFJSONNormalizer
    pretokenizer::HFJSONPreTokenizer
    postprocessor::HFJSONPostProcessor
    decoder::HFJSONDecoder
    added_token_ids::Dict{String,Int}
    token_special_ids::Dict{String,Int}
    special_token_ids::Dict{Symbol,Int}
    metadata::TokenizerMetadata
    source_path::String
end
