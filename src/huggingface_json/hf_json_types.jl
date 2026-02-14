abstract type HFJSONModelSpec end

struct HFBPEModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    merges::Vector{Tuple{String,String}}
    unk_token::String
    byte_level::Bool
    continuing_subword_prefix::Union{Nothing,String}
    end_of_word_suffix::Union{Nothing,String}
    fuse_unk::Bool
    byte_fallback::Bool
    dropout::Union{Nothing,Float64}
end

struct HFWordPieceModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    unk_token::String
    continuation_prefix::String
    max_input_chars_per_word::Int
end

struct HFUnigramModelSpec <: HFJSONModelSpec
    vocab::Vector{String}
    scores::Vector{Float64}
    unk_id::Int
    byte_fallback::Bool
end

abstract type HFJSONNormalizer end
struct HFNoopNormalizer <: HFJSONNormalizer end
struct HFLowercaseNormalizer <: HFJSONNormalizer end
struct HFNFCNormalizer <: HFJSONNormalizer end
struct HFNFKCNormalizer <: HFJSONNormalizer end
struct HFNFDNormalizer <: HFJSONNormalizer end
struct HFStripAccentsNormalizer <: HFJSONNormalizer end
struct HFReplaceNormalizer <: HFJSONNormalizer
    pattern::Regex
    replacement::String
end
struct HFPrependNormalizer <: HFJSONNormalizer
    prefix::String
end
struct HFSequenceNormalizer <: HFJSONNormalizer
    items::Vector{HFJSONNormalizer}
end
struct HFBertNormalizer <: HFJSONNormalizer
    clean_text::Bool
    handle_chinese_chars::Bool
    strip_accents::Union{Nothing,Bool}
    lowercase::Bool
end

effective_strip_accents(normalizer::HFBertNormalizer)::Bool =
    normalizer.strip_accents === nothing ? normalizer.lowercase : normalizer.strip_accents

abstract type HFJSONPreTokenizer end
struct HFNoopPreTokenizer <: HFJSONPreTokenizer end
struct HFBertPreTokenizer <: HFJSONPreTokenizer end
struct HFByteLevelPreTokenizer <: HFJSONPreTokenizer
    add_prefix_space::Bool
    trim_offsets::Bool
    use_regex::Bool
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
struct HFDigitsPreTokenizer <: HFJSONPreTokenizer
    individual_digits::Bool
end
struct HFPunctuationPreTokenizer <: HFJSONPreTokenizer
    behavior::Symbol
end
struct HFSequencePreTokenizer <: HFJSONPreTokenizer
    items::Vector{HFJSONPreTokenizer}
end

abstract type HFJSONPostProcessor end
struct HFNoopPostProcessor <: HFJSONPostProcessor end
struct HFByteLevelPostProcessor <: HFJSONPostProcessor
    add_prefix_space::Bool
    trim_offsets::Bool
end
struct HFTemplateProcessingPostProcessor <: HFJSONPostProcessor
    single::Vector{String}
    pair::Vector{String}
    special_tokens::Dict{String,Int}
end
struct HFBertProcessingPostProcessor <: HFJSONPostProcessor
    cls_token::String
    cls_id::Int
    sep_token::String
    sep_id::Int
end
struct HFRobertaProcessingPostProcessor <: HFJSONPostProcessor
    cls_token::String
    cls_id::Int
    sep_token::String
    sep_id::Int
end
struct HFSequencePostProcessor <: HFJSONPostProcessor
    items::Vector{HFJSONPostProcessor}
end

abstract type HFJSONDecoder end
struct HFNoopDecoder <: HFJSONDecoder end
struct HFByteLevelDecoder <: HFJSONDecoder
    add_prefix_space::Bool
    trim_offsets::Bool
    use_regex::Bool
end
struct HFWordPieceDecoder <: HFJSONDecoder
    prefix::String
end
struct HFMetaspaceDecoder <: HFJSONDecoder
    replacement::String
    add_prefix_space::Bool
end
struct HFBPEDecoder <: HFJSONDecoder
    suffix::String
end
struct HFSequenceDecoder <: HFJSONDecoder
    items::Vector{HFJSONDecoder}
end

struct HFAddedToken
    content::String
    id::Int
    special::Bool
    single_word::Bool
    lstrip::Bool
    rstrip::Bool
    normalized::Bool
end

struct HFAddedTokenPattern
    content::String
    id::Int
    single_word::Bool
    lstrip::Bool
    rstrip::Bool
end

struct HFJSONSpec
    model::HFJSONModelSpec
    normalizer::HFJSONNormalizer
    pretokenizer::HFJSONPreTokenizer
    postprocessor::HFJSONPostProcessor
    decoder::HFJSONDecoder
    added_tokens::Vector{HFAddedToken}
    added_token_ids::Dict{String,Int}
    special_token_ids::Dict{String,Int}
    truncation::Union{Nothing,NamedTuple}
    padding::Union{Nothing,NamedTuple}
    source_path::String
end

struct HuggingFaceJSONTokenizer <: AbstractSubwordTokenizer
    model::HFJSONModelSpec
    base::AbstractSubwordTokenizer
    normalizer::HFJSONNormalizer
    pretokenizer::HFJSONPreTokenizer
    postprocessor::HFJSONPostProcessor
    decoder::HFJSONDecoder
    added_tokens::Vector{HFAddedToken}
    special_added_patterns::Vector{HFAddedTokenPattern}
    raw_added_patterns::Vector{HFAddedTokenPattern}
    normalized_added_patterns::Vector{HFAddedTokenPattern}
    added_token_ids::Dict{String,Int}
    token_special_ids::Dict{String,Int}
    special_token_ids::Dict{Symbol,Int}
    truncation::Union{Nothing,NamedTuple}
    padding::Union{Nothing,NamedTuple}
    metadata::TokenizerMetadata
    source_path::String
end
