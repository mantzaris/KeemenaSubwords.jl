abstract type AbstractTrainingConfig end
abstract type AbstractTrainingArtifacts end

struct TrainingResult{T,C<:AbstractTrainingConfig,A<:AbstractTrainingArtifacts}
    tokenizer::T
    config::C
    artifacts::A
end

struct BPETrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    pretokenizer::Union{Nothing,Function}
    end_of_word_marker::String
    model_name::String
    version::VersionNumber
end

struct BPETrainingArtifacts <: AbstractTrainingArtifacts
    vocab_tokens::Vector{String}
    merge_pairs::Vector{Tuple{String,String}}
    pair_ranks::Dict{Tuple{String,String},Int}
    word_counts::Dict{String,Int}
end

struct ByteBPETrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    end_of_word_marker::String
    pretokenizer::Union{Nothing,Function}
    include_full_byte_alphabet::Bool
    model_name::String
    version::VersionNumber
end

struct ByteBPETrainingArtifacts <: AbstractTrainingArtifacts
    vocab_tokens::Vector{String}
    merge_pairs::Vector{Tuple{String,String}}
    pair_ranks::Dict{Tuple{String,String},Int}
    word_counts::Dict{String,Int}
end

struct UnigramTrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    seed_size::Int
    num_iters::Int
    max_subword_length::Int
    prune_fraction::Float64
    special_tokens::Dict{Symbol,String}
    pretokenizer::Union{Nothing,Function}
    whitespace_marker::String
    model_name::String
    version::VersionNumber
end

struct UnigramTrainingArtifacts <: AbstractTrainingArtifacts
    vocab_tokens::Vector{String}
    token_logprobs::Vector{Float64}
    word_counts::Dict{String,Int}
end

struct WordPieceTrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    pretokenizer::Union{Nothing,Function}
    continuation_prefix::String
    max_input_chars_per_word::Int
    model_name::String
    version::VersionNumber
end

struct WordPieceTrainingArtifacts <: AbstractTrainingArtifacts
    vocab_tokens::Vector{String}
    merge_pairs::Vector{Tuple{String,String}}
    merge_scores::Vector{Float64}
    word_counts::Dict{String,Int}
end

struct SentencePieceTrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    model_type::Symbol
    min_frequency::Int
    seed_size::Int
    num_iters::Int
    max_subword_length::Int
    prune_fraction::Float64
    special_tokens::Dict{Symbol,String}
    pretokenizer::Union{Nothing,Function}
    whitespace_marker::String
    model_name::String
    version::VersionNumber
end

struct SentencePieceTrainingArtifacts <: AbstractTrainingArtifacts
    model_type::Symbol
    whitespace_marker::String
    inner_artifacts::Union{UnigramTrainingArtifacts,BPETrainingArtifacts}
end

struct BertWordPieceTrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    continuation_prefix::String
    max_input_chars_per_word::Int
    clean_text::Bool
    handle_chinese_chars::Bool
    lowercase::Bool
    strip_accents::Union{Nothing,Bool}
    model_name::String
    version::VersionNumber
end

struct BertWordPieceTrainingArtifacts <: AbstractTrainingArtifacts
    inner::WordPieceTrainingArtifacts
    hf_added_tokens::Vector{HFAddedToken}
end

struct RobertaByteBPETrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    end_of_word_marker::String
    add_prefix_space::Bool
    trim_offsets::Bool
    use_regex::Bool
    nfkc::Bool
    lowercase::Bool
    model_name::String
    version::VersionNumber
end

struct RobertaByteBPETrainingArtifacts <: AbstractTrainingArtifacts
    inner::ByteBPETrainingArtifacts
    hf_added_tokens::Vector{HFAddedToken}
end

struct GPT2ByteBPETrainingConfig <: AbstractTrainingConfig
    vocab_size::Int
    min_frequency::Int
    special_tokens::Dict{Symbol,String}
    end_of_word_marker::String
    add_prefix_space::Bool
    trim_offsets::Bool
    use_regex::Bool
    export_unk_token_null::Bool
    model_name::String
    version::VersionNumber
end

struct GPT2ByteBPETrainingArtifacts <: AbstractTrainingArtifacts
    inner::ByteBPETrainingArtifacts
    hf_added_tokens::Vector{HFAddedToken}
end
