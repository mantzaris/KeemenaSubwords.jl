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
end

struct BPETrainingArtifacts <: AbstractTrainingArtifacts
    vocab_tokens::Vector{String}
    merge_pairs::Vector{Tuple{String,String}}
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
