"""
Train a character-level BPE tokenizer.
"""
function train_bpe(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
    end_of_word_marker::String="</w>",
    model_name::String="trained_bpe",
    version::VersionNumber=v"0.3.0",
)::BPETokenizer
    return train_bpe_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        pretokenizer=pretokenizer,
        end_of_word_marker=end_of_word_marker,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
Train a character-level BPE tokenizer and return model artifacts.
"""
function train_bpe_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
    end_of_word_marker::String="</w>",
    model_name::String="trained_bpe",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{BPETokenizer,BPETrainingConfig,BPETrainingArtifacts}
    config = BPETrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        pretokenizer,
        String(end_of_word_marker),
        String(model_name),
        version,
    )
    return _train_bpe_result_impl(corpus, config)
end

"""
Train a byte-level BPE tokenizer.
"""
function train_bytebpe(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    end_of_word_marker::String="</w>",
    pretokenizer::Union{Nothing,Function}=nothing,
    include_full_byte_alphabet::Bool=true,
    model_name::String="trained_bytebpe",
    version::VersionNumber=v"0.3.0",
)::ByteBPETokenizer
    return train_bytebpe_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        end_of_word_marker=end_of_word_marker,
        pretokenizer=pretokenizer,
        include_full_byte_alphabet=include_full_byte_alphabet,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
Train a byte-level BPE tokenizer and return model artifacts.
"""
function train_bytebpe_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    end_of_word_marker::String="</w>",
    pretokenizer::Union{Nothing,Function}=nothing,
    include_full_byte_alphabet::Bool=true,
    model_name::String="trained_bytebpe",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{ByteBPETokenizer,ByteBPETrainingConfig,ByteBPETrainingArtifacts}
    config = ByteBPETrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        String(end_of_word_marker),
        pretokenizer,
        include_full_byte_alphabet,
        String(model_name),
        version,
    )
    return _train_bytebpe_result_impl(corpus, config)
end

"""
High-level Unigram training entry point.
"""
function train_unigram(
    corpus;
    vocab_size::Int,
    seed_size::Int=200_000,
    num_iters::Int=5,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
)::UnigramTokenizer
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    seed_size > 0 || throw(ArgumentError("seed_size must be positive"))
    num_iters > 0 || throw(ArgumentError("num_iters must be positive"))

    normalized_specials = _normalize_special_tokens(special_tokens)
    return _train_unigram_impl(
        corpus;
        vocab_size=vocab_size,
        seed_size=seed_size,
        num_iters=num_iters,
        special_tokens=normalized_specials,
        pretokenizer=pretokenizer,
    )
end

"""
Optional WordPiece training entry point.

This API is reserved for a later iteration.
"""
function train_wordpiece(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "[UNK]", :pad => "[PAD]"),
    continuation_prefix::String="##",
)::WordPieceTokenizer
    return _train_wordpiece_impl(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        continuation_prefix=continuation_prefix,
    )
end

"""
Optional SentencePiece training entry point.

This API is reserved for a later iteration.
"""
function train_sentencepiece(
    corpus;
    vocab_size::Int,
    model_type::Symbol=:unigram,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<unk>", :pad => "<pad>"),
)::SentencePieceTokenizer
    return _train_sentencepiece_impl(
        corpus;
        vocab_size=vocab_size,
        model_type=model_type,
        special_tokens=special_tokens,
    )
end
