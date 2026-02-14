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
    max_subword_length::Int=6,
    prune_fraction::Float64=0.2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
    whitespace_marker::String="▁",
    model_name::String="trained_unigram",
    version::VersionNumber=v"0.3.0",
)::UnigramTokenizer
    return train_unigram_result(
        corpus;
        vocab_size=vocab_size,
        seed_size=seed_size,
        num_iters=num_iters,
        max_subword_length=max_subword_length,
        prune_fraction=prune_fraction,
        special_tokens=special_tokens,
        pretokenizer=pretokenizer,
        whitespace_marker=whitespace_marker,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
Train a Unigram tokenizer and return model artifacts.
"""
function train_unigram_result(
    corpus;
    vocab_size::Int,
    seed_size::Int=200_000,
    num_iters::Int=5,
    max_subword_length::Int=6,
    prune_fraction::Float64=0.2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
    whitespace_marker::String="▁",
    model_name::String="trained_unigram",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{UnigramTokenizer,UnigramTrainingConfig,UnigramTrainingArtifacts}
    config = UnigramTrainingConfig(
        vocab_size,
        seed_size,
        num_iters,
        max_subword_length,
        prune_fraction,
        _normalize_special_tokens(special_tokens),
        pretokenizer,
        String(whitespace_marker),
        String(model_name),
        version,
    )
    return _train_unigram_result_impl(corpus, config)
end

"""
Train a WordPiece tokenizer.
"""
function train_wordpiece(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "[UNK]",
        :pad => "[PAD]",
        :cls => "[CLS]",
        :sep => "[SEP]",
        :mask => "[MASK]",
    ),
    pretokenizer::Union{Nothing,Function}=nothing,
    continuation_prefix::String="##",
    max_input_chars_per_word::Int=100,
    model_name::String="trained_wordpiece",
    version::VersionNumber=v"0.3.0",
)::WordPieceTokenizer
    return train_wordpiece_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        pretokenizer=pretokenizer,
        continuation_prefix=continuation_prefix,
        max_input_chars_per_word=max_input_chars_per_word,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
Train a WordPiece tokenizer and return model artifacts.
"""
function train_wordpiece_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "[UNK]",
        :pad => "[PAD]",
        :cls => "[CLS]",
        :sep => "[SEP]",
        :mask => "[MASK]",
    ),
    pretokenizer::Union{Nothing,Function}=nothing,
    continuation_prefix::String="##",
    max_input_chars_per_word::Int=100,
    model_name::String="trained_wordpiece",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{WordPieceTokenizer,WordPieceTrainingConfig,WordPieceTrainingArtifacts}
    config = WordPieceTrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        pretokenizer,
        String(continuation_prefix),
        max_input_chars_per_word,
        String(model_name),
        version,
    )
    return _train_wordpiece_result_impl(corpus, config)
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
