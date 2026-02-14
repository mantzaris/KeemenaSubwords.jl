function _train_sentencepiece_result_impl(
    corpus,
    config::SentencePieceTrainingConfig,
)::TrainingResult{SentencePieceTokenizer,SentencePieceTrainingConfig,SentencePieceTrainingArtifacts}
    _validate_sentencepiece_config(config)

    if config.model_type == :unigram
        return _train_sentencepiece_unigram_result_impl(corpus, config)
    end

    return _train_sentencepiece_bpe_result_impl(corpus, config)
end

function _train_sentencepiece_unigram_result_impl(
    corpus,
    config::SentencePieceTrainingConfig,
)::TrainingResult{SentencePieceTokenizer,SentencePieceTrainingConfig,SentencePieceTrainingArtifacts}
    inner_config = UnigramTrainingConfig(
        config.vocab_size,
        config.seed_size,
        config.num_iters,
        config.max_subword_length,
        config.prune_fraction,
        config.special_tokens,
        config.pretokenizer,
        config.whitespace_marker,
        config.model_name * "_inner",
        config.version,
    )
    inner_result = _train_unigram_result_impl(corpus, inner_config)

    metadata = TokenizerMetadata(:sentencepiece, config.model_name, config.version, :none)
    tokenizer = SentencePieceTokenizer(inner_result.tokenizer, config.whitespace_marker, metadata)
    artifacts = SentencePieceTrainingArtifacts(
        :unigram,
        config.whitespace_marker,
        inner_result.artifacts,
    )

    return TrainingResult(tokenizer, config, artifacts)
end

function _train_sentencepiece_bpe_result_impl(
    corpus,
    config::SentencePieceTrainingConfig,
)::TrainingResult{SentencePieceTokenizer,SentencePieceTrainingConfig,SentencePieceTrainingArtifacts}
    raw_word_counts = _collect_word_counts(corpus; pretokenizer=config.pretokenizer)
    isempty(raw_word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    marked_word_counts = _sentencepiece_marked_word_counts(raw_word_counts, config.whitespace_marker)
    sorted_word_counts = collect(marked_word_counts)
    sort!(sorted_word_counts; by = first)

    words, freqs = _sentencepiece_bpe_sequences(sorted_word_counts)
    base_alphabet = _sentencepiece_bpe_alphabet(words)

    ordered_special_pairs = _ordered_special_token_pairs(config.special_tokens)
    special_vocab_tokens = _ordered_special_token_values(config.special_tokens)
    vocab_tokens, vocab_token_set = _sentencepiece_initialize_bpe_vocab(
        special_vocab_tokens,
        base_alphabet,
        config.vocab_size,
    )

    merge_pairs, pair_ranks = _run_merge_driven_training!(
        words,
        freqs,
        vocab_tokens,
        vocab_token_set;
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
    )

    special_map = _special_map_for_vocab(ordered_special_pairs, vocab_token_set)
    haskey(special_map, :unk) || throw(ArgumentError("trained vocabulary dropped the required :unk token"))

    vocab = build_vocab(vocab_tokens; special_tokens=special_map)
    inner_metadata = TokenizerMetadata(:bpe, config.model_name * "_inner", config.version, :none)
    inner = BPETokenizer(vocab, pair_ranks, config.special_tokens[:unk], nothing, inner_metadata)

    metadata = TokenizerMetadata(:sentencepiece, config.model_name, config.version, :none)
    tokenizer = SentencePieceTokenizer(inner, config.whitespace_marker, metadata)

    inner_artifacts = BPETrainingArtifacts(
        copy(vocab_tokens),
        copy(merge_pairs),
        copy(pair_ranks),
        copy(marked_word_counts),
    )
    artifacts = SentencePieceTrainingArtifacts(
        :bpe,
        config.whitespace_marker,
        inner_artifacts,
    )

    return TrainingResult(tokenizer, config, artifacts)
end

function _sentencepiece_marked_word_counts(
    word_counts::Dict{String,Int},
    whitespace_marker::String,
)::Dict{String,Int}
    marked = Dict{String,Int}()
    for (word, freq) in word_counts
        token = startswith(word, whitespace_marker) ? word : string(whitespace_marker, word)
        marked[token] = get(marked, token, 0) + freq
    end
    return marked
end

function _sentencepiece_bpe_sequences(
    sorted_word_counts::Vector{Pair{String,Int}},
)::Tuple{Vector{Vector{String}},Vector{Int}}
    words = Vector{Vector{String}}()
    freqs = Int[]

    for (word, freq) in sorted_word_counts
        push!(words, [string(ch) for ch in collect(word)])
        push!(freqs, freq)
    end

    return (words, freqs)
end

function _sentencepiece_bpe_alphabet(
    words::Vector{Vector{String}},
)::Vector{String}
    alphabet_set = Set{String}()
    for symbols in words
        for symbol in symbols
            push!(alphabet_set, symbol)
        end
    end
    alphabet = collect(alphabet_set)
    sort!(alphabet)
    return alphabet
end

function _sentencepiece_initialize_bpe_vocab(
    special_vocab_tokens::Vector{String},
    base_alphabet::Vector{String},
    vocab_size::Int,
)::Tuple{Vector{String},Set{String}}
    vocab_tokens = String[]
    vocab_token_set = Set{String}()

    for token in special_vocab_tokens
        _push_unique_token!(vocab_tokens, vocab_token_set, token)
    end
    for token in base_alphabet
        _push_unique_token!(vocab_tokens, vocab_token_set, token)
    end

    _validate_required_vocab_capacity(vocab_size, vocab_tokens)
    return (vocab_tokens, vocab_token_set)
end
