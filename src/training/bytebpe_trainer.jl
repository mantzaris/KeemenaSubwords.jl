function _train_bytebpe_result_impl(
    corpus,
    config::ByteBPETrainingConfig,
)::TrainingResult{ByteBPETokenizer,ByteBPETrainingConfig,ByteBPETrainingArtifacts}
    _validate_bytebpe_config(config)
    word_counts = _collect_word_counts(corpus; pretokenizer=config.pretokenizer)
    isempty(word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    marker = config.end_of_word_marker
    ordered_special_pairs = _ordered_special_token_pairs(config.special_tokens)
    special_vocab_tokens = _ordered_special_token_values(config.special_tokens)
    required_non_special = _required_non_special_tokens(config.special_tokens, marker)

    byte_to_unicode, unicode_to_byte = _byte_unicode_tables()
    words, freqs = _byte_bpe_sequences(word_counts, marker, byte_to_unicode)
    base_alphabet = config.include_full_byte_alphabet ?
        _full_byte_alphabet_symbols(byte_to_unicode) :
        _base_alphabet_from_sequences(words; marker=marker)

    vocab_tokens, vocab_token_set = _initialize_bpe_vocab_tokens(
        special_vocab_tokens,
        required_non_special,
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
    base_meta = TokenizerMetadata(:bpe, config.model_name, config.version, :none)
    base = BPETokenizer(vocab, pair_ranks, config.special_tokens[:unk], marker, base_meta)
    metadata = TokenizerMetadata(:bytebpe, config.model_name, config.version, :none)
    tokenizer = ByteBPETokenizer(base, byte_to_unicode, unicode_to_byte, metadata)

    artifacts = ByteBPETrainingArtifacts(
        copy(vocab_tokens),
        copy(merge_pairs),
        copy(pair_ranks),
        copy(word_counts),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _byte_bpe_sequences(
    word_counts::Dict{String,Int},
    marker::String,
    byte_to_unicode::Vector{Char},
)::Tuple{Vector{Vector{String}},Vector{Int}}
    words = Vector{Vector{String}}()
    freqs = Int[]

    sorted_word_counts = collect(word_counts)
    sort!(sorted_word_counts; by = first)

    for (word, freq) in sorted_word_counts
        push!(words, _byte_word_symbols(word, marker, byte_to_unicode))
        push!(freqs, freq)
    end

    return (words, freqs)
end

function _byte_word_symbols(
    word::String,
    marker::String,
    byte_to_unicode::Vector{Char},
)::Vector{String}
    symbols = String[]
    for byte in codeunits(word)
        push!(symbols, string(byte_to_unicode[Int(byte) + 1]))
    end
    push!(symbols, marker)
    return symbols
end

function _full_byte_alphabet_symbols(byte_to_unicode::Vector{Char})::Vector{String}
    return [string(byte_to_unicode[byte + 1]) for byte in 0:255]
end
