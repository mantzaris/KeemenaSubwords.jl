function _train_bpe_result_impl(
    corpus,
    config::BPETrainingConfig,
)::TrainingResult{BPETokenizer,BPETrainingConfig,BPETrainingArtifacts}
    _validate_bpe_config(config)
    word_counts = _collect_word_counts(corpus; pretokenizer=config.pretokenizer)
    isempty(word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    marker = config.end_of_word_marker
    ordered_special_pairs = _ordered_special_token_pairs(config.special_tokens)
    special_vocab_tokens = _ordered_special_token_values(config.special_tokens)
    required_non_special = _required_non_special_tokens(config.special_tokens, marker)

    words, freqs = _char_bpe_sequences(word_counts, marker)
    base_alphabet = _base_alphabet_from_sequences(words; marker=marker)
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
    metadata = TokenizerMetadata(:bpe, config.model_name, config.version, :none)
    tokenizer = BPETokenizer(vocab, pair_ranks, config.special_tokens[:unk], marker, metadata)
    artifacts = BPETrainingArtifacts(
        copy(vocab_tokens),
        copy(merge_pairs),
        copy(pair_ranks),
        copy(word_counts),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _char_bpe_sequences(
    word_counts::Dict{String,Int},
    marker::String,
)::Tuple{Vector{Vector{String}},Vector{Int}}
    words = Vector{Vector{String}}()
    freqs = Int[]

    sorted_word_counts = collect(word_counts)
    sort!(sorted_word_counts; by = first)

    for (word, freq) in sorted_word_counts
        push!(words, _word_symbols(word, marker))
        push!(freqs, freq)
    end

    return (words, freqs)
end

function _word_symbols(word::String, marker::String)::Vector{String}
    symbols = [string(ch) for ch in collect(word)]
    push!(symbols, marker)
    return symbols
end

function _required_non_special_tokens(
    special_tokens::Dict{Symbol,String},
    marker::String,
)::Vector{String}
    special_token_set = Set(values(special_tokens))
    required = String[]
    marker in special_token_set || push!(required, marker)
    return required
end

function _base_alphabet_from_sequences(
    words::Vector{Vector{String}};
    marker::String,
)::Vector{String}
    symbol_set = Set{String}()
    for symbols in words
        for symbol in symbols
            symbol == marker && continue
            push!(symbol_set, symbol)
        end
    end

    base_alphabet = collect(symbol_set)
    sort!(base_alphabet)
    return base_alphabet
end

function _initialize_bpe_vocab_tokens(
    special_vocab_tokens::Vector{String},
    required_non_special::Vector{String},
    base_alphabet::Vector{String},
    vocab_size::Int,
)::Tuple{Vector{String},Set{String}}
    vocab_tokens = String[]
    vocab_token_set = Set{String}()

    for token in special_vocab_tokens
        _push_unique_token!(vocab_tokens, vocab_token_set, token)
    end
    for token in required_non_special
        _push_unique_token!(vocab_tokens, vocab_token_set, token)
    end
    for symbol in base_alphabet
        _push_unique_token!(vocab_tokens, vocab_token_set, symbol)
    end

    _validate_required_vocab_capacity(vocab_size, vocab_tokens)
    return (vocab_tokens, vocab_token_set)
end

function _run_merge_driven_training!(
    words::Vector{Vector{String}},
    freqs::Vector{Int},
    vocab_tokens::Vector{String},
    vocab_token_set::Set{String};
    vocab_size::Int,
    min_frequency::Int,
)::Tuple{Vector{Tuple{String,String}},Dict{Tuple{String,String},Int}}
    merges = Tuple{String,String}[]

    while length(vocab_tokens) < vocab_size
        pair_counts = _pair_frequencies(words, freqs)
        best_pair, best_count = _best_pair(pair_counts)
        best_pair === nothing && break
        best_count < min_frequency && break

        merged_symbol = best_pair[1] * best_pair[2]
        _merge_pair_in_place!(words, best_pair, merged_symbol)
        push!(merges, best_pair)
        _push_unique_token!(vocab_tokens, vocab_token_set, merged_symbol)
    end

    pair_ranks = Dict{Tuple{String,String},Int}()
    for (rank, pair) in enumerate(merges)
        haskey(pair_ranks, pair) || (pair_ranks[pair] = rank)
    end

    return (merges, pair_ranks)
end

function _special_map_for_vocab(
    ordered_special_pairs::Vector{Pair{Symbol,String}},
    vocab_token_set::Set{String},
)::Dict{Symbol,String}
    special_map = Dict{Symbol,String}()
    for (symbol, token) in ordered_special_pairs
        token in vocab_token_set && (special_map[symbol] = token)
    end
    return special_map
end

function _pair_frequencies(
    words::Vector{Vector{String}},
    freqs::Vector{Int},
)::Dict{Tuple{String,String},Int}
    pair_counts = Dict{Tuple{String,String},Int}()
    for i in eachindex(words)
        symbols = words[i]
        freq = freqs[i]
        for j in 1:(length(symbols) - 1)
            pair = (symbols[j], symbols[j + 1])
            pair_counts[pair] = get(pair_counts, pair, 0) + freq
        end
    end
    return pair_counts
end

function _best_pair(
    pair_counts::Dict{Tuple{String,String},Int},
)::Tuple{Union{Nothing,Tuple{String,String}},Int}
    isempty(pair_counts) && return (nothing, 0)

    best_pair = nothing
    best_count = typemin(Int)

    for (pair, count) in pair_counts
        if count > best_count
            best_pair = pair
            best_count = count
        elseif count == best_count && best_pair !== nothing && _pair_is_lex_less(pair, best_pair)
            best_pair = pair
        end
    end

    best_pair === nothing && return (nothing, 0)
    return (best_pair, best_count)
end

function _pair_is_lex_less(
    lhs::Tuple{String,String},
    rhs::Tuple{String,String},
)::Bool
    lhs[1] < rhs[1] && return true
    lhs[1] > rhs[1] && return false
    return lhs[2] < rhs[2]
end

function _merge_pair_in_place!(
    words::Vector{Vector{String}},
    pair::Tuple{String,String},
    merged_symbol::String,
)::Nothing
    left, right = pair
    for i in eachindex(words)
        source = words[i]
        destination = String[]
        j = 1
        while j <= length(source)
            if j < length(source) && source[j] == left && source[j + 1] == right
                push!(destination, merged_symbol)
                j += 2
            else
                push!(destination, source[j])
                j += 1
            end
        end
        words[i] = destination
    end
    return nothing
end
