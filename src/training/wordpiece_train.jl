function _train_wordpiece_result_impl(
    corpus,
    config::WordPieceTrainingConfig,
)::TrainingResult{WordPieceTokenizer,WordPieceTrainingConfig,WordPieceTrainingArtifacts}
    _validate_wordpiece_config(config)
    raw_word_counts = _collect_word_counts(corpus; pretokenizer=config.pretokenizer)
    isempty(raw_word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    word_counts = _filter_wordpiece_training_words(
        raw_word_counts,
        config.max_input_chars_per_word,
    )
    isempty(word_counts) && throw(ArgumentError(
        "Empty corpus after applying max_input_chars_per_word=$(config.max_input_chars_per_word)",
    ))

    ordered_special_pairs = _ordered_special_token_pairs(config.special_tokens)
    special_vocab_tokens = _ordered_special_token_values(config.special_tokens)
    sorted_word_counts = collect(word_counts)
    sort!(sorted_word_counts; by = first)

    sequences, frequencies = _wordpiece_initial_sequences(
        sorted_word_counts,
        config.continuation_prefix,
    )
    charset = _wordpiece_charset(sorted_word_counts)

    vocab_tokens, vocab_token_set = _initialize_wordpiece_vocab_tokens(
        special_vocab_tokens,
        charset,
        config.continuation_prefix,
        config.vocab_size,
    )

    merge_pairs, merge_scores = _run_wordpiece_merge_training!(
        sequences,
        frequencies,
        vocab_tokens,
        vocab_token_set;
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        continuation_prefix=config.continuation_prefix,
    )

    special_map = _special_map_for_vocab(ordered_special_pairs, vocab_token_set)
    haskey(special_map, :unk) || throw(ArgumentError("trained vocabulary dropped the required :unk token"))

    vocab = build_vocab(vocab_tokens; special_tokens=special_map)
    metadata = TokenizerMetadata(:wordpiece, config.model_name, config.version, :none)
    tokenizer = WordPieceTokenizer(
        vocab,
        config.continuation_prefix,
        config.special_tokens[:unk],
        config.max_input_chars_per_word,
        metadata,
    )
    artifacts = WordPieceTrainingArtifacts(
        copy(vocab_tokens),
        copy(merge_pairs),
        copy(merge_scores),
        copy(word_counts),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _filter_wordpiece_training_words(
    raw_word_counts::Dict{String,Int},
    max_input_chars_per_word::Int,
)::Dict{String,Int}
    max_input_chars_per_word == 0 && return copy(raw_word_counts)

    filtered = Dict{String,Int}()
    for (word, freq) in raw_word_counts
        length(word) <= max_input_chars_per_word || continue
        filtered[word] = freq
    end
    return filtered
end

function _wordpiece_initial_sequences(
    sorted_word_counts::Vector{Pair{String,Int}},
    continuation_prefix::String,
)::Tuple{Vector{Vector{String}},Vector{Int}}
    sequences = Vector{Vector{String}}()
    frequencies = Int[]

    for (word, freq) in sorted_word_counts
        push!(sequences, _wordpiece_word_sequence(word, continuation_prefix))
        push!(frequencies, freq)
    end

    return (sequences, frequencies)
end

function _wordpiece_word_sequence(
    word::String,
    continuation_prefix::String,
)::Vector{String}
    characters = collect(word)
    isempty(characters) && return String[]

    symbols = String[string(characters[1])]
    for character in characters[2:end]
        push!(symbols, string(continuation_prefix, character))
    end
    return symbols
end

function _wordpiece_charset(
    sorted_word_counts::Vector{Pair{String,Int}},
)::Vector{String}
    charset = Set{String}()
    for (word, _) in sorted_word_counts
        for character in word
            push!(charset, string(character))
        end
    end
    ordered_charset = collect(charset)
    sort!(ordered_charset)
    return ordered_charset
end

function _initialize_wordpiece_vocab_tokens(
    special_vocab_tokens::Vector{String},
    charset::Vector{String},
    continuation_prefix::String,
    vocab_size::Int,
)::Tuple{Vector{String},Set{String}}
    required_tokens = String[]
    required_set = Set{String}()

    for token in special_vocab_tokens
        _push_unique_token!(required_tokens, required_set, token)
    end
    for char_token in charset
        _push_unique_token!(required_tokens, required_set, char_token)
    end
    for char_token in charset
        _push_unique_token!(required_tokens, required_set, string(continuation_prefix, char_token))
    end

    _validate_required_vocab_capacity(vocab_size, required_tokens)
    return (required_tokens, required_set)
end

function _run_wordpiece_merge_training!(
    sequences::Vector{Vector{String}},
    frequencies::Vector{Int},
    vocab_tokens::Vector{String},
    vocab_token_set::Set{String};
    vocab_size::Int,
    min_frequency::Int,
    continuation_prefix::String,
)::Tuple{Vector{Tuple{String,String}},Vector{Float64}}
    merge_pairs = Tuple{String,String}[]
    merge_scores = Float64[]

    while length(vocab_tokens) < vocab_size
        token_counts, pair_counts = _wordpiece_token_and_pair_counts(sequences, frequencies)
        best_pair, best_score, best_pair_count = _best_wordpiece_pair(
            token_counts,
            pair_counts;
            min_frequency=min_frequency,
        )

        best_pair === nothing && break
        best_pair_count < min_frequency && break

        merged_token = _wordpiece_merged_token(best_pair, continuation_prefix)
        _merge_pair_in_place!(sequences, best_pair, merged_token)
        push!(merge_pairs, best_pair)
        push!(merge_scores, best_score)
        _push_unique_token!(vocab_tokens, vocab_token_set, merged_token)
    end

    return (merge_pairs, merge_scores)
end

function _wordpiece_token_and_pair_counts(
    sequences::Vector{Vector{String}},
    frequencies::Vector{Int},
)::Tuple{Dict{String,Int},Dict{Tuple{String,String},Int}}
    token_counts = Dict{String,Int}()
    pair_counts = Dict{Tuple{String,String},Int}()

    for i in eachindex(sequences)
        sequence = sequences[i]
        frequency = frequencies[i]

        for token in sequence
            token_counts[token] = get(token_counts, token, 0) + frequency
        end

        for j in 1:(length(sequence) - 1)
            pair = (sequence[j], sequence[j + 1])
            pair_counts[pair] = get(pair_counts, pair, 0) + frequency
        end
    end

    return (token_counts, pair_counts)
end

function _best_wordpiece_pair(
    token_counts::Dict{String,Int},
    pair_counts::Dict{Tuple{String,String},Int};
    min_frequency::Int,
)::Tuple{Union{Nothing,Tuple{String,String}},Float64,Int}
    isempty(pair_counts) && return (nothing, -Inf, 0)

    best_pair = nothing
    best_score = -Inf
    best_pair_count = typemin(Int)

    for (pair, pair_count) in pair_counts
        pair_count >= min_frequency || continue

        left, right = pair
        left_count = get(token_counts, left, 0)
        right_count = get(token_counts, right, 0)
        left_count > 0 || continue
        right_count > 0 || continue

        denominator = Float64(left_count) * Float64(right_count)
        score = Float64(pair_count) / denominator

        if best_pair === nothing || score > best_score
            best_pair = pair
            best_score = score
            best_pair_count = pair_count
        elseif score == best_score
            if pair_count > best_pair_count
                best_pair = pair
                best_pair_count = pair_count
            elseif pair_count == best_pair_count && _pair_is_lex_less(pair, best_pair)
                best_pair = pair
            end
        end
    end

    best_pair === nothing && return (nothing, -Inf, 0)
    return (best_pair, best_score, best_pair_count)
end

function _wordpiece_merged_token(
    pair::Tuple{String,String},
    continuation_prefix::String,
)::String
    left, right = pair
    merged_base = _wordpiece_strip_prefix(left, continuation_prefix) *
                  _wordpiece_strip_prefix(right, continuation_prefix)
    startswith(left, continuation_prefix) && return string(continuation_prefix, merged_base)
    return merged_base
end

function _wordpiece_strip_prefix(token::String, continuation_prefix::String)::String
    startswith(token, continuation_prefix) || return token
    start_idx = nextind(token, firstindex(token), length(continuation_prefix))
    start_idx > lastindex(token) && return ""
    return String(SubString(token, start_idx))
end
