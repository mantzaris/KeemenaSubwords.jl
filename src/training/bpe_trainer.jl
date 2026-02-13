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
    special_token_set = Set(special_vocab_tokens)

    required_tokens = String[config.special_tokens[:unk], marker]
    unique_required = String[]
    for token in required_tokens
        token in unique_required || push!(unique_required, token)
    end
    required_non_special = [token for token in unique_required if !(token in special_token_set)]

    _validate_required_vocab_capacity(config.vocab_size, special_vocab_tokens, required_non_special)

    words = Vector{Vector{String}}()
    freqs = Int[]
    symbol_set = Set{String}([marker])

    sorted_word_counts = collect(word_counts)
    sort!(sorted_word_counts; by = first)

    for (word, freq) in sorted_word_counts
        symbols = _word_symbols(word, marker)
        push!(words, symbols)
        push!(freqs, freq)
        for symbol in symbols
            push!(symbol_set, symbol)
        end
    end

    merges = Tuple{String,String}[]
    while _effective_vocab_size(symbol_set, special_token_set, special_vocab_tokens) < config.vocab_size
        pair_counts = _pair_frequencies(words, freqs)
        best_pair, best_count = _best_pair(pair_counts)
        best_pair === nothing && break
        best_count < config.min_frequency && break

        push!(merges, best_pair)
        merged_symbol = best_pair[1] * best_pair[2]
        push!(symbol_set, merged_symbol)
        _merge_pair_in_place!(words, best_pair, merged_symbol)
    end

    non_special = [token for token in symbol_set if !(token in special_token_set) && !(token in required_non_special)]
    sort!(non_special)

    vocab_tokens = vcat(special_vocab_tokens, required_non_special, non_special)
    if length(vocab_tokens) > config.vocab_size
        resize!(vocab_tokens, config.vocab_size)
    end

    _ensure_required_tokens_present!(vocab_tokens, unique_required, config.vocab_size)

    special_map = Dict{Symbol,String}()
    for (symbol, token) in ordered_special_pairs
        token in vocab_tokens && (special_map[symbol] = token)
    end
    haskey(special_map, :unk) || throw(ArgumentError("trained vocabulary dropped the required :unk token"))

    vocab = build_vocab(vocab_tokens; special_tokens=special_map)
    pair_ranks, kept_pairs = _pair_ranks_from_merges(vocab_tokens, merges)

    metadata = TokenizerMetadata(:bpe, config.model_name, v"0.3.0", :none)
    tokenizer = BPETokenizer(vocab, pair_ranks, config.special_tokens[:unk], marker, metadata)
    artifacts = BPETrainingArtifacts(copy(vocab_tokens), kept_pairs)
    return TrainingResult(tokenizer, config, artifacts)
end

function _word_symbols(word::String, marker::String)::Vector{String}
    symbols = [string(ch) for ch in collect(word)]
    push!(symbols, marker)
    return symbols
end

function _effective_vocab_size(
    symbol_set::Set{String},
    special_token_set::Set{String},
    ordered_special::Vector{String},
)::Int
    non_special_count = 0
    for symbol in symbol_set
        symbol in special_token_set || (non_special_count += 1)
    end
    return length(ordered_special) + non_special_count
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

function _ensure_required_tokens_present!(
    vocab_tokens::Vector{String},
    required_tokens::Vector{String},
    vocab_size::Int,
)::Nothing
    missing = [token for token in required_tokens if !(token in vocab_tokens)]
    isempty(missing) || throw(ArgumentError(
        "vocab_size=$vocab_size cannot keep required token(s): $(join(missing, ", "))",
    ))
    return nothing
end

function _pair_ranks_from_merges(
    vocab_tokens::Vector{String},
    merges::Vector{Tuple{String,String}},
)::Tuple{Dict{Tuple{String,String},Int},Vector{Tuple{String,String}}}
    vocab_token_set = Set(vocab_tokens)
    pair_ranks = Dict{Tuple{String,String},Int}()
    kept_pairs = Tuple{String,String}[]
    rank = 1

    for pair in merges
        merged = pair[1] * pair[2]
        merged in vocab_token_set || continue
        pair_ranks[pair] = rank
        push!(kept_pairs, pair)
        rank += 1
    end

    return (pair_ranks, kept_pairs)
end
