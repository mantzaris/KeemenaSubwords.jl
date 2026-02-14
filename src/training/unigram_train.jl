function _train_unigram_impl(
    corpus;
    vocab_size::Int,
    seed_size::Int=200_000,
    num_iters::Int=5,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
)::UnigramTokenizer
    config = UnigramTrainingConfig(
        vocab_size,
        seed_size,
        num_iters,
        6,
        0.2,
        _normalize_special_tokens(special_tokens),
        pretokenizer,
        "",
        "trained_unigram",
        v"0.3.0",
    )
    return _train_unigram_result_impl(corpus, config).tokenizer
end

function _train_unigram_result_impl(
    corpus,
    config::UnigramTrainingConfig,
)::TrainingResult{UnigramTokenizer,UnigramTrainingConfig,UnigramTrainingArtifacts}
    _validate_unigram_config(config)
    raw_word_counts = _collect_word_counts(corpus; pretokenizer=config.pretokenizer)
    isempty(raw_word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    training_word_counts = _apply_unigram_whitespace_marker(raw_word_counts, config.whitespace_marker)
    sorted_word_counts = _sorted_unigram_word_counts(training_word_counts)

    ordered_special = _ordered_special_token_pairs(config.special_tokens)
    special_values = _ordered_special_token_values(config.special_tokens)
    special_set = Set(special_values)

    candidate_freq, char_freq = _build_unigram_candidate_freqs(
        sorted_word_counts;
        max_subword_length=config.max_subword_length,
    )
    for token in special_values
        pop!(candidate_freq, token, nothing)
    end

    non_special_target = config.vocab_size - length(special_values)
    non_special_target > 0 || throw(ArgumentError("vocab_size must allow at least one non-special token"))

    charset = collect(keys(char_freq))
    sort!(charset; by = c -> (-get(char_freq, c, 0), c))
    length(charset) <= non_special_target || throw(ArgumentError(
        "vocab_size=$(config.vocab_size) is too small: need at least $(length(special_values) + length(charset)) to keep specials + character coverage",
    ))

    ranked_candidates = collect(candidate_freq)
    sort!(ranked_candidates; by = p -> (-p.second, -length(p.first), p.first))

    seed_limit = max(non_special_target, min(config.seed_size, max(64, 4 * non_special_target)))
    selected = copy(charset)
    for (token, _) in ranked_candidates
        length(selected) >= seed_limit && break
        token in selected && continue
        push!(selected, token)
    end
    isempty(selected) && throw(ArgumentError("Could not build a unigram seed vocabulary from the corpus"))

    logprobs = _initial_unigram_logprobs(selected, candidate_freq)
    for _ in 1:config.num_iters
        expected = _expected_unigram_counts(sorted_word_counts, selected, logprobs)
        logprobs = _normalize_logprobs(expected)
    end

    mandatory_tokens = Set(charset)
    selected, logprobs = _iterative_unigram_prune(
        sorted_word_counts,
        selected,
        logprobs,
        mandatory_tokens;
        target_size=non_special_target,
        prune_fraction=config.prune_fraction,
        candidate_freq=candidate_freq,
    )

    for _ in 1:config.num_iters
        expected = _expected_unigram_counts(sorted_word_counts, selected, logprobs)
        logprobs = _normalize_logprobs(expected)
    end

    tokens = vcat(special_values, selected)
    length(tokens) <= config.vocab_size || throw(ArgumentError("Unexpected unigram vocabulary overflow"))

    special_map = Dict{Symbol,String}()
    for (symbol, token) in ordered_special
        token in tokens && (special_map[symbol] = token)
    end
    haskey(special_map, :unk) || throw(ArgumentError("special_tokens must include :unk"))

    vocab = build_vocab(tokens; special_tokens=special_map)
    token_logprobs = _final_unigram_logprobs(tokens, selected, logprobs, special_set)

    metadata = TokenizerMetadata(:unigram, config.model_name, config.version, :none)
    tokenizer = UnigramTokenizer(
        vocab,
        token_logprobs,
        config.special_tokens[:unk],
        config.whitespace_marker,
        metadata,
    )
    artifacts = UnigramTrainingArtifacts(copy(tokens), copy(token_logprobs), copy(training_word_counts))
    return TrainingResult(tokenizer, config, artifacts)
end

function _apply_unigram_whitespace_marker(
    word_counts::Dict{String,Int},
    whitespace_marker::String,
)::Dict{String,Int}
    isempty(whitespace_marker) && return copy(word_counts)

    marked = Dict{String,Int}()
    for (word, freq) in word_counts
        token = string(whitespace_marker, word)
        marked[token] = get(marked, token, 0) + freq
    end
    return marked
end

function _sorted_unigram_word_counts(
    word_counts::Dict{String,Int},
)::Vector{Pair{String,Int}}
    sorted_word_counts = collect(word_counts)
    sort!(sorted_word_counts; by = first)
    return sorted_word_counts
end

function _build_unigram_candidate_freqs(
    sorted_word_counts::Vector{Pair{String,Int}};
    max_subword_length::Int,
)::Tuple{Dict{String,Int},Dict{String,Int}}
    candidate_freq = Dict{String,Int}()
    char_freq = Dict{String,Int}()

    for (word, freq) in sorted_word_counts
        chars = collect(word)
        n = length(chars)
        n == 0 && continue

        candidate_freq[word] = get(candidate_freq, word, 0) + 2 * freq

        for ch in chars
            c = string(ch)
            candidate_freq[c] = get(candidate_freq, c, 0) + freq
            char_freq[c] = get(char_freq, c, 0) + freq
        end

        max_len = min(max_subword_length, n)
        for len in 2:max_len
            for i in 1:(n - len + 1)
                subword = String(chars[i:(i + len - 1)])
                candidate_freq[subword] = get(candidate_freq, subword, 0) + freq
            end
        end
    end

    return (candidate_freq, char_freq)
end

function _iterative_unigram_prune(
    sorted_word_counts::Vector{Pair{String,Int}},
    tokens::Vector{String},
    logprobs::Vector{Float64},
    mandatory_tokens::Set{String};
    target_size::Int,
    prune_fraction::Float64,
    candidate_freq::Dict{String,Int},
)::Tuple{Vector{String},Vector{Float64}}
    length(tokens) <= target_size && return (copy(tokens), copy(logprobs))

    current_tokens = copy(tokens)
    current_logprobs = copy(logprobs)

    while length(current_tokens) > target_size
        expected = _expected_unigram_counts(sorted_word_counts, current_tokens, current_logprobs)
        current_logprobs = _normalize_logprobs(expected)
        current_tokens, current_logprobs = _drop_unigram_low_tokens(
            current_tokens,
            current_logprobs,
            expected,
            mandatory_tokens,
            target_size,
            candidate_freq,
            prune_fraction,
        )
    end

    return (current_tokens, current_logprobs)
end

function _drop_unigram_low_tokens(
    tokens::Vector{String},
    logprobs::Vector{Float64},
    expected::Vector{Float64},
    mandatory_tokens::Set{String},
    target_size::Int,
    candidate_freq::Dict{String,Int},
    prune_fraction::Float64,
)::Tuple{Vector{String},Vector{Float64}}
    _ = logprobs
    length(tokens) <= target_size && return (copy(tokens), copy(logprobs))

    excess = length(tokens) - target_size
    removable_indices = Int[]
    scored = Tuple{Float64,Int,Int,String,Int}[]

    for (i, token) in enumerate(tokens)
        token in mandatory_tokens && continue
        push!(removable_indices, i)
        push!(scored, (
            expected[i],
            get(candidate_freq, token, 0),
            length(token),
            token,
            i,
        ))
    end

    length(removable_indices) >= excess || throw(ArgumentError(
        "Target size cannot satisfy mandatory unigram tokens",
    ))

    remove_count = if prune_fraction == 0.0
        1
    else
        max(1, Int(ceil(length(tokens) * prune_fraction)))
    end
    remove_count = min(remove_count, excess)

    sort!(scored; by = s -> (s[1], s[2], s[3], s[4]))
    remove_indices = Set{Int}(s[5] for s in scored[1:remove_count])

    kept_tokens = String[]
    kept_weights = Float64[]
    for (i, token) in enumerate(tokens)
        i in remove_indices && continue
        push!(kept_tokens, token)
        push!(kept_weights, max(expected[i], 1e-12))
    end

    return (kept_tokens, _normalize_logprobs(kept_weights))
end

function _initial_unigram_logprobs(
    tokens::Vector{String},
    candidate_freq::Dict{String,Int},
)::Vector{Float64}
    weights = Float64[max(get(candidate_freq, token, 1), 1) for token in tokens]
    return _normalize_logprobs(weights)
end

function _normalize_logprobs(weights::Vector{Float64})::Vector{Float64}
    adjusted = Float64[max(weight, 1e-12) for weight in weights]
    total = sum(adjusted)
    total > 0 || throw(ArgumentError("Unigram probability normalization failed: zero total weight"))
    return Float64[log(weight / total) for weight in adjusted]
end

function _expected_unigram_counts(
    sorted_word_counts::Vector{Pair{String,Int}},
    tokens::Vector{String},
    logprobs::Vector{Float64},
)::Vector{Float64}
    length(tokens) == length(logprobs) || throw(ArgumentError("Unigram token/probability length mismatch"))

    token_to_idx = Dict{String,Int}()
    max_token_len = 1
    for (idx, token) in enumerate(tokens)
        token_to_idx[token] = idx
        max_token_len = max(max_token_len, length(token))
    end

    expected = zeros(Float64, length(tokens))

    for (word, freq) in sorted_word_counts
        chars = collect(word)
        n = length(chars)
        n == 0 && continue

        matches = [Tuple{Int,Int}[] for _ in 1:n]
        for i in 1:n
            upper = min(n, i + max_token_len - 1)
            for j in i:upper
                token = String(chars[i:j])
                idx = get(token_to_idx, token, 0)
                idx == 0 && continue
                push!(matches[i], (j, idx))
            end
        end

        forward = fill(-Inf, n + 1)
        forward[1] = 0.0
        for i in 1:n
            fi = forward[i]
            isfinite(fi) || continue
            for (j, idx) in matches[i]
                forward[j + 1] = _logaddexp(forward[j + 1], fi + logprobs[idx])
            end
        end

        logz = forward[n + 1]
        isfinite(logz) || continue

        backward = fill(-Inf, n + 1)
        backward[n + 1] = 0.0
        for i in n:-1:1
            acc = -Inf
            for (j, idx) in matches[i]
                acc = _logaddexp(acc, logprobs[idx] + backward[j + 1])
            end
            backward[i] = acc
        end

        for i in 1:n
            fi = forward[i]
            isfinite(fi) || continue
            for (j, idx) in matches[i]
                bj = backward[j + 1]
                isfinite(bj) || continue
                log_post = fi + logprobs[idx] + bj - logz
                expected[idx] += freq * exp(log_post)
            end
        end
    end

    return expected
end

function _logaddexp(a::Float64, b::Float64)::Float64
    if !isfinite(a)
        return b
    elseif !isfinite(b)
        return a
    end

    if a < b
        a, b = b, a
    end
    return a + log1p(exp(b - a))
end

function _final_unigram_logprobs(
    final_tokens::Vector{String},
    segment_tokens::Vector{String},
    segment_logprobs::Vector{Float64},
    special_token_set::Set{String},
)::Vector{Float64}
    probs = Dict{String,Float64}()
    for (token, logprob) in zip(segment_tokens, segment_logprobs)
        probs[token] = exp(logprob)
    end

    weights = Float64[]
    for token in final_tokens
        if token in special_token_set
            push!(weights, 1e-4)
        else
            push!(weights, max(get(probs, token, 1e-12), 1e-12))
        end
    end

    return _normalize_logprobs(weights)
end
