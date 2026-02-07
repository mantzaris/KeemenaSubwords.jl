"""
Implement Unigram LM training (SentencePiece-style).

Contract:
- Build seed vocab (frequent substrings)
- Run EM iterations to estimate token probabilities
- Prune to target vocab_size
- Output vocab + logprobs
"""
function _train_unigram_impl(
    corpus;
    vocab_size::Int,
    seed_size::Int=200_000,
    num_iters::Int=5,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
)::UnigramTokenizer
    word_counts = _collect_word_counts(corpus; pretokenizer=pretokenizer)
    isempty(word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    ordered_special = _ordered_special_token_pairs(special_tokens)
    special_values = String[]
    for pair in ordered_special
        token = pair.second
        token in special_values || push!(special_values, token)
    end

    candidate_freq = Dict{String,Int}()
    char_freq = Dict{String,Int}()

    for (word, freq) in word_counts
        chars = collect(word)
        n = length(chars)
        n == 0 && continue

        candidate_freq[word] = get(candidate_freq, word, 0) + 2 * freq

        for ch in chars
            c = string(ch)
            candidate_freq[c] = get(candidate_freq, c, 0) + freq
            char_freq[c] = get(char_freq, c, 0) + freq
        end

        max_len = min(6, n)
        for len in 2:max_len
            for i in 1:(n - len + 1)
                sub = String(chars[i:(i + len - 1)])
                candidate_freq[sub] = get(candidate_freq, sub, 0) + freq
            end
        end
    end

    for tok in special_values
        pop!(candidate_freq, tok, nothing)
    end

    ranked = collect(candidate_freq)
    sort!(ranked; by = p -> (-p.second, -length(p.first), p.first))
    if length(ranked) > seed_size
        ranked = ranked[1:seed_size]
    end

    non_special_target = vocab_size - length(special_values)
    non_special_target > 0 || throw(ArgumentError("vocab_size must allow at least one non-special token"))

    charset = collect(keys(char_freq))
    sort!(charset; by = c -> (-get(char_freq, c, 0), c))
    length(charset) <= non_special_target || throw(ArgumentError(
        "vocab_size=$vocab_size is too small: need at least $(length(special_values) + length(charset)) to keep specials + character coverage",
    ))

    seed_limit = max(non_special_target, min(seed_size, max(64, 4 * non_special_target)))
    selected = copy(charset)

    for (tok, _) in ranked
        length(selected) >= seed_limit && break
        tok in selected || push!(selected, tok)
    end

    isempty(selected) && throw(ArgumentError("Could not build a unigram seed vocabulary from the corpus"))

    logprobs = _initial_unigram_logprobs(selected, candidate_freq)
    expected = zeros(Float64, length(selected))

    for _ in 1:num_iters
        expected = _expected_unigram_counts(word_counts, selected, logprobs)
        logprobs = _normalize_logprobs(expected)
    end

    if length(selected) > non_special_target
        selected, logprobs = _prune_unigram_tokens(
            selected,
            logprobs,
            expected,
            Set(charset),
            non_special_target,
            candidate_freq,
        )

        # One refinement pass after pruning gives stable probabilities on the final vocab.
        expected = _expected_unigram_counts(word_counts, selected, logprobs)
        logprobs = _normalize_logprobs(expected)
    end

    unk_token = get(special_tokens, :unk, "<UNK>")
    tokens = vcat(special_values, selected)
    if !(unk_token in tokens)
        pushfirst!(tokens, unk_token)
        length(tokens) > vocab_size && pop!(tokens)
    end
    length(tokens) <= vocab_size || throw(ArgumentError("Unexpected unigram vocabulary overflow"))

    special_map = Dict{Symbol,String}()
    for (sym, tok) in ordered_special
        tok in tokens && (special_map[sym] = tok)
    end
    haskey(special_map, :unk) || (special_map[:unk] = unk_token)

    vocab = build_vocab(tokens; special_tokens=special_map)

    token_logprobs = _final_unigram_logprobs(tokens, selected, logprobs, Set(special_values))

    metadata = TokenizerMetadata(:unigram, "trained_unigram", v"0.3.0", :none)
    return UnigramTokenizer(vocab, token_logprobs, unk_token, "", metadata)
end

function _initial_unigram_logprobs(
    tokens::Vector{String},
    candidate_freq::Dict{String,Int},
)::Vector{Float64}
    weights = Float64[max(get(candidate_freq, tok, 1), 1) for tok in tokens]
    return _normalize_logprobs(weights)
end

function _normalize_logprobs(weights::Vector{Float64})::Vector{Float64}
    adjusted = Float64[max(w, 1e-12) for w in weights]
    total = sum(adjusted)
    total > 0 || throw(ArgumentError("Unigram probability normalization failed: zero total weight"))
    return Float64[log(w / total) for w in adjusted]
end

function _expected_unigram_counts(
    word_counts::Dict{String,Int},
    tokens::Vector{String},
    logprobs::Vector{Float64},
)::Vector{Float64}
    length(tokens) == length(logprobs) || throw(ArgumentError("Unigram token/probability length mismatch"))

    token_to_idx = Dict{String,Int}()
    max_token_len = 1
    for (i, tok) in enumerate(tokens)
        token_to_idx[tok] = i
        max_token_len = max(max_token_len, length(tok))
    end

    expected = zeros(Float64, length(tokens))

    for (word, freq) in word_counts
        chars = collect(word)
        n = length(chars)
        n == 0 && continue

        matches = [Tuple{Int,Int}[] for _ in 1:n]
        for i in 1:n
            upper = min(n, i + max_token_len - 1)
            for j in i:upper
                tok = String(chars[i:j])
                idx = get(token_to_idx, tok, 0)
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

function _prune_unigram_tokens(
    tokens::Vector{String},
    logprobs::Vector{Float64},
    expected::Vector{Float64},
    mandatory::Set{String},
    target_size::Int,
    candidate_freq::Dict{String,Int},
)::Tuple{Vector{String},Vector{Float64}}
    length(tokens) <= target_size && return (copy(tokens), copy(logprobs))

    must_keep = String[]
    for tok in tokens
        tok in mandatory && push!(must_keep, tok)
    end

    extra_slots = target_size - length(must_keep)
    extra_slots >= 0 || throw(ArgumentError("Target size cannot satisfy mandatory unigram tokens"))

    scored = Tuple{String,Float64,Int}[]
    for (i, tok) in enumerate(tokens)
        tok in mandatory && continue
        push!(scored, (tok, expected[i], get(candidate_freq, tok, 0)))
    end
    sort!(scored; by = t -> (-t[2], -t[3], -length(t[1]), t[1]))

    keep_extra = String[]
    for i in 1:min(extra_slots, length(scored))
        push!(keep_extra, scored[i][1])
    end

    pruned_tokens = vcat(must_keep, keep_extra)
    old_index = Dict{String,Int}(tok => i for (i, tok) in enumerate(tokens))
    pruned_weights = Float64[max(expected[old_index[tok]], 1e-12) for tok in pruned_tokens]

    return (pruned_tokens, _normalize_logprobs(pruned_weights))
end

function _final_unigram_logprobs(
    final_tokens::Vector{String},
    segment_tokens::Vector{String},
    segment_logprobs::Vector{Float64},
    special_token_set::Set{String},
)::Vector{Float64}
    probs = Dict{String,Float64}()
    for (tok, lp) in zip(segment_tokens, segment_logprobs)
        probs[tok] = exp(lp)
    end

    weights = Float64[]
    for tok in final_tokens
        if tok in special_token_set
            push!(weights, 1e-4)
        else
            push!(weights, max(get(probs, tok, 1e-12), 1e-12))
        end
    end

    return _normalize_logprobs(weights)
end
