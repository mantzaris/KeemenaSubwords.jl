"""
Implement BPE training.

Contract:
- Build initial symbol vocabulary (characters or byte symbols)
- Iteratively merge most frequent pairs until vocab_size reached
- Output merges + vocab

Implementation note:
- Use a priority queue for pair counts OR a batched recomputation strategy.
- Keep it correct first; optimize later.
"""
function _train_bpe_impl(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
)::BPETokenizer
    word_counts = _collect_word_counts(corpus; pretokenizer=pretokenizer)
    isempty(word_counts) && throw(ArgumentError("Empty corpus: no trainable tokens found"))

    ordered_special = _ordered_special_token_pairs(special_tokens)
    special_values = String[]
    for pair in ordered_special
        token = pair.second
        token in special_values || push!(special_values, token)
    end

    marker = "</w>"
    words = Vector{Vector{String}}()
    freqs = Int[]
    token_set = Set{String}()
    push!(token_set, marker)

    sorted_words = sort(collect(word_counts); by=first)
    for (word, freq) in sorted_words
        symbols = [string(ch) for ch in collect(word)]
        push!(symbols, marker)
        push!(words, symbols)
        push!(freqs, freq)
        foreach(sym -> push!(token_set, sym), symbols)
    end

    merges = Tuple{String,String}[]
    while true
        current_vocab_size = length(special_values) + length(setdiff(token_set, Set(special_values)))
        current_vocab_size >= vocab_size && break

        pair_counts = Dict{Tuple{String,String},Int}()
        for i in eachindex(words)
            seq = words[i]
            w = freqs[i]
            for j in 1:(length(seq)-1)
                pair = (seq[j], seq[j + 1])
                pair_counts[pair] = get(pair_counts, pair, 0) + w
            end
        end

        isempty(pair_counts) && break
        ranked_pairs = collect(pair_counts)
        sort!(ranked_pairs; by = p -> (-p.second, p.first[1], p.first[2]))
        best_pair, best_count = ranked_pairs[1]
        best_count < min_frequency && break

        push!(merges, best_pair)
        merged_symbol = best_pair[1] * best_pair[2]
        push!(token_set, merged_symbol)

        for i in eachindex(words)
            src = words[i]
            dst = String[]
            k = 1
            while k <= length(src)
                if k < length(src) && src[k] == best_pair[1] && src[k + 1] == best_pair[2]
                    push!(dst, merged_symbol)
                    k += 2
                else
                    push!(dst, src[k])
                    k += 1
                end
            end
            words[i] = dst
        end
    end

    non_special = sort(collect(setdiff(token_set, Set(special_values))))
    tokens = vcat(special_values, non_special)
    if length(tokens) > vocab_size
        tokens = tokens[1:vocab_size]
    end

    unk_token = get(special_tokens, :unk, "<UNK>")
    if !(unk_token in tokens)
        pushfirst!(tokens, unk_token)
        length(tokens) > vocab_size && pop!(tokens)
    end

    special_map = Dict{Symbol,String}()
    for (sym, tok) in ordered_special
        tok in tokens && (special_map[sym] = tok)
    end
    haskey(special_map, :unk) || (special_map[:unk] = unk_token)

    vocab = build_vocab(tokens; special_tokens=special_map)

    pair_ranks = Dict{Tuple{String,String},Int}()
    rank = 1
    for pair in merges
        merged = pair[1] * pair[2]
        if haskey(vocab.token_to_id, merged)
            pair_ranks[pair] = rank
            rank += 1
        end
    end

    metadata = TokenizerMetadata(:bpe, "trained_bpe", v"0.3.0", :none)
    return BPETokenizer(vocab, pair_ranks, unk_token, marker, metadata)
end
