"""
High-level training entry points.

These should accept either:
- Vector{String} documents
- iterator of strings
- (future) file paths

Keep memory predictable where possible.
"""
function _normalize_special_tokens(
    special_tokens::Dict{Symbol,String},
)::Dict{Symbol,String}
    normalized = Dict{Symbol,String}()

    alias = Dict(
        :UNK => :unk,
        :PAD => :pad,
        :BOS => :bos,
        :EOS => :eos,
    )

    for (sym, tok) in special_tokens
        mapped = get(alias, sym, sym)
        normalized[mapped] = tok
    end

    haskey(normalized, :unk) || (normalized[:unk] = "<UNK>")
    return normalized
end

function _ordered_special_token_pairs(
    special_tokens::Dict{Symbol,String},
)::Vector{Pair{Symbol,String}}
    pairs = collect(special_tokens)
    sort!(pairs; by = p -> (p.first == :unk ? 0 : 1, String(p.first)))
    return pairs
end

function _collect_word_counts(
    corpus;
    pretokenizer::Union{Nothing,Function}=nothing,
)::Dict{String,Int}
    counts = Dict{String,Int}()

    for doc in corpus
        text = String(doc)
        pieces = pretokenizer === nothing ? eachsplit(text) : pretokenizer(text)

        for piece in pieces
            token = String(piece)
            isempty(token) && continue
            counts[token] = get(counts, token, 0) + 1
        end
    end

    return counts
end

function train_bpe(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(:unk => "<UNK>", :pad => "<PAD>"),
    pretokenizer::Union{Nothing,Function}=nothing,
)::BPETokenizer
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    min_frequency > 0 || throw(ArgumentError("min_frequency must be positive"))

    normalized_specials = _normalize_special_tokens(special_tokens)
    return _train_bpe_impl(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=normalized_specials,
        pretokenizer=pretokenizer,
    )
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
