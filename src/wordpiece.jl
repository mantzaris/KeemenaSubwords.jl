struct WordPieceTokenizer <: AbstractSubwordTokenizer
    vocab::SubwordVocabulary
    continuation_prefix::String
    unk_token::String
    metadata::TokenizerMetadata
end

"""
Load a WordPiece tokenizer from a vocab file path or a directory containing `vocab.txt`.
"""
function load_wordpiece(
    path::AbstractString;
    continuation_prefix::AbstractString="##",
    unk_token::AbstractString="[UNK]",
    model_name::Union{Nothing,AbstractString}=nothing,
)::WordPieceTokenizer
    vocab_path = _wordpiece_vocab_path(path)
    tokens = read_token_file(vocab_path)

    special_map = detect_special_tokens(tokens, String(unk_token))
    vocab = build_vocab(tokens; special_tokens=special_map)

    haskey(vocab.token_to_id, String(unk_token)) || throw(ArgumentError("WordPiece vocabulary must include unk token $(unk_token)"))

    name = model_name === nothing ? basename(vocab_path) : String(model_name)
    metadata = TokenizerMetadata(:wordpiece, name, v"0.2.0", :none)

    return WordPieceTokenizer(vocab, String(continuation_prefix), String(unk_token), metadata)
end

(tokenizer::WordPieceTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Greedy longest-match WordPiece tokenization.
"""
function tokenize(tokenizer::WordPieceTokenizer, text::AbstractString)::Vector{String}
    normalized = normalize_text(text)
    pieces = String[]

    for word in eachsplit(normalized)
        append!(pieces, _tokenize_word(tokenizer, String(word)))
    end

    return pieces
end

"""
Encode text to WordPiece token IDs.
"""
function encode(
    tokenizer::WordPieceTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)::Vector{Int}
    ids = Int[token_to_id(tokenizer, piece) for piece in tokenize(tokenizer, text)]
    add_special_tokens || return ids

    out = Int[]
    cls = get(tokenizer.vocab.special_token_ids, :cls, nothing)
    sep = get(tokenizer.vocab.special_token_ids, :sep, nothing)

    cls !== nothing && push!(out, cls)
    append!(out, ids)
    sep !== nothing && push!(out, sep)

    return out
end

"""
Decode WordPiece token IDs back into text.
"""
function decode(tokenizer::WordPieceTokenizer, ids::AbstractVector{Int})::String
    tokens = String[]
    for id in ids
        is_skippable_special_id(tokenizer.vocab, id) && continue
        push!(tokens, id_to_token(tokenizer, id))
    end

    words = String[]
    current = ""

    for token in tokens
        if startswith(token, tokenizer.continuation_prefix)
            fragment = _strip_prefix(token, tokenizer.continuation_prefix)
            if isempty(current)
                current = fragment
            else
                current *= fragment
            end
        else
            if !isempty(current)
                push!(words, current)
            end
            current = token
        end
    end

    !isempty(current) && push!(words, current)
    return join(words, " ")
end

"""
Forward token lookup.
"""
token_to_id(tokenizer::WordPieceTokenizer, token::AbstractString)::Int = token_to_id(tokenizer.vocab, token)

"""
Reverse token lookup.
"""
id_to_token(tokenizer::WordPieceTokenizer, id::Int)::String = id_to_token(tokenizer.vocab, id)

"""
Vocabulary size.
"""
vocab_size(tokenizer::WordPieceTokenizer)::Int = vocab_size(tokenizer.vocab)

"""
Special token IDs.
"""
special_tokens(tokenizer::WordPieceTokenizer)::Dict{Symbol,Int} = copy(tokenizer.vocab.special_token_ids)

"""
Tokenizer metadata.
"""
model_info(tokenizer::WordPieceTokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
unk_id(tokenizer::WordPieceTokenizer)::Int = unk_id(tokenizer.vocab)

"""
Padding token ID if available.
"""
pad_id(tokenizer::WordPieceTokenizer)::Union{Int,Nothing} = pad_id(tokenizer.vocab)

"""
BOS token ID if available.
"""
bos_id(tokenizer::WordPieceTokenizer)::Union{Int,Nothing} = bos_id(tokenizer.vocab)

"""
EOS token ID if available.
"""
eos_id(tokenizer::WordPieceTokenizer)::Union{Int,Nothing} = eos_id(tokenizer.vocab)

function _wordpiece_vocab_path(path::AbstractString)::String
    if isdir(path)
        candidate = joinpath(path, "vocab.txt")
        isfile(candidate) || throw(ArgumentError("No vocab.txt found in directory: $path"))
        return candidate
    end

    isfile(path) || throw(ArgumentError("WordPiece vocab file not found: $path"))
    return String(path)
end

function _tokenize_word(tokenizer::WordPieceTokenizer, word::String)::Vector{String}
    isempty(word) && return String[]

    pieces = String[]
    first_char = firstindex(word)
    last_char = lastindex(word)
    start = first_char

    while start <= last_char
        stop = last_char
        matched_piece = nothing
        matched_stop = stop

        while stop >= start
            base_piece = word[start:stop]
            candidate = start == first_char ? base_piece : string(tokenizer.continuation_prefix, base_piece)

            if haskey(tokenizer.vocab.token_to_id, candidate)
                matched_piece = candidate
                matched_stop = stop
                break
            end

            stop == start && break
            stop = prevind(word, stop)
        end

        if matched_piece === nothing
            return [tokenizer.unk_token]
        end

        push!(pieces, matched_piece)

        matched_stop == last_char && break
        start = nextind(word, matched_stop)
    end

    return pieces
end

function _strip_prefix(token::String, prefix::String)::String
    startswith(token, prefix) || return token
    start_idx = firstindex(token)
    for _ in 1:length(prefix)
        start_idx = nextind(token, start_idx)
    end
    start_idx > lastindex(token) && return ""
    return String(SubString(token, start_idx))
end
