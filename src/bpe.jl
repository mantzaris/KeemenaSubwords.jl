struct BPETokenizer <: AbstractSubwordTokenizer
    vocab::SubwordVocabulary
    pair_ranks::Dict{Tuple{String,String},Int}
    unk_token::String
    end_of_word_marker::Union{Nothing,String}
    metadata::TokenizerMetadata
end

"""
Load a BPE tokenizer from either a directory (`vocab.txt` + `merges.txt`) or a vocab file path.
"""
function load_bpe(
    path::AbstractString;
    unk_token::AbstractString="<unk>",
    end_of_word_marker::Union{Nothing,AbstractString}="</w>",
    model_name::Union{Nothing,AbstractString}=nothing,
)::BPETokenizer
    vocab_path, merges_path = _resolve_bpe_paths(path)
    return load_bpe(vocab_path, merges_path;
        unk_token=unk_token,
        end_of_word_marker=end_of_word_marker,
        model_name=model_name,
    )
end

"""
Load a BPE tokenizer from explicit vocab + merges paths.
"""
function load_bpe(
    vocab_path::AbstractString,
    merges_path::AbstractString;
    unk_token::AbstractString="<unk>",
    end_of_word_marker::Union{Nothing,AbstractString}="</w>",
    model_name::Union{Nothing,AbstractString}=nothing,
)::BPETokenizer
    tokens = read_token_file(vocab_path)
    pairs = _read_merge_pairs(merges_path)

    resolved_unk_token = _resolve_unk_token(tokens, String(unk_token))
    special_map = detect_special_tokens(tokens, resolved_unk_token)
    vocab = build_vocab(tokens; special_tokens=special_map)

    haskey(vocab.token_to_id, resolved_unk_token) || throw(ArgumentError("BPE vocabulary must include unk token $(unk_token)"))

    pair_ranks = Dict{Tuple{String,String},Int}()
    for (i, pair) in enumerate(pairs)
        pair_ranks[pair] = i
    end

    name = model_name === nothing ? basename(vocab_path) : String(model_name)
    metadata = TokenizerMetadata(:bpe, name, v"0.2.0", :none)

    marker = end_of_word_marker === nothing ? nothing : String(end_of_word_marker)

    return BPETokenizer(vocab, pair_ranks, resolved_unk_token, marker, metadata)
end

(tokenizer::BPETokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Tokenize with classic BPE merges.
"""
function tokenize(tokenizer::BPETokenizer, text::AbstractString)::Vector{String}
    normalized = normalize_text(text)
    pieces = String[]

    for word in eachsplit(normalized)
        append!(pieces, _tokenize_bpe_word(tokenizer, String(word); append_end_marker=true))
    end

    return pieces
end

"""
Encode text to token IDs.
"""
function encode(
    tokenizer::BPETokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)::Vector{Int}
    ids = Int[token_to_id(tokenizer, piece) for piece in tokenize(tokenizer, text)]
    add_special_tokens || return ids

    out = Int[]
    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    bos !== nothing && push!(out, bos)
    append!(out, ids)
    eos !== nothing && push!(out, eos)
    return out
end

"""
Decode token IDs to text.
"""
function decode(tokenizer::BPETokenizer, ids::AbstractVector{Int})::String
    kept = String[]
    for id in ids
        is_skippable_special_id(tokenizer.vocab, id) && continue
        push!(kept, id_to_token(tokenizer, id))
    end

    joined = join(kept, "")
    if tokenizer.end_of_word_marker === nothing
        return strip(joined)
    end

    with_spaces = replace(joined, tokenizer.end_of_word_marker => " ")
    return strip(with_spaces)
end

"""
Forward token lookup.
"""
token_to_id(tokenizer::BPETokenizer, token::AbstractString)::Int = token_to_id(tokenizer.vocab, token)

"""
Reverse token lookup.
"""
id_to_token(tokenizer::BPETokenizer, id::Int)::String = id_to_token(tokenizer.vocab, id)

"""
Vocabulary size.
"""
vocab_size(tokenizer::BPETokenizer)::Int = vocab_size(tokenizer.vocab)

"""
Special token IDs.
"""
special_tokens(tokenizer::BPETokenizer)::Dict{Symbol,Int} = copy(tokenizer.vocab.special_token_ids)

"""
Tokenizer metadata.
"""
model_info(tokenizer::BPETokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
unk_id(tokenizer::BPETokenizer)::Int = unk_id(tokenizer.vocab)

"""
Padding token ID if available.
"""
pad_id(tokenizer::BPETokenizer)::Union{Int,Nothing} = pad_id(tokenizer.vocab)

"""
BOS token ID if available.
"""
bos_id(tokenizer::BPETokenizer)::Union{Int,Nothing} = bos_id(tokenizer.vocab)

"""
EOS token ID if available.
"""
eos_id(tokenizer::BPETokenizer)::Union{Int,Nothing} = eos_id(tokenizer.vocab)

function _resolve_bpe_paths(path::AbstractString)::Tuple{String,String}
    if isdir(path)
        vocab = joinpath(path, "vocab.txt")
        merges = joinpath(path, "merges.txt")
        isfile(vocab) || throw(ArgumentError("Missing vocab.txt in BPE directory: $path"))
        isfile(merges) || throw(ArgumentError("Missing merges.txt in BPE directory: $path"))
        return (vocab, merges)
    end

    isfile(path) || throw(ArgumentError("BPE path does not exist: $path"))
    if lowercase(basename(path)) == "vocab.txt"
        merges = joinpath(dirname(path), "merges.txt")
        isfile(merges) || throw(ArgumentError("Missing merges.txt next to $path"))
        return (String(path), merges)
    end

    throw(ArgumentError("BPE file path must point to vocab.txt or a model directory: $path"))
end

function _read_merge_pairs(path::AbstractString)::Vector{Tuple{String,String}}
    pairs = Tuple{String,String}[]
    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue
            fields = split(line)
            length(fields) == 2 || throw(ArgumentError("Invalid merge line in $path: $line"))
            push!(pairs, (fields[1], fields[2]))
        end
    end
    return pairs
end

function _tokenize_bpe_word(
    tokenizer::BPETokenizer,
    word::String;
    append_end_marker::Bool=true,
)::Vector{String}
    isempty(word) && return String[]

    symbols = [string(c) for c in collect(word)]

    if append_end_marker && tokenizer.end_of_word_marker !== nothing
        push!(symbols, tokenizer.end_of_word_marker)
    end

    symbols = _apply_bpe_merges(symbols, tokenizer.pair_ranks)

    for sym in symbols
        if !haskey(tokenizer.vocab.token_to_id, sym)
            return [tokenizer.unk_token]
        end
    end

    return symbols
end

function _apply_bpe_merges(
    symbols::Vector{String},
    pair_ranks::Dict{Tuple{String,String},Int},
)::Vector{String}
    length(symbols) <= 1 && return symbols

    current = copy(symbols)

    while true
        best_pair = nothing
        best_rank = typemax(Int)

        for i in 1:(length(current) - 1)
            pair = (current[i], current[i + 1])
            rank = get(pair_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end

        best_pair === nothing && break

        merged = String[]
        i = 1
        while i <= length(current)
            if i < length(current) && current[i] == best_pair[1] && current[i + 1] == best_pair[2]
                push!(merged, current[i] * current[i + 1])
                i += 2
            else
                push!(merged, current[i])
                i += 1
            end
        end

        current = merged
    end

    return current
end

function _resolve_unk_token(tokens::Vector{String}, requested::String)::String
    if requested in tokens
        return requested
    end

    for candidate in ("<unk>", "<UNK>", "[UNK]")
        if candidate in tokens
            return candidate
        end
    end

    return requested
end
