struct SentencePieceTokenizer <: AbstractSubwordTokenizer
    inner::AbstractSubwordTokenizer
    whitespace_marker::String
    metadata::TokenizerMetadata
end

"""
Load a SentencePiece `.model` file.

Supported inputs:
- standard SentencePiece binary protobuf `.model`/`.model.v3` payloads
- Keemena text-exported model files:
  - key/value lines (`type=unigram|bpe`, `whitespace_marker=▁`, `unk_token=<unk>`)
  - piece rows: `piece<TAB>token<TAB>score[<TAB>special_symbol]`
  - bpe merge rows (for `type=bpe`): `merge<TAB>left<TAB>right`

Examples:
- `load_sentencepiece("/path/to/tokenizer.model"; kind=:auto)`
- `load_sentencepiece("/path/to/tokenizer.model.v3"; kind=:bpe)`
"""
function load_sentencepiece(
    path::AbstractString;
    kind::Symbol=:auto,
    model_name::Union{Nothing,AbstractString}=nothing,
)::SentencePieceTokenizer
    model_path = _resolve_sentencepiece_model_path(path)

    kind in (:auto, :unigram, :bpe) || throw(ArgumentError(
        "Unsupported SentencePiece kind=$kind. Expected :auto, :unigram, or :bpe.",
    ))

    lower_model_path = lowercase(model_path)
    if (endswith(lower_model_path, ".model") || endswith(lower_model_path, ".model.v3")) &&
       _looks_tiktoken_text_payload(_sample_file_bytes(model_path))
        throw(ArgumentError(
            "File appears to be tiktoken text, not a SentencePiece model: $model_path. " *
            "Example: load_tiktoken(\"$model_path\") or load_tokenizer(\"$model_path\"; format=:tiktoken)",
        ))
    end

    parsed = _read_sentencepiece_model(model_path)
    mtype = parsed.model_type
    if kind == :unigram && mtype != :unigram
        throw(ArgumentError("SentencePiece model at $model_path is type=$mtype, not :unigram"))
    elseif kind == :bpe && mtype != :bpe
        throw(ArgumentError("SentencePiece model at $model_path is type=$mtype, not :bpe"))
    end

    marker = parsed.whitespace_marker
    unk = parsed.unk_token

    name = model_name === nothing ? basename(model_path) : String(model_name)

    inner = if mtype == :unigram
        vocab = build_vocab(parsed.tokens; special_tokens=parsed.special_map)
        logprobs = parsed.scores
        length(logprobs) == vocab_size(vocab) || throw(ArgumentError("SentencePiece unigram score count mismatch"))
        haskey(vocab.token_to_id, unk) || throw(ArgumentError("SentencePiece unigram model missing unk token: $unk"))
        inner_meta = TokenizerMetadata(:sentencepiece_unigram, name, v"0.2.0", :none)
        UnigramTokenizer(vocab, logprobs, unk, marker, inner_meta)
    elseif mtype == :bpe
        vocab = build_vocab(parsed.tokens; special_tokens=parsed.special_map)
        haskey(vocab.token_to_id, unk) || throw(ArgumentError("SentencePiece BPE model missing unk token: $unk"))
        ranks = Dict{Tuple{String,String},Int}()
        for (i, pair) in enumerate(parsed.merge_pairs)
            ranks[pair] = i
        end
        inner_meta = TokenizerMetadata(:sentencepiece_bpe, name, v"0.2.0", :none)
        BPETokenizer(vocab, ranks, unk, nothing, inner_meta)
    else
        throw(ArgumentError("Unsupported sentencepiece model type: $mtype"))
    end

    metadata = TokenizerMetadata(:sentencepiece, name, v"0.2.0", :none)
    return SentencePieceTokenizer(inner, marker, metadata)
end

function _resolve_sentencepiece_model_path(path::AbstractString)::String
    if isdir(path)
        candidates = String[]
        for filename in ("spm.model", "spiece.model", "tokenizer.model", "tokenizer.model.v3", "sentencepiece.bpe.model")
            candidate = joinpath(path, filename)
            isfile(candidate) && push!(candidates, candidate)
        end

        isempty(candidates) && throw(ArgumentError(
            "No supported SentencePiece model file found in directory: $path. " *
            "Expected one of spm.model, spiece.model, tokenizer.model, tokenizer.model.v3, sentencepiece.bpe.model. " *
            "Example: load_sentencepiece(\"/path/to/tokenizer.model\")",
        ))
        return candidates[1]
    end

    isfile(path) || throw(ArgumentError(
        "SentencePiece model path does not exist: $path. " *
        "Example: load_sentencepiece(\"/path/to/tokenizer.model\")",
    ))
    return String(path)
end

(tokenizer::SentencePieceTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Tokenize text with SentencePiece wrapper behavior.
"""
function tokenize(tokenizer::SentencePieceTokenizer, text::AbstractString)::Vector{String}
    if tokenizer.inner isa BPETokenizer
        normalized = normalize_text(text)
        out = String[]
        for word in eachsplit(normalized)
            sp_word = string(tokenizer.whitespace_marker, String(word))
            append!(out, _tokenize_bpe_word(tokenizer.inner, sp_word; append_end_marker=false))
        end
        return out
    end

    return tokenize(tokenizer.inner, text)
end

"""
Encode text to SentencePiece IDs.
"""
function encode(
    tokenizer::SentencePieceTokenizer,
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
Decode SentencePiece IDs back to text.
"""
function decode(tokenizer::SentencePieceTokenizer, ids::AbstractVector{Int})::String
    kept = String[]
    for id in ids
        is_skippable_special_id(_inner_vocab(tokenizer.inner), id) && continue
        push!(kept, id_to_token(tokenizer, id))
    end

    joined = join(kept, "")
    with_spaces = replace(joined, tokenizer.whitespace_marker => " ")
    return strip(with_spaces)
end

"""
Forward token lookup.
"""
token_to_id(tokenizer::SentencePieceTokenizer, token::AbstractString)::Int = token_to_id(tokenizer.inner, token)

"""
Reverse token lookup.
"""
id_to_token(tokenizer::SentencePieceTokenizer, id::Int)::String = id_to_token(tokenizer.inner, id)

"""
Vocabulary size.
"""
vocab_size(tokenizer::SentencePieceTokenizer)::Int = vocab_size(tokenizer.inner)

"""
Special token IDs.
"""
special_tokens(tokenizer::SentencePieceTokenizer)::Dict{Symbol,Int} = special_tokens(tokenizer.inner)

"""
Tokenizer metadata.
"""
model_info(tokenizer::SentencePieceTokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
unk_id(tokenizer::SentencePieceTokenizer)::Int = unk_id(tokenizer.inner)

"""
Padding token ID if available.
"""
pad_id(tokenizer::SentencePieceTokenizer)::Union{Int,Nothing} = pad_id(tokenizer.inner)

"""
BOS token ID if available.
"""
bos_id(tokenizer::SentencePieceTokenizer)::Union{Int,Nothing} = bos_id(tokenizer.inner)

"""
EOS token ID if available.
"""
eos_id(tokenizer::SentencePieceTokenizer)::Union{Int,Nothing} = eos_id(tokenizer.inner)

function _inner_vocab(inner::AbstractSubwordTokenizer)::SubwordVocabulary
    if inner isa WordPieceTokenizer
        return inner.vocab
    elseif inner isa BPETokenizer
        return inner.vocab
    elseif inner isa UnigramTokenizer
        return inner.vocab
    elseif inner isa ByteBPETokenizer
        return inner.base.vocab
    end
    throw(ArgumentError("Unsupported inner tokenizer for vocabulary lookup: $(typeof(inner))"))
end

function _read_sentencepiece_model(path::AbstractString)
    bytes = read(path)
    if _looks_text_sentencepiece(bytes)
        try
            return _read_sentencepiece_text_model(path)
        catch
            # Fall through to protobuf parsing for malformed/unknown text-like payloads.
        end
    end
    return _read_sentencepiece_proto_model(bytes, path)
end

function _read_sentencepiece_text_model(path::AbstractString)
    model_type = :unigram
    whitespace_marker = "▁"
    unk_token = "<unk>"

    tokens = String[]
    scores = Float64[]
    merge_pairs = Tuple{String,String}[]
    special_map = Dict{Symbol,String}()

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue

            if occursin('=', line)
                key, value = split(line, '='; limit=2)
                key = strip(key)
                value = strip(value)

                if key == "type"
                    model_type = Symbol(lowercase(value))
                elseif key == "whitespace_marker"
                    whitespace_marker = value
                elseif key == "unk_token"
                    unk_token = value
                end
                continue
            end

            fields = split(line, '\t')
            isempty(fields) && continue

            if fields[1] == "piece"
                length(fields) >= 3 || throw(ArgumentError("Invalid sentencepiece piece row in $path: $line"))
                token = fields[2]
                score = tryparse(Float64, fields[3])
                score === nothing && throw(ArgumentError("Invalid sentencepiece score in $path: $line"))
                push!(tokens, token)
                push!(scores, score)

                if length(fields) >= 4
                    special_map[Symbol(strip(fields[4]))] = token
                end
            elseif fields[1] == "merge"
                length(fields) == 3 || throw(ArgumentError("Invalid sentencepiece merge row in $path: $line"))
                push!(merge_pairs, (fields[2], fields[3]))
            else
                throw(ArgumentError("Unknown sentencepiece row type in $path: $line"))
            end
        end
    end

    isempty(tokens) && throw(ArgumentError("SentencePiece model contains no pieces: $path"))

    if !haskey(special_map, :unk)
        if unk_token in tokens
            special_map[:unk] = unk_token
        else
            pushfirst!(tokens, unk_token)
            pushfirst!(scores, -99.0)
            special_map[:unk] = unk_token
        end
    end

    return (
        model_type=model_type,
        whitespace_marker=whitespace_marker,
        unk_token=unk_token,
        tokens=tokens,
        scores=scores,
        merge_pairs=merge_pairs,
        special_map=special_map,
    )
end

function _looks_text_sentencepiece(bytes::Vector{UInt8})::Bool
    isempty(bytes) && return false
    sample_end = min(length(bytes), 256)
    sample = bytes[1:sample_end]
    any(==(0x00), sample) && return false
    isvalid(String, sample) || return false

    s = String(sample)
    return occursin("type=", s) || occursin("piece\t", s)
end

function _read_sentencepiece_proto_model(bytes::Vector{UInt8}, path::AbstractString)
    pieces = NamedTuple{(:token, :score, :ptype),Tuple{String,Float64,Int}}[]

    i = 1
    n = length(bytes)
    while i <= n
        tag, i = _pb_read_varint(bytes, i, path)
        field = Int(tag >> 3)
        wire = Int(tag & 0x07)

        if field == 1 && wire == 2
            msg_len, i = _pb_read_varint(bytes, i, path)
            stop = i + Int(msg_len) - 1
            (i <= n && stop <= n) || throw(ArgumentError("Invalid sentencepiece protobuf message length in $path"))
            piece = _pb_read_piece(bytes, i, stop, path)
            piece !== nothing && push!(pieces, piece)
            i = stop + 1
        else
            i = _pb_skip_field(bytes, i, wire, path)
        end
    end

    isempty(pieces) && throw(ArgumentError("No piece entries found in SentencePiece protobuf model: $path"))

    tokens = String[]
    scores = Float64[]
    special_map = Dict{Symbol,String}()
    unk_token = "<unk>"

    for piece in pieces
        push!(tokens, piece.token)
        push!(scores, piece.score)

        if piece.ptype == 2
            special_map[:unk] = piece.token
            unk_token = piece.token
        elseif piece.token == "<pad>"
            special_map[:pad] = piece.token
        elseif piece.token == "<s>"
            special_map[:bos] = piece.token
        elseif piece.token == "</s>"
            special_map[:eos] = piece.token
        end
    end

    if !haskey(special_map, :unk)
        if unk_token in tokens
            special_map[:unk] = unk_token
        else
            pushfirst!(tokens, unk_token)
            pushfirst!(scores, -99.0)
            special_map[:unk] = unk_token
        end
    end

    return (
        model_type = :unigram,
        whitespace_marker = "▁",
        unk_token = unk_token,
        tokens = tokens,
        scores = scores,
        merge_pairs = Tuple{String,String}[],
        special_map = special_map,
    )
end

function _pb_read_piece(
    bytes::Vector{UInt8},
    start::Int,
    stop::Int,
    path::AbstractString,
)::Union{Nothing,NamedTuple{(:token, :score, :ptype),Tuple{String,Float64,Int}}}
    token = ""
    score = -99.0
    ptype = 1

    i = start
    while i <= stop
        tag, i = _pb_read_varint(bytes, i, path)
        field = Int(tag >> 3)
        wire = Int(tag & 0x07)

        if field == 1 && wire == 2
            len, i = _pb_read_varint(bytes, i, path)
            end_idx = i + Int(len) - 1
            (i <= stop && end_idx <= stop) || throw(ArgumentError("Invalid piece length in SentencePiece protobuf: $path"))
            token_bytes = bytes[i:end_idx]
            token = String(token_bytes)
            i = end_idx + 1
        elseif field == 2 && wire == 5
            i + 3 <= stop || throw(ArgumentError("Invalid score field in SentencePiece protobuf: $path"))
            raw = UInt32(bytes[i]) |
                  (UInt32(bytes[i + 1]) << 8) |
                  (UInt32(bytes[i + 2]) << 16) |
                  (UInt32(bytes[i + 3]) << 24)
            score = Float64(reinterpret(Float32, raw))
            i += 4
        elseif field == 3 && wire == 0
            v, i = _pb_read_varint(bytes, i, path)
            ptype = Int(v)
        else
            i = _pb_skip_field(bytes, i, wire, path)
        end
    end

    isempty(token) && return nothing
    return (token = token, score = score, ptype = ptype)
end

function _pb_read_varint(
    bytes::Vector{UInt8},
    start::Int,
    path::AbstractString,
)::Tuple{UInt64,Int}
    start <= length(bytes) || throw(ArgumentError("Unexpected end of protobuf stream in $path"))

    value = UInt64(0)
    shift = 0
    i = start

    while true
        i <= length(bytes) || throw(ArgumentError("Unexpected end of varint in $path"))
        b = bytes[i]
        i += 1

        value |= UInt64(b & 0x7f) << shift
        if (b & 0x80) == 0
            return (value, i)
        end

        shift += 7
        shift <= 63 || throw(ArgumentError("Invalid varint in SentencePiece protobuf: $path"))
    end
end

function _pb_skip_field(
    bytes::Vector{UInt8},
    start::Int,
    wire_type::Int,
    path::AbstractString,
)::Int
    if wire_type == 0
        _, i = _pb_read_varint(bytes, start, path)
        return i
    elseif wire_type == 1
        i = start + 8
        i <= length(bytes) + 1 || throw(ArgumentError("Invalid 64-bit field in SentencePiece protobuf: $path"))
        return i
    elseif wire_type == 2
        len, i = _pb_read_varint(bytes, start, path)
        j = i + Int(len)
        j <= length(bytes) + 1 || throw(ArgumentError("Invalid length-delimited field in SentencePiece protobuf: $path"))
        return j
    elseif wire_type == 5
        i = start + 4
        i <= length(bytes) + 1 || throw(ArgumentError("Invalid 32-bit field in SentencePiece protobuf: $path"))
        return i
    end

    throw(ArgumentError("Unsupported protobuf wire type $wire_type in $path"))
end
