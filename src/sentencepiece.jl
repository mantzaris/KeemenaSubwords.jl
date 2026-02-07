struct SentencePieceTokenizer <: AbstractSubwordTokenizer
    inner::AbstractSubwordTokenizer
    whitespace_marker::String
    metadata::TokenizerMetadata
end

"""
Load a SentencePiece `.model` file.

Supported model file format in this package is a lightweight text form:
- key/value lines (`type=unigram|bpe`, `whitespace_marker=▁`, `unk_token=<unk>`)
- piece rows: `piece<TAB>token<TAB>score[<TAB>special_symbol]`
- bpe merge rows (for `type=bpe`): `merge<TAB>left<TAB>right`
"""
function load_sentencepiece(
    path::AbstractString;
    model_name::Union{Nothing,AbstractString}=nothing,
)::SentencePieceTokenizer
    isfile(path) || throw(ArgumentError("SentencePiece model path does not exist: $path"))

    parsed = _read_sentencepiece_model(path)
    mtype = parsed.model_type
    marker = parsed.whitespace_marker
    unk = parsed.unk_token

    name = model_name === nothing ? basename(path) : String(model_name)

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
