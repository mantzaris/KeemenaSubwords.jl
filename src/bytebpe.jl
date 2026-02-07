struct ByteBPETokenizer <: AbstractSubwordTokenizer
    base::BPETokenizer
    byte_to_unicode::Vector{Char}
    unicode_to_byte::Dict{Char,UInt8}
    metadata::TokenizerMetadata
end

"""
Load a byte-level BPE tokenizer from a directory (`vocab.txt` + `merges.txt`) or vocab path.
"""
function load_bytebpe(
    path::AbstractString;
    unk_token::AbstractString="<unk>",
    end_of_word_marker::Union{Nothing,AbstractString}="</w>",
    model_name::Union{Nothing,AbstractString}=nothing,
)::ByteBPETokenizer
    vocab_path, merges_path = _resolve_bpe_paths(path)
    return load_bytebpe(vocab_path, merges_path;
        unk_token=unk_token,
        end_of_word_marker=end_of_word_marker,
        model_name=model_name,
    )
end

"""
Load a byte-level BPE tokenizer from explicit vocab + merges paths.
"""
function load_bytebpe(
    vocab_path::AbstractString,
    merges_path::AbstractString;
    unk_token::AbstractString="<unk>",
    end_of_word_marker::Union{Nothing,AbstractString}="</w>",
    model_name::Union{Nothing,AbstractString}=nothing,
)::ByteBPETokenizer
    base = load_bpe(vocab_path, merges_path;
        unk_token=unk_token,
        end_of_word_marker=end_of_word_marker,
        model_name=model_name,
    )

    b2u, u2b = _byte_unicode_tables()
    name = model_name === nothing ? basename(vocab_path) : String(model_name)
    metadata = TokenizerMetadata(:bytebpe, name, v"0.2.0", :none)

    return ByteBPETokenizer(base, b2u, u2b, metadata)
end

(tokenizer::ByteBPETokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Tokenize text by first mapping bytes to unicode symbols, then applying BPE merges.
"""
function tokenize(tokenizer::ByteBPETokenizer, text::AbstractString)::Vector{String}
    normalized = normalize_text(text)
    pieces = String[]

    for word in eachsplit(normalized)
        append!(pieces, _tokenize_byte_word(tokenizer, String(word)))
    end

    return pieces
end

"""
Encode text to byte-level BPE IDs.
"""
function encode(
    tokenizer::ByteBPETokenizer,
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
Decode byte-level BPE IDs back to text.
"""
function decode(tokenizer::ByteBPETokenizer, ids::AbstractVector{Int})::String
    kept = String[]
    for id in ids
        is_skippable_special_id(tokenizer.base.vocab, id) && continue
        push!(kept, id_to_token(tokenizer, id))
    end

    joined = join(kept, "")
    marker = tokenizer.base.end_of_word_marker
    chunks = marker === nothing ? [joined] : split(joined, marker)

    words = String[]
    for chunk in chunks
        isempty(chunk) && continue
        bytes = UInt8[]
        for c in chunk
            if haskey(tokenizer.unicode_to_byte, c)
                push!(bytes, tokenizer.unicode_to_byte[c])
            else
                return tokenizer.base.unk_token
            end
        end

        word = try
            String(bytes)
        catch
            tokenizer.base.unk_token
        end

        push!(words, word)
    end

    return join(words, " ")
end

"""
Forward token lookup.
"""
token_to_id(tokenizer::ByteBPETokenizer, token::AbstractString)::Int = token_to_id(tokenizer.base, token)

"""
Reverse token lookup.
"""
id_to_token(tokenizer::ByteBPETokenizer, id::Int)::String = id_to_token(tokenizer.base, id)

"""
Vocabulary size.
"""
vocab_size(tokenizer::ByteBPETokenizer)::Int = vocab_size(tokenizer.base)

"""
Special token IDs.
"""
special_tokens(tokenizer::ByteBPETokenizer)::Dict{Symbol,Int} = special_tokens(tokenizer.base)

"""
Tokenizer metadata.
"""
model_info(tokenizer::ByteBPETokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
unk_id(tokenizer::ByteBPETokenizer)::Int = unk_id(tokenizer.base)

"""
Padding token ID if available.
"""
pad_id(tokenizer::ByteBPETokenizer)::Union{Int,Nothing} = pad_id(tokenizer.base)

"""
BOS token ID if available.
"""
bos_id(tokenizer::ByteBPETokenizer)::Union{Int,Nothing} = bos_id(tokenizer.base)

"""
EOS token ID if available.
"""
eos_id(tokenizer::ByteBPETokenizer)::Union{Int,Nothing} = eos_id(tokenizer.base)

function _tokenize_byte_word(tokenizer::ByteBPETokenizer, word::String)::Vector{String}
    isempty(word) && return String[]

    symbols = String[]
    for b in codeunits(word)
        mapped = tokenizer.byte_to_unicode[Int(b) + 1]
        push!(symbols, string(mapped))
    end

    if tokenizer.base.end_of_word_marker !== nothing
        push!(symbols, tokenizer.base.end_of_word_marker)
    end

    merged = _apply_bpe_merges(symbols, tokenizer.base.pair_ranks)

    for sym in merged
        if !haskey(tokenizer.base.vocab.token_to_id, sym)
            return [tokenizer.base.unk_token]
        end
    end

    return merged
end

function _byte_unicode_tables()::Tuple{Vector{Char},Dict{Char,UInt8}}
    bs = Int[]
    append!(bs, 33:126)
    append!(bs, 161:172)
    append!(bs, 174:255)

    cs = copy(bs)
    seen = Set(bs)
    extra = 0

    for b in 0:255
        if !(b in seen)
            push!(bs, b)
            push!(cs, 256 + extra)
            extra += 1
        end
    end

    byte_to_unicode = Vector{Char}(undef, 256)
    unicode_to_byte = Dict{Char,UInt8}()

    for i in eachindex(bs)
        b = bs[i]
        c = Char(cs[i])
        byte_to_unicode[b + 1] = c
        unicode_to_byte[c] = UInt8(b)
    end

    return byte_to_unicode, unicode_to_byte
end
