struct UnigramTokenizer <: AbstractSubwordTokenizer
    vocab::SubwordVocabulary
    logprobs::Vector{Float64}
    unk_token::String
    whitespace_marker::String
    metadata::TokenizerMetadata
end

"""
Load a Unigram tokenizer from `unigram.tsv` (file or directory).

Expected format (tab-separated):
`token<TAB>score[<TAB>special_symbol]`
"""
function load_unigram(
    path::AbstractString;
    whitespace_marker::AbstractString="",
    unk_token::AbstractString="<unk>",
    model_name::Union{Nothing,AbstractString}=nothing,
)::UnigramTokenizer
    model_path = _resolve_unigram_path(path)
    parsed = _read_unigram_tsv(model_path)

    tokens = parsed.tokens
    scores = parsed.scores
    special_map = parsed.special_map

    if !haskey(special_map, :unk)
        if String(unk_token) in tokens
            special_map[:unk] = String(unk_token)
        else
            pushfirst!(tokens, String(unk_token))
            pushfirst!(scores, -99.0)
            special_map = Dict(k => v for (k, v) in special_map)
            special_map[:unk] = String(unk_token)
        end
    end

    vocab = build_vocab(tokens; special_tokens=special_map)
    length(scores) == vocab_size(vocab) || throw(ArgumentError("Unigram score count must equal token count"))

    name = model_name === nothing ? basename(model_path) : String(model_name)
    metadata = TokenizerMetadata(:unigram, name, v"0.2.0", :none)

    return UnigramTokenizer(vocab, scores, String(unk_token), String(whitespace_marker), metadata)
end

(tokenizer::UnigramTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Tokenize text using deterministic Viterbi segmentation.
"""
function tokenize(tokenizer::UnigramTokenizer, text::AbstractString)::Vector{String}
    normalized = normalize_text(text)
    pieces = String[]

    for word in eachsplit(normalized)
        input = isempty(tokenizer.whitespace_marker) ? String(word) : string(tokenizer.whitespace_marker, String(word))
        append!(pieces, _viterbi_segment(tokenizer, input))
    end

    return pieces
end

"""
Encode text to unigram token IDs.
"""
function encode(
    tokenizer::UnigramTokenizer,
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
Decode unigram token IDs back to text.
"""
function decode(tokenizer::UnigramTokenizer, ids::AbstractVector{Int})::String
    kept = String[]
    for id in ids
        is_skippable_special_id(tokenizer.vocab, id) && continue
        push!(kept, id_to_token(tokenizer, id))
    end

    joined = join(kept, "")
    if isempty(tokenizer.whitespace_marker)
        return strip(joined)
    end

    with_spaces = replace(joined, tokenizer.whitespace_marker => " ")
    return strip(with_spaces)
end

"""
Forward token lookup.
"""
token_to_id(tokenizer::UnigramTokenizer, token::AbstractString)::Int = token_to_id(tokenizer.vocab, token)

"""
Reverse token lookup.
"""
id_to_token(tokenizer::UnigramTokenizer, id::Int)::String = id_to_token(tokenizer.vocab, id)

"""
Vocabulary size.
"""
vocab_size(tokenizer::UnigramTokenizer)::Int = vocab_size(tokenizer.vocab)

"""
Special token IDs.
"""
special_tokens(tokenizer::UnigramTokenizer)::Dict{Symbol,Int} = copy(tokenizer.vocab.special_token_ids)

"""
Tokenizer metadata.
"""
model_info(tokenizer::UnigramTokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
unk_id(tokenizer::UnigramTokenizer)::Int = unk_id(tokenizer.vocab)

"""
Padding token ID if available.
"""
pad_id(tokenizer::UnigramTokenizer)::Union{Int,Nothing} = pad_id(tokenizer.vocab)

"""
BOS token ID if available.
"""
bos_id(tokenizer::UnigramTokenizer)::Union{Int,Nothing} = bos_id(tokenizer.vocab)

"""
EOS token ID if available.
"""
eos_id(tokenizer::UnigramTokenizer)::Union{Int,Nothing} = eos_id(tokenizer.vocab)

function _resolve_unigram_path(path::AbstractString)::String
    if isdir(path)
        candidate = joinpath(path, "unigram.tsv")
        isfile(candidate) || throw(ArgumentError("Missing unigram.tsv in directory: $path"))
        return candidate
    end

    isfile(path) || throw(ArgumentError("Unigram model path does not exist: $path"))
    return String(path)
end

function _read_unigram_tsv(path::AbstractString)::NamedTuple{(:tokens, :scores, :special_map),Tuple{Vector{String},Vector{Float64},Dict{Symbol,String}}}
    tokens = String[]
    scores = Float64[]
    special_map = Dict{Symbol,String}()

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue

            fields = split(line, '\t')
            length(fields) >= 2 || throw(ArgumentError("Invalid unigram row in $path: $line"))

            token = fields[1]
            score = tryparse(Float64, fields[2])
            score === nothing && throw(ArgumentError("Invalid unigram score in $path: $line"))

            push!(tokens, token)
            push!(scores, score)

            if length(fields) >= 3
                symbol = Symbol(strip(fields[3]))
                isempty(String(symbol)) || (special_map[symbol] = token)
            end
        end
    end

    isempty(tokens) && throw(ArgumentError("Unigram file is empty: $path"))
    return (tokens=tokens, scores=scores, special_map=special_map)
end

function _viterbi_segment(tokenizer::UnigramTokenizer, input::String)::Vector{String}
    chars = collect(input)
    n = length(chars)
    n == 0 && return String[]

    best = fill(-Inf, n + 1)
    back = fill(0, n + 1)
    back_token = fill("", n + 1)
    best[1] = 0.0

    for i in 1:n
        if !isfinite(best[i])
            continue
        end

        for j in i:n
            token = String(chars[i:j])
            id = get(tokenizer.vocab.token_to_id, token, 0)
            id == 0 && continue

            score = best[i] + tokenizer.logprobs[id]
            if score > best[j + 1]
                best[j + 1] = score
                back[j + 1] = i
                back_token[j + 1] = token
            end
        end
    end

    if !isfinite(best[n + 1])
        return [tokenizer.unk_token]
    end

    out = String[]
    pos = n + 1
    while pos > 1
        tok = back_token[pos]
        isempty(tok) && return [tokenizer.unk_token]
        push!(out, tok)
        pos = back[pos]
    end

    reverse!(out)
    return out
end
