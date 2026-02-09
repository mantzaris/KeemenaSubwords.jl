using Base64: base64decode, base64encode

struct TiktokenTokenizer <: AbstractSubwordTokenizer
    id_to_bytes::Vector{Vector{UInt8}}
    id_to_rank::Vector{Int}
    key_to_id::Dict{String,Int}
    max_token_bytes::Int
    metadata::TokenizerMetadata
end

"""
Load a tiktoken encoding file (`*.tiktoken`).

The expected format is line-based:
`<base64_token_bytes><space><rank>`
where ranks are non-negative integers.
"""
function load_tiktoken(
    path::AbstractString;
    model_name::Union{Nothing,AbstractString}=nothing,
)::TiktokenTokenizer
    model_path = _resolve_tiktoken_path(path)
    sample = _sample_file_bytes(model_path)
    _looks_tiktoken_text_payload(sample) || throw(ArgumentError(
        "Invalid tiktoken file format at $model_path. " *
        "Expected line format '<base64_bytes> <rank>'. " *
        "Example: load_tiktoken(\"/path/to/encoding.tiktoken\")",
    ))
    ranked_tokens = _read_tiktoken_file(model_path)
    id_to_bytes = [item.bytes for item in ranked_tokens]
    id_to_rank = [item.rank for item in ranked_tokens]

    key_to_id = Dict{String,Int}()
    for (id, bytes) in enumerate(id_to_bytes)
        key = _token_key(bytes)
        haskey(key_to_id, key) && throw(ArgumentError("Duplicate byte token in tiktoken file: $model_path"))
        key_to_id[key] = id
    end

    max_token_bytes = maximum(length(bytes) for bytes in id_to_bytes)
    name = model_name === nothing ? basename(model_path) : String(model_name)
    metadata = TokenizerMetadata(:tiktoken, name, v"0.3.0", :none)

    return TiktokenTokenizer(id_to_bytes, id_to_rank, key_to_id, max_token_bytes, metadata)
end

(tokenizer::TiktokenTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

"""
Tokenize text into `b64:<...>` token pieces.
"""
function tokenize(tokenizer::TiktokenTokenizer, text::AbstractString)::Vector{String}
    ids = encode(tokenizer, text)
    return String[id_to_token(tokenizer, id) for id in ids]
end

"""
Encode text into tiktoken rank IDs (1-based in this package).
"""
function encode(
    tokenizer::TiktokenTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)::Vector{Int}
    add_special_tokens && throw(ArgumentError("TiktokenTokenizer does not define add_special_tokens behavior"))

    bytes = collect(codeunits(normalize_text(text)))
    n = length(bytes)
    ids = Int[]

    i = 1
    while i <= n
        best_id = 0
        best_len = 0
        best_rank = typemax(Int)
        max_len = min(tokenizer.max_token_bytes, n - i + 1)

        for len in 1:max_len
            key = _token_key(@view bytes[i:(i + len - 1)])
            id = get(tokenizer.key_to_id, key, 0)
            id == 0 && continue

            rank = tokenizer.id_to_rank[id]
            if len > best_len || (len == best_len && rank < best_rank)
                best_len = len
                best_id = id
                best_rank = rank
            end
        end

        if best_id == 0
            key = _token_key(UInt8[bytes[i]])
            best_id = get(tokenizer.key_to_id, key, 0)
            best_id == 0 && throw(ArgumentError("Could not tokenize byte 0x$(string(bytes[i], base=16, pad=2)) at position $i"))
            best_len = 1
        end

        push!(ids, best_id)
        i += best_len
    end

    return ids
end

"""
Decode tiktoken rank IDs to text.
"""
function decode(tokenizer::TiktokenTokenizer, ids::AbstractVector{Int})::String
    bytes = UInt8[]
    for id in ids
        (1 <= id <= length(tokenizer.id_to_bytes)) || throw(BoundsError(tokenizer.id_to_bytes, id))
        append!(bytes, tokenizer.id_to_bytes[id])
    end

    if isvalid(String, bytes)
        return String(bytes)
    end

    return _hex_escape(bytes)
end

"""
Forward token lookup.
"""
function token_to_id(tokenizer::TiktokenTokenizer, token::AbstractString)::Int
    token_str = String(token)
    key = startswith(token_str, "b64:") ? token_str[5:end] : _token_key(collect(codeunits(token_str)))
    haskey(tokenizer.key_to_id, key) || throw(ArgumentError("Unknown tiktoken token: $token"))
    return tokenizer.key_to_id[key]
end

"""
Reverse token lookup.
"""
function id_to_token(tokenizer::TiktokenTokenizer, id::Int)::String
    (1 <= id <= length(tokenizer.id_to_bytes)) || throw(BoundsError(tokenizer.id_to_bytes, id))
    return "b64:" * _token_key(tokenizer.id_to_bytes[id])
end

"""
Vocabulary size.
"""
vocab_size(tokenizer::TiktokenTokenizer)::Int = length(tokenizer.id_to_bytes)

"""
Special token IDs.
"""
special_tokens(tokenizer::TiktokenTokenizer)::Dict{Symbol,Int} = Dict{Symbol,Int}()

"""
Tokenizer metadata.
"""
model_info(tokenizer::TiktokenTokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)

"""
Unknown token ID.
"""
function unk_id(tokenizer::TiktokenTokenizer)::Int
    _ = tokenizer
    throw(ArgumentError("TiktokenTokenizer has no unk token"))
end

"""
Padding token ID if available.
"""
pad_id(tokenizer::TiktokenTokenizer)::Union{Int,Nothing} = nothing

"""
BOS token ID if available.
"""
bos_id(tokenizer::TiktokenTokenizer)::Union{Int,Nothing} = nothing

"""
EOS token ID if available.
"""
eos_id(tokenizer::TiktokenTokenizer)::Union{Int,Nothing} = nothing

function _resolve_tiktoken_path(path::AbstractString)::String
    if isdir(path)
        files = detect_tokenizer_files(path)
        if length(files.tiktoken_files) == 1
            return String(files.tiktoken_files[1])
        end

        model_candidate = joinpath(String(path), "tokenizer.model")
        if isfile(model_candidate) && _looks_tiktoken_text_payload(_sample_file_bytes(model_candidate))
            return model_candidate
        end

        throw(ArgumentError(
            "Expected one tiktoken file in directory: $path. " *
            "Provide exactly one *.tiktoken file, or tokenizer.model containing tiktoken text. " *
            "Example: load_tiktoken(\"/path/to/tokenizer.model\")",
        ))
    end

    isfile(path) || throw(ArgumentError(
        "Tiktoken file not found: $path. " *
        "Example: load_tiktoken(\"/path/to/encoding.tiktoken\")",
    ))

    lower = lowercase(String(path))
    if endswith(lower, ".tiktoken")
        return String(path)
    end

    _looks_tiktoken_text_payload(_sample_file_bytes(path)) || throw(ArgumentError(
        "Tiktoken path does not look like a valid tiktoken file: $path. " *
        "Expected .tiktoken or text tokenizer.model content.",
    ))
    return String(path)
end

function _read_tiktoken_file(
    path::AbstractString,
)::Vector{NamedTuple{(:rank, :bytes),Tuple{Int,Vector{UInt8}}}}
    pairs = Tuple{Int,Vector{UInt8}}[]

    open(path, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue
            startswith(line, "#") && continue

            fields = split(line)
            length(fields) == 2 || throw(ArgumentError("Invalid tiktoken row in $path: $line"))

            bytes = try
                base64decode(fields[1])
            catch
                throw(ArgumentError("Invalid base64 token in $path: $line"))
            end

            rank = tryparse(Int, fields[2])
            rank === nothing && throw(ArgumentError("Invalid rank in $path: $line"))
            rank >= 0 || throw(ArgumentError("Negative rank in $path: $line"))

            push!(pairs, (rank, bytes))
        end
    end

    isempty(pairs) && throw(ArgumentError("Tiktoken file is empty: $path"))

    sort!(pairs; by = first)

    for i in 2:length(pairs)
        pairs[i - 1][1] == pairs[i][1] && throw(ArgumentError("Duplicate rank $(pairs[i][1]) in $path"))
    end

    return [(rank = rank, bytes = bytes) for (rank, bytes) in pairs]
end

function _token_key(bytes::AbstractVector{UInt8})::String
    return base64encode(Vector{UInt8}(bytes))
end

function _hex_escape(bytes::Vector{UInt8})::String
    parts = String[]
    for b in bytes
        push!(parts, "\\x" * uppercase(string(b, base=16, pad=2)))
    end
    return join(parts)
end
