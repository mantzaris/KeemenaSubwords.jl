(tokenizer::HuggingFaceJSONTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

function tokenize(tokenizer::HuggingFaceJSONTokenizer, text::AbstractString)::Vector{String}
    normalized = _apply_hf_normalizer(tokenizer.normalizer, String(text))
    pretokenized = _apply_hf_pretokenizer(tokenizer.pretokenizer, normalized)
    return tokenize(tokenizer.base, pretokenized)
end

function encode(
    tokenizer::HuggingFaceJSONTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)::Vector{Int}
    normalized = _apply_hf_normalizer(tokenizer.normalizer, String(text))
    pretokenized = _apply_hf_pretokenizer(tokenizer.pretokenizer, normalized)
    ids = encode(tokenizer.base, pretokenized; add_special_tokens=false)
    add_special_tokens || return ids
    return _apply_hf_postprocessor(tokenizer.postprocessor, ids, tokenizer)
end

function decode(tokenizer::HuggingFaceJSONTokenizer, ids::AbstractVector{Int})::String
    text = decode(tokenizer.base, ids)
    return _apply_hf_decoder(tokenizer.decoder, text, tokenizer)
end

token_to_id(tokenizer::HuggingFaceJSONTokenizer, token::AbstractString)::Int = token_to_id(tokenizer.base, token)
id_to_token(tokenizer::HuggingFaceJSONTokenizer, id::Int)::String = id_to_token(tokenizer.base, id)
vocab_size(tokenizer::HuggingFaceJSONTokenizer)::Int = vocab_size(tokenizer.base)
model_info(tokenizer::HuggingFaceJSONTokenizer)::NamedTuple = metadata_namedtuple(tokenizer.metadata)
unk_id(tokenizer::HuggingFaceJSONTokenizer)::Int = unk_id(tokenizer.base)
pad_id(tokenizer::HuggingFaceJSONTokenizer)::Union{Int,Nothing} = get(tokenizer.special_token_ids, :pad, pad_id(tokenizer.base))
bos_id(tokenizer::HuggingFaceJSONTokenizer)::Union{Int,Nothing} = get(tokenizer.special_token_ids, :bos, bos_id(tokenizer.base))
eos_id(tokenizer::HuggingFaceJSONTokenizer)::Union{Int,Nothing} = get(tokenizer.special_token_ids, :eos, eos_id(tokenizer.base))

function special_tokens(tokenizer::HuggingFaceJSONTokenizer)::Dict{Symbol,Int}
    return copy(tokenizer.special_token_ids)
end

function _apply_hf_normalizer(::HFNoopNormalizer, text::String)::String
    return text
end

function _apply_hf_normalizer(::HFLowercaseNormalizer, text::String)::String
    return lowercase(text)
end

function _apply_hf_normalizer(::HFNFCNormalizer, text::String)::String
    return Base.Unicode.normalize(text, :NFC)
end

function _apply_hf_normalizer(::HFNFKCNormalizer, text::String)::String
    return Base.Unicode.normalize(text, :NFKC)
end

function _apply_hf_normalizer(normalizer::HFSequenceNormalizer, text::String)::String
    out = text
    for item in normalizer.items
        out = _apply_hf_normalizer(item, out)
    end
    return out
end

function _apply_hf_pretokenizer(::HFNoopPreTokenizer, text::String)::String
    return text
end

function _apply_hf_pretokenizer(::HFWhitespacePreTokenizer, text::String)::String
    return text
end

function _apply_hf_pretokenizer(::HFByteLevelPreTokenizer, text::String)::String
    return text
end

function _apply_hf_pretokenizer(pre::HFMetaspacePreTokenizer, text::String)::String
    working = text
    if pre.add_prefix_space && !isempty(working) && !startswith(working, " ")
        working = " " * working
    end
    return replace(working, " " => pre.replacement)
end

function _apply_hf_pretokenizer(pre::HFSplitPreTokenizer, text::String)::String
    if pre.behavior == :isolated
        matches = String[]
        for m in eachmatch(pre.pattern, text)
            push!(matches, m.match)
        end
        return isempty(matches) ? text : join(matches, " ")
    elseif pre.behavior == :removed
        return replace(text, pre.pattern => " ")
    end

    return text
end

function _apply_hf_pretokenizer(pre::HFSequencePreTokenizer, text::String)::String
    out = text
    for item in pre.items
        out = _apply_hf_pretokenizer(item, out)
    end
    return out
end

function _apply_hf_postprocessor(
    ::HFNoopPostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    bos = bos_id(tokenizer)
    eos = eos_id(tokenizer)
    if bos === nothing && eos === nothing
        return ids
    end

    out = Int[]
    bos !== nothing && push!(out, bos)
    append!(out, ids)
    eos !== nothing && push!(out, eos)
    return out
end

function _apply_hf_postprocessor(
    ::HFByteLevelPostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    _ = tokenizer
    return ids
end

function _apply_hf_postprocessor(
    post::HFTemplateProcessingPostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    template = isempty(post.single) ? String["\$A"] : post.single
    out = Int[]

    for item in template
        if _is_hf_sequence_a(item)
            append!(out, ids)
        elseif _is_hf_sequence_b(item)
            throw(ArgumentError("TemplateProcessing item '$item' requires pair encoding, which is not supported in encode(text)"))
        else
            special_id = _resolve_hf_template_special_id(item, post, tokenizer)
            push!(out, special_id)
        end
    end

    return out
end

function _apply_hf_postprocessor(
    post::HFSequencePostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    out = ids
    for item in post.items
        out = _apply_hf_postprocessor(item, out, tokenizer)
    end
    return out
end

function _resolve_hf_template_special_id(
    token::String,
    post::HFTemplateProcessingPostProcessor,
    tokenizer::HuggingFaceJSONTokenizer,
)::Int
    if haskey(post.special_tokens, token)
        return post.special_tokens[token]
    elseif haskey(tokenizer.token_special_ids, token)
        return tokenizer.token_special_ids[token]
    end

    return token_to_id(tokenizer, token)
end

function _is_hf_sequence_a(item::String)::Bool
    startswith(item, "\$A") || startswith(item, "\$0")
end

function _is_hf_sequence_b(item::String)::Bool
    startswith(item, "\$B") || startswith(item, "\$1")
end

function _apply_hf_decoder(
    ::HFNoopDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = tokenizer
    return text
end

function _apply_hf_decoder(
    ::HFByteLevelDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = tokenizer
    return text
end

function _apply_hf_decoder(
    decoder::HFWordPieceDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = decoder
    _ = tokenizer
    return text
end

function _apply_hf_decoder(
    decoder::HFMetaspaceDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = tokenizer
    return strip(replace(text, decoder.replacement => " "))
end

function _apply_hf_decoder(
    decoder::HFSequenceDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    out = text
    for item in decoder.items
        out = _apply_hf_decoder(item, out, tokenizer)
    end
    return out
end
