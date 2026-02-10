"""
Load a Hugging Face `tokenizer.json` tokenizer in pure Julia.

Examples:
- `load_hf_tokenizer_json("/path/to/tokenizer.json")`
- `load_hf_tokenizer_json("/path/to/model_dir")` (expects `tokenizer.json` inside)
"""
function load_hf_tokenizer_json(
    path::AbstractString;
    model_name::Union{Nothing,AbstractString}=nothing,
)::HuggingFaceJSONTokenizer
    spec = parse_hf_tokenizer_json(path)
    name = model_name === nothing ? basename(spec.source_path) : String(model_name)
    model = _with_hf_added_tokens(spec.model, spec.added_token_ids)
    model = _with_hf_bytelevel_flag(model, spec)
    base = _build_hf_json_base_tokenizer(model; model_name=name)

    token_special_ids = copy(spec.special_token_ids)
    special_ids = _derive_symbol_special_ids(base, token_special_ids)
    metadata = TokenizerMetadata(:hf_tokenizer_json, name, v"0.4.0", :hf_json)

    return HuggingFaceJSONTokenizer(
        base,
        spec.normalizer,
        spec.pretokenizer,
        spec.postprocessor,
        spec.decoder,
        spec.added_token_ids,
        token_special_ids,
        special_ids,
        metadata,
        spec.source_path,
    )
end

function _with_hf_added_tokens(
    model::HFBPEModelSpec,
    added_token_ids::Dict{String,Int},
)::HFBPEModelSpec
    vocab = _merge_added_tokens(model.vocab, added_token_ids)
    return HFBPEModelSpec(vocab, model.merges, model.unk_token, model.byte_level, model.end_of_word_suffix)
end

function _with_hf_added_tokens(
    model::HFWordPieceModelSpec,
    added_token_ids::Dict{String,Int},
)::HFWordPieceModelSpec
    vocab = _merge_added_tokens(model.vocab, added_token_ids)
    return HFWordPieceModelSpec(vocab, model.unk_token, model.continuation_prefix)
end

function _with_hf_added_tokens(
    model::HFUnigramModelSpec,
    added_token_ids::Dict{String,Int},
)::HFUnigramModelSpec
    vocab, scores = _merge_added_tokens_with_scores(model.vocab, model.scores, added_token_ids)
    unk_id = min(model.unk_id, length(vocab))
    return HFUnigramModelSpec(vocab, scores, unk_id)
end

function _with_hf_bytelevel_flag(model::HFJSONModelSpec, spec::HFJSONSpec)::HFJSONModelSpec
    model isa HFBPEModelSpec || return model
    byte_level = _uses_hf_bytelevel(spec.pretokenizer) || _uses_hf_bytelevel(spec.decoder)
    return HFBPEModelSpec(
        model.vocab,
        model.merges,
        model.unk_token,
        byte_level,
        model.end_of_word_suffix,
    )
end

_uses_hf_bytelevel(::HFJSONPreTokenizer)::Bool = false
_uses_hf_bytelevel(::HFJSONDecoder)::Bool = false
_uses_hf_bytelevel(::HFByteLevelPreTokenizer)::Bool = true
_uses_hf_bytelevel(::HFByteLevelDecoder)::Bool = true

function _uses_hf_bytelevel(pre::HFSequencePreTokenizer)::Bool
    return any(_uses_hf_bytelevel, pre.items)
end

function _uses_hf_bytelevel(dec::HFSequenceDecoder)::Bool
    return any(_uses_hf_bytelevel, dec.items)
end

function _merge_added_tokens(tokens::Vector{String}, added::Dict{String,Int})::Vector{String}
    isempty(added) && return copy(tokens)

    target_len = max(length(tokens), maximum(values(added)))
    merged = Vector{Union{Nothing,String}}(undef, target_len)
    fill!(merged, nothing)

    for (i, token) in enumerate(tokens)
        merged[i] = token
    end

    for (token, id) in added
        id < 1 && continue
        if id > length(merged)
            old_len = length(merged)
            resize!(merged, id)
            for i in (old_len + 1):id
                merged[i] = nothing
            end
        end

        existing = merged[id]
        if existing === nothing || existing == token
            merged[id] = token
        end
    end

    for i in eachindex(merged)
        if merged[i] === nothing
            merged[i] = "<unused_$(i)>"
        end
    end

    return String[token::String for token in merged]
end

function _merge_added_tokens_with_scores(
    tokens::Vector{String},
    scores::Vector{Float64},
    added::Dict{String,Int},
)::Tuple{Vector{String},Vector{Float64}}
    merged_tokens = _merge_added_tokens(tokens, added)
    merged_scores = fill(-99.0, length(merged_tokens))

    for i in 1:min(length(tokens), length(merged_scores))
        merged_scores[i] = scores[i]
    end

    return (merged_tokens, merged_scores)
end

function _build_hf_json_base_tokenizer(
    model::HFBPEModelSpec;
    model_name::String,
)::AbstractSubwordTokenizer
    special_map = detect_special_tokens(model.vocab, model.unk_token)
    vocab = build_vocab(model.vocab; special_tokens=special_map)

    pair_ranks = Dict{Tuple{String,String},Int}()
    for (i, pair) in enumerate(model.merges)
        pair_ranks[pair] = i
    end

    end_marker = model.end_of_word_suffix
    base_meta = TokenizerMetadata(:bpe_gpt2, model_name, v"0.4.0", :none)
    bpe = BPETokenizer(vocab, pair_ranks, model.unk_token, end_marker, base_meta)

    if model.byte_level
        b2u, u2b = _byte_unicode_tables()
        meta = TokenizerMetadata(:bytebpe, model_name, v"0.4.0", :none)
        return ByteBPETokenizer(bpe, b2u, u2b, meta)
    end

    return bpe
end

function _build_hf_json_base_tokenizer(
    model::HFWordPieceModelSpec;
    model_name::String,
)::AbstractSubwordTokenizer
    special_map = detect_special_tokens(model.vocab, model.unk_token)
    vocab = build_vocab(model.vocab; special_tokens=special_map)
    meta = TokenizerMetadata(:wordpiece, model_name, v"0.4.0", :none)
    return WordPieceTokenizer(vocab, model.continuation_prefix, model.unk_token, meta)
end

function _build_hf_json_base_tokenizer(
    model::HFUnigramModelSpec;
    model_name::String,
)::AbstractSubwordTokenizer
    unk_token = model.vocab[model.unk_id]
    special_map = detect_special_tokens(model.vocab, unk_token)
    vocab = build_vocab(model.vocab; special_tokens=special_map)
    meta = TokenizerMetadata(:unigram, model_name, v"0.4.0", :none)
    return UnigramTokenizer(vocab, copy(model.scores), unk_token, "", meta)
end

function _derive_symbol_special_ids(
    base::AbstractSubwordTokenizer,
    token_special_ids::Dict{String,Int},
)::Dict{Symbol,Int}
    merged = Dict{Symbol,Int}()
    for (k, v) in special_tokens(base)
        merged[k] = v
    end

    for (token, id) in token_special_ids
        symbol = _special_symbol_for_token(token)
        symbol === nothing && continue
        merged[symbol] = id
    end

    return merged
end

function _special_symbol_for_token(token::String)::Union{Nothing,Symbol}
    lower = lowercase(token)

    if token in ("<unk>", "<UNK>", "[UNK]")
        return :unk
    elseif token in ("[PAD]", "<pad>")
        return :pad
    elseif token in ("<s>", "[BOS]", "<|begin_of_text|>", "<|bos|>")
        return :bos
    elseif token in ("</s>", "[EOS]", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>")
        return :eos
    elseif token == "[CLS]"
        return :cls
    elseif token == "[SEP]"
        return :sep
    elseif occursin("pad", lower)
        return :pad
    elseif occursin("bos", lower)
        return :bos
    elseif occursin("eos", lower) || occursin("eot", lower)
        return :eos
    elseif occursin("unk", lower)
        return :unk
    end

    return nothing
end
