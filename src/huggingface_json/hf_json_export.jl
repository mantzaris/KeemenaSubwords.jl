using JSON3

"""
Export any supported tokenizer to Hugging Face `tokenizer.json`.

All exported ids are 0-based to match the HF JSON contract.
"""
function export_hf_tokenizer_json(
    tokenizer::AbstractSubwordTokenizer,
    outpath::AbstractString,
)::Nothing
    payload = _hf_export_payload(tokenizer)
    open(outpath, "w") do io
        JSON3.write(io, payload)
    end
    return nothing
end

function _hf_export_payload(tokenizer::AbstractSubwordTokenizer)
    return (
        version = "1.0",
        truncation = nothing,
        padding = nothing,
        added_tokens = _hf_export_added_tokens(tokenizer),
        normalizer = _hf_export_normalizer(tokenizer),
        pre_tokenizer = _hf_export_pretokenizer(tokenizer),
        post_processor = _hf_export_postprocessor(tokenizer),
        decoder = _hf_export_decoder(tokenizer),
        model = _hf_export_model(tokenizer),
    )
end

function _hf_export_payload(tokenizer::HuggingFaceJSONTokenizer)
    return (
        version = "1.0",
        truncation = _hf_export_truncation(tokenizer.truncation),
        padding = _hf_export_padding(tokenizer.padding),
        added_tokens = _hf_export_added_tokens(tokenizer),
        normalizer = _hf_export_normalizer(tokenizer),
        pre_tokenizer = _hf_export_pretokenizer(tokenizer),
        post_processor = _hf_export_postprocessor(tokenizer),
        decoder = _hf_export_decoder(tokenizer),
        model = _hf_export_model(tokenizer),
    )
end

function _hf_export_added_tokens(tokenizer::AbstractSubwordTokenizer)::Vector{Any}
    special_ids = unique!(sort!(collect(values(special_tokens(tokenizer)))))
    added = Any[]

    for id in special_ids
        push!(added, _hf_export_added_token(
            id_to_token(tokenizer, id),
            id;
            special=true,
            single_word=false,
            lstrip=false,
            rstrip=false,
            normalized=false,
        ))
    end

    return added
end

function _hf_export_added_tokens(tokenizer::HuggingFaceJSONTokenizer)::Vector{Any}
    ordered = sort(copy(tokenizer.added_tokens); by=token -> (token.id, token.content))
    return Any[_hf_export_added_token(
        token.content,
        token.id;
        special=token.special,
        single_word=token.single_word,
        lstrip=token.lstrip,
        rstrip=token.rstrip,
        normalized=token.normalized,
    ) for token in ordered]
end

function _hf_export_added_token(
    token::String,
    id::Int;
    special::Bool,
    single_word::Bool,
    lstrip::Bool,
    rstrip::Bool,
    normalized::Bool,
)::Dict{String,Any}
    return Dict(
        "id" => id - 1,
        "content" => token,
        "special" => special,
        "single_word" => single_word,
        "lstrip" => lstrip,
        "rstrip" => rstrip,
        "normalized" => normalized,
    )
end

_hf_export_normalizer(::AbstractSubwordTokenizer) = nothing
_hf_export_normalizer(tokenizer::HuggingFaceJSONTokenizer) = _hf_export_normalizer(tokenizer.normalizer)
_hf_export_normalizer(::HFNoopNormalizer) = nothing
_hf_export_normalizer(::HFLowercaseNormalizer) = Dict("type" => "Lowercase")
function _hf_export_normalizer(normalizer::HFBertNormalizer)
    return Dict(
        "type" => "BertNormalizer",
        "clean_text" => normalizer.clean_text,
        "handle_chinese_chars" => normalizer.handle_chinese_chars,
        "strip_accents" => normalizer.strip_accents,
        "lowercase" => normalizer.lowercase,
    )
end
_hf_export_normalizer(::HFNFCNormalizer) = Dict("type" => "NFC")
_hf_export_normalizer(::HFNFKCNormalizer) = Dict("type" => "NFKC")
_hf_export_normalizer(::HFNFDNormalizer) = Dict("type" => "NFD")
_hf_export_normalizer(::HFStripAccentsNormalizer) = Dict("type" => "StripAccents")

function _hf_export_normalizer(normalizer::HFReplaceNormalizer)
    return Dict(
        "type" => "Replace",
        "pattern" => Dict("Regex" => normalizer.pattern.pattern),
        "content" => normalizer.replacement,
    )
end

function _hf_export_normalizer(normalizer::HFPrependNormalizer)
    return Dict(
        "type" => "Prepend",
        "prepend" => normalizer.prefix,
    )
end

function _hf_export_normalizer(normalizer::HFSequenceNormalizer)
    return Dict(
        "type" => "Sequence",
        "normalizers" => Any[_hf_export_normalizer(item) for item in normalizer.items],
    )
end

_hf_export_pretokenizer(tokenizer::HuggingFaceJSONTokenizer) = _hf_export_pretokenizer(tokenizer.pretokenizer)
_hf_export_pretokenizer(::HFNoopPreTokenizer) = nothing
_hf_export_pretokenizer(::WordPieceTokenizer) = Dict("type" => "WhitespaceSplit")
_hf_export_pretokenizer(::BPETokenizer) = Dict("type" => "WhitespaceSplit")

function _hf_export_bytelevel_dict(
    add_prefix_space::Bool,
    trim_offsets::Bool,
    use_regex::Bool,
)::Dict{String,Any}
    return Dict(
        "type" => "ByteLevel",
        "add_prefix_space" => add_prefix_space,
        "trim_offsets" => trim_offsets,
        "use_regex" => use_regex,
    )
end

function _hf_export_pretokenizer(::ByteBPETokenizer)
    return Dict(
        "type" => "Sequence",
        "pretokenizers" => Any[
            Dict("type" => "WhitespaceSplit"),
            _hf_export_bytelevel_dict(false, false, false),
        ],
    )
end

function _hf_export_pretokenizer(tokenizer::UnigramTokenizer)
    if isempty(tokenizer.whitespace_marker)
        return Dict("type" => "WhitespaceSplit")
    end
    return _hf_export_metaspace_pretokenizer(tokenizer.whitespace_marker)
end

function _hf_export_pretokenizer(tokenizer::SentencePieceTokenizer)
    return _hf_export_metaspace_pretokenizer(tokenizer.whitespace_marker)
end

function _hf_export_pretokenizer(pre::HFByteLevelPreTokenizer)
    return _hf_export_bytelevel_dict(
        pre.add_prefix_space,
        pre.trim_offsets,
        pre.use_regex,
    )
end
_hf_export_pretokenizer(::HFBertPreTokenizer) = Dict("type" => "BertPreTokenizer")

_hf_export_pretokenizer(::HFWhitespacePreTokenizer) = Dict("type" => "WhitespaceSplit")

function _hf_export_pretokenizer(pre::HFMetaspacePreTokenizer)
    return Dict(
        "type" => "Metaspace",
        "replacement" => pre.replacement,
        "add_prefix_space" => pre.add_prefix_space,
    )
end

function _hf_export_pretokenizer(pre::HFSplitPreTokenizer)
    return Dict(
        "type" => "Split",
        "pattern" => Dict("Regex" => pre.pattern.pattern),
        "behavior" => string(pre.behavior),
        "invert" => false,
    )
end

function _hf_export_pretokenizer(pre::HFDigitsPreTokenizer)
    return Dict(
        "type" => "Digits",
        "individual_digits" => pre.individual_digits,
    )
end

function _hf_export_pretokenizer(pre::HFPunctuationPreTokenizer)
    return Dict(
        "type" => "Punctuation",
        "behavior" => string(pre.behavior),
    )
end

function _hf_export_pretokenizer(pre::HFSequencePreTokenizer)
    return Dict(
        "type" => "Sequence",
        "pretokenizers" => Any[_hf_export_pretokenizer(item) for item in pre.items],
    )
end

function _hf_export_metaspace_pretokenizer(marker::String)
    return Dict(
        "type" => "Sequence",
        "pretokenizers" => Any[
            Dict("type" => "WhitespaceSplit"),
            Dict(
                "type" => "Metaspace",
                "replacement" => marker,
                "add_prefix_space" => true,
            ),
        ],
    )
end

_hf_export_postprocessor(tokenizer::HuggingFaceJSONTokenizer) = _hf_export_postprocessor(tokenizer.postprocessor)

function _hf_export_postprocessor(tokenizer::WordPieceTokenizer)
    specials = special_tokens(tokenizer)
    cls_id = get(specials, :cls, nothing)
    sep_id = get(specials, :sep, nothing)

    if cls_id === nothing || sep_id === nothing
        return nothing
    end

    return Dict(
        "type" => "BertProcessing",
        "cls" => Any[id_to_token(tokenizer, cls_id), cls_id - 1],
        "sep" => Any[id_to_token(tokenizer, sep_id), sep_id - 1],
    )
end

function _hf_export_postprocessor(tokenizer::AbstractSubwordTokenizer)
    return _hf_export_bos_eos_postprocessor(tokenizer)
end

_hf_export_postprocessor(::HFNoopPostProcessor) = nothing
function _hf_export_postprocessor(post::HFByteLevelPostProcessor)
    return Dict(
        "type" => "ByteLevel",
        "add_prefix_space" => post.add_prefix_space,
        "trim_offsets" => post.trim_offsets,
    )
end

function _hf_export_postprocessor(post::HFBertProcessingPostProcessor)
    return Dict(
        "type" => "BertProcessing",
        "cls" => Any[post.cls_token, post.cls_id - 1],
        "sep" => Any[post.sep_token, post.sep_id - 1],
    )
end

function _hf_export_postprocessor(post::HFRobertaProcessingPostProcessor)
    return Dict(
        "type" => "RobertaProcessing",
        "cls" => Any[post.cls_token, post.cls_id - 1],
        "sep" => Any[post.sep_token, post.sep_id - 1],
    )
end

function _hf_export_postprocessor(post::HFTemplateProcessingPostProcessor)
    pairs = collect(post.special_tokens)
    sort!(pairs; by=entry -> (entry[2], entry[1]))
    ordered_entries = Tuple{String,Int}[(String(token), id) for (token, id) in pairs]

    return Dict(
        "type" => "TemplateProcessing",
        "single" => _hf_export_template_items(post.single; is_pair=false),
        "pair" => _hf_export_template_items(post.pair; is_pair=true),
        "special_tokens" => _hf_export_template_special_map(ordered_entries),
    )
end

function _hf_export_postprocessor(post::HFSequencePostProcessor)
    return Dict(
        "type" => "Sequence",
        "processors" => Any[_hf_export_postprocessor(item) for item in post.items],
    )
end

function _hf_export_bos_eos_postprocessor(tokenizer::AbstractSubwordTokenizer)
    specials = special_tokens(tokenizer)
    bos_id = get(specials, :bos, nothing)
    eos_id = get(specials, :eos, nothing)

    if bos_id === nothing && eos_id === nothing
        return nothing
    end

    single = String[]
    pair = String[]
    special_entries = Tuple{String,Int}[]
    seen_ids = Set{Int}()

    if bos_id !== nothing
        bos_token = id_to_token(tokenizer, bos_id)
        push!(single, bos_token)
        push!(pair, bos_token)
        if bos_id ∉ seen_ids
            push!(special_entries, (bos_token, bos_id))
            push!(seen_ids, bos_id)
        end
    end

    push!(single, "\$A")
    push!(pair, "\$A")

    if eos_id !== nothing
        eos_token = id_to_token(tokenizer, eos_id)
        push!(single, eos_token)
        push!(pair, eos_token)
        if eos_id ∉ seen_ids
            push!(special_entries, (eos_token, eos_id))
            push!(seen_ids, eos_id)
        end
    end

    push!(pair, "\$B")
    if eos_id !== nothing
        push!(pair, id_to_token(tokenizer, eos_id))
    end

    return Dict(
        "type" => "TemplateProcessing",
        "single" => _hf_export_template_items(single; is_pair=false),
        "pair" => _hf_export_template_items(pair; is_pair=true),
        "special_tokens" => _hf_export_template_special_map(special_entries),
    )
end

function _hf_export_template_special_entry(token::String, id::Int)::Dict{String,Any}
    return Dict(
        "id" => token,
        "ids" => Any[id - 1],
        "tokens" => Any[token],
    )
end

function _hf_export_template_items(
    items::Vector{String};
    is_pair::Bool,
)::Vector{Any}
    exported = Any[]
    current_type_id = 0

    for item in items
        if item == "\$A"
            push!(exported, Dict(
                "Sequence" => Dict(
                    "id" => "A",
                    "type_id" => 0,
                ),
            ))
            continue
        elseif item == "\$B"
            current_type_id = 1
            push!(exported, Dict(
                "Sequence" => Dict(
                    "id" => "B",
                    "type_id" => 1,
                ),
            ))
            continue
        end

        type_id = is_pair ? current_type_id : 0
        push!(exported, Dict(
            "SpecialToken" => Dict(
                "id" => item,
                "type_id" => type_id,
            ),
        ))
    end

    return exported
end

function _hf_export_template_special_map(
    entries::Vector{Tuple{String,Int}},
)::Dict{String,Any}
    special_map = Dict{String,Any}()
    for (token, id) in entries
        special_map[token] = _hf_export_template_special_entry(token, id)
    end
    return special_map
end

_hf_export_decoder(tokenizer::HuggingFaceJSONTokenizer) = _hf_export_decoder(tokenizer.decoder)
_hf_export_decoder(::HFNoopDecoder) = nothing
function _hf_export_decoder(decoder::HFByteLevelDecoder)
    return _hf_export_bytelevel_dict(
        decoder.add_prefix_space,
        decoder.trim_offsets,
        decoder.use_regex,
    )
end
_hf_export_decoder(decoder::HFWordPieceDecoder) = Dict("type" => "WordPiece", "prefix" => decoder.prefix)
_hf_export_decoder(decoder::HFBPEDecoder) = Dict("type" => "BPEDecoder", "suffix" => decoder.suffix)
function _hf_export_decoder(decoder::HFMetaspaceDecoder)
    return Dict(
        "type" => "Metaspace",
        "replacement" => decoder.replacement,
        "add_prefix_space" => decoder.add_prefix_space,
    )
end

function _hf_export_decoder(decoder::HFSequenceDecoder)
    return Dict(
        "type" => "Sequence",
        "decoders" => Any[_hf_export_decoder(item) for item in decoder.items],
    )
end

function _hf_export_decoder(tokenizer::WordPieceTokenizer)
    return Dict(
        "type" => "WordPiece",
        "prefix" => tokenizer.continuation_prefix,
    )
end

function _hf_export_decoder(tokenizer::BPETokenizer)
    marker = tokenizer.end_of_word_marker
    marker === nothing && return nothing
    return Dict(
        "type" => "BPEDecoder",
        "suffix" => marker,
    )
end

function _hf_export_decoder(tokenizer::ByteBPETokenizer)
    parts = Any[]
    marker = tokenizer.base.end_of_word_marker

    if marker !== nothing
        push!(parts, Dict("type" => "BPEDecoder", "suffix" => marker))
    end
    push!(parts, _hf_export_bytelevel_dict(false, false, false))

    if length(parts) == 1
        return only(parts)
    end

    return Dict(
        "type" => "Sequence",
        "decoders" => parts,
    )
end

function _hf_export_decoder(tokenizer::UnigramTokenizer)
    if isempty(tokenizer.whitespace_marker)
        return nothing
    end

    return Dict(
        "type" => "Metaspace",
        "replacement" => tokenizer.whitespace_marker,
        "add_prefix_space" => true,
    )
end

function _hf_export_decoder(tokenizer::SentencePieceTokenizer)
    if isempty(tokenizer.whitespace_marker)
        return nothing
    end

    return Dict(
        "type" => "Metaspace",
        "replacement" => tokenizer.whitespace_marker,
        "add_prefix_space" => true,
    )
end

_hf_export_model(tokenizer::HuggingFaceJSONTokenizer) = _hf_export_model(tokenizer.model)

function _hf_export_model(tokenizer::WordPieceTokenizer)
    return Dict(
        "type" => "WordPiece",
        "vocab" => _hf_export_vocab_map(tokenizer.vocab.id_to_token),
        "unk_token" => tokenizer.unk_token,
        "continuing_subword_prefix" => tokenizer.continuation_prefix,
        "max_input_chars_per_word" => tokenizer.max_input_chars_per_word,
    )
end

function _hf_export_model(tokenizer::BPETokenizer)
    return Dict(
        "type" => "BPE",
        "vocab" => _hf_export_vocab_map(tokenizer.vocab.id_to_token),
        "merges" => _hf_export_merges(_hf_export_sorted_pairs(tokenizer.pair_ranks)),
        "unk_token" => tokenizer.unk_token,
        "continuing_subword_prefix" => nothing,
        "end_of_word_suffix" => tokenizer.end_of_word_marker,
        "fuse_unk" => false,
        "byte_fallback" => false,
        "dropout" => nothing,
    )
end

_hf_export_model(tokenizer::ByteBPETokenizer) = _hf_export_model(tokenizer.base)

function _hf_export_model(tokenizer::UnigramTokenizer)
    rows = Any[]
    for id in 1:length(tokenizer.vocab.id_to_token)
        push!(rows, Any[tokenizer.vocab.id_to_token[id], tokenizer.logprobs[id]])
    end

    return Dict(
        "type" => "Unigram",
        "unk_id" => unk_id(tokenizer) - 1,
        "byte_fallback" => false,
        "vocab" => rows,
    )
end

function _hf_export_model(tokenizer::SentencePieceTokenizer)
    inner = tokenizer.inner

    if inner isa UnigramTokenizer
        return _hf_export_model(inner)
    elseif inner isa BPETokenizer
        return _hf_export_model(inner)
    end

    throw(ArgumentError("Cannot export SentencePiece inner tokenizer $(typeof(inner)) as HF tokenizer.json"))
end

function _hf_export_model(model::HFWordPieceModelSpec)
    return Dict(
        "type" => "WordPiece",
        "vocab" => _hf_export_vocab_map(model.vocab),
        "unk_token" => model.unk_token,
        "continuing_subword_prefix" => model.continuation_prefix,
        "max_input_chars_per_word" => model.max_input_chars_per_word,
    )
end

function _hf_export_model(model::HFBPEModelSpec)
    return Dict(
        "type" => "BPE",
        "vocab" => _hf_export_vocab_map(model.vocab),
        "merges" => _hf_export_merges(model.merges),
        "unk_token" => model.unk_token,
        "continuing_subword_prefix" => model.continuing_subword_prefix,
        "end_of_word_suffix" => model.end_of_word_suffix,
        "fuse_unk" => model.fuse_unk,
        "byte_fallback" => model.byte_fallback,
        "dropout" => model.dropout,
    )
end

function _hf_export_model(model::HFUnigramModelSpec)
    rows = Any[]
    for i in 1:length(model.vocab)
        push!(rows, Any[model.vocab[i], model.scores[i]])
    end

    return Dict(
        "type" => "Unigram",
        "unk_id" => model.unk_id - 1,
        "byte_fallback" => model.byte_fallback,
        "vocab" => rows,
    )
end

function _hf_export_model(tokenizer::AbstractSubwordTokenizer)
    throw(ArgumentError("Cannot export tokenizer type $(typeof(tokenizer)) as HF tokenizer.json"))
end

function _hf_export_vocab_map(tokens::Vector{String})::Dict{String,Int}
    vocab = Dict{String,Int}()
    for (id, token) in enumerate(tokens)
        vocab[token] = id - 1
    end
    return vocab
end

function _hf_export_sorted_pairs(
    pair_ranks::Dict{Tuple{String,String},Int},
)::Vector{Tuple{String,String}}
    ranked = collect(pair_ranks)
    sort!(ranked; by=entry -> (entry[2], entry[1][1], entry[1][2]))
    return [pair for (pair, _) in ranked]
end

function _hf_export_merges(merges::Vector{Tuple{String,String}})::Vector{String}
    return [left * " " * right for (left, right) in merges]
end

function _hf_export_truncation(
    truncation::Union{Nothing,NamedTuple},
)::Union{Nothing,Dict{String,Any}}
    truncation === nothing && return nothing
    return Dict(
        "max_length" => truncation.max_length,
        "strategy" => string(truncation.strategy),
        "stride" => truncation.stride,
        "direction" => string(truncation.direction),
    )
end

function _hf_export_padding(
    padding::Union{Nothing,NamedTuple},
)::Union{Nothing,Dict{String,Any}}
    padding === nothing && return nothing
    return Dict(
        "strategy" => string(padding.strategy),
        "direction" => string(padding.direction),
        "pad_id" => padding.pad_id - 1,
        "pad_type_id" => padding.pad_type_id,
        "pad_token" => string(padding.pad_token),
        "length" => padding.length,
    )
end
