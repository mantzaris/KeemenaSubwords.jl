"""
Load tokenizer by built-in model name.
"""
function load_tokenizer(
    name::Symbol;
    prefetch::Bool=true,
    kwargs...,
)::AbstractSubwordTokenizer
    if prefetch
        prefetch_models([name])
    end
    info = describe_model(name)
    if !info.exists
        if info.distribution == :installable_gated
            throw(ArgumentError(
                "Model $name is gated and not installed. Run install_model!(:$name; token=ENV[\"HF_TOKEN\"]) first.",
            ))
        end
        throw(ArgumentError("Model asset missing on disk for $name at $(info.path)"))
    end
    return load_tokenizer(info.path; format=info.format, model_name=String(name), kwargs...)
end

"""
Load tokenizer from file system path.

Common `format` contracts:
- `:hf_tokenizer_json` -> `tokenizer.json`
- `:bpe_gpt2` -> `vocab.json` + `merges.txt`
- `:bpe_encoder` -> `encoder.json` + `vocab.bpe`
- `:wordpiece` / `:wordpiece_vocab` -> `vocab.txt`
- `:sentencepiece_model` -> `*.model` / `*.model.v3` / `sentencepiece.bpe.model`
- `:tiktoken` -> `*.tiktoken` or tiktoken-text `tokenizer.model`

Examples:
- `load_tokenizer("/path/to/model_dir")`
- `load_tokenizer("/path/to/tokenizer.model"; format=:tiktoken)`
- `load_tokenizer("/path/to/tokenizer.json"; format=:hf_tokenizer_json)`
"""
function load_tokenizer(
    path::AbstractString;
    format::Symbol=:auto,
    kwargs...,
)::AbstractSubwordTokenizer
    resolved = String(path)
    if format === :bpe_gpt2
        vocab_json, merges_txt = _resolve_gpt2_paths(resolved)
        return load_bpe_gpt2(vocab_json, merges_txt; kwargs...)
    elseif format === :bpe_encoder
        encoder_json, vocab_bpe = _resolve_bpe_encoder_paths(resolved)
        return load_bpe_encoder(encoder_json, vocab_bpe; kwargs...)
    elseif format === :tiktoken
        return load_tiktoken(resolved; kwargs...)
    elseif format === :hf_tokenizer_json
        if isdir(resolved) && !isfile(joinpath(resolved, "tokenizer.json")) && _has_gpt2_assets(resolved)
            vocab_json, merges_txt = _resolve_gpt2_paths(resolved)
            return load_bpe_gpt2(vocab_json, merges_txt; kwargs...)
        end
        return load_hf_tokenizer_json(resolved; kwargs...)
    elseif format === :sentencepiece_model
        return load_sentencepiece(resolved; kwargs...)
    end
    selected_format = format === :auto ? detect_tokenizer_format(resolved) : _canonical_load_format(format)

    if selected_format === :wordpiece
        return load_wordpiece(resolved; kwargs...)
    elseif selected_format === :bpe
        return load_bpe(resolved; kwargs...)
    elseif selected_format === :bpe_gpt2
        vocab_json, merges_txt = _resolve_gpt2_paths(resolved)
        return load_bpe_gpt2(vocab_json, merges_txt; kwargs...)
    elseif selected_format === :bpe_encoder
        encoder_json, vocab_bpe = _resolve_bpe_encoder_paths(resolved)
        return load_bpe_encoder(encoder_json, vocab_bpe; kwargs...)
    elseif selected_format === :bytebpe
        return load_bytebpe(resolved; kwargs...)
    elseif selected_format === :unigram
        return load_unigram(resolved; kwargs...)
    elseif selected_format in (:sentencepiece, :sentencepiece_model)
        return load_sentencepiece(resolved; kwargs...)
    elseif selected_format === :tiktoken
        return load_tiktoken(resolved; kwargs...)
    elseif selected_format === :hf_tokenizer_json
        return load_hf_tokenizer_json(resolved; kwargs...)
    end

    throw(ArgumentError("Unsupported tokenizer format: $selected_format"))
end

"""
Load tokenizer from explicit `(vocab_path, merges_path)` tuple.

This tuple form is for classic BPE/byte-level BPE (`vocab.txt` + `merges.txt`)
or explicit JSON-pair loaders (`vocab.json` + `merges.txt`, `encoder.json` + `vocab.bpe`)
when accompanied by `format`.
"""
function load_tokenizer(
    paths::Tuple{<:AbstractString,<:AbstractString};
    format::Symbol=:bpe,
    kwargs...,
)::AbstractSubwordTokenizer
    vocab_path, merges_path = paths
    if format === :bpe_gpt2
        if lowercase(basename(vocab_path)) == "vocab.txt"
            return load_bpe(vocab_path, merges_path; kwargs...)
        end
        return load_bpe_gpt2(vocab_path, merges_path; kwargs...)
    elseif format === :bpe_encoder
        return load_bpe_encoder(vocab_path, merges_path; kwargs...)
    end

    selected_format = _canonical_load_format(format)

    if selected_format === :bpe
        return load_bpe(vocab_path, merges_path; kwargs...)
    elseif selected_format === :bytebpe
        return load_bytebpe(vocab_path, merges_path; kwargs...)
    end

    throw(ArgumentError("Tuple model path loading supports :bpe, :bytebpe, :bpe_gpt2, and :bpe_encoder; got: $format"))
end

"""
Load tokenizer from a named specification.

Examples:
- `(format=:wordpiece, path="/.../vocab.txt")`
- `(format=:hf_tokenizer_json, path="/.../tokenizer.json")`
- `(format=:unigram, path="/.../unigram.tsv")`
- `(format=:bpe_gpt2, vocab_json="/.../vocab.json", merges_txt="/.../merges.txt")`
- `(format=:bpe_encoder, encoder_json="/.../encoder.json", vocab_bpe="/.../vocab.bpe")`
- `(format=:wordpiece, vocab_txt="/.../vocab.txt")` (alias)
- `(format=:sentencepiece_model, model_file="/.../tokenizer.model")` (alias)
- `(format=:tiktoken, encoding_file="/.../o200k_base.tiktoken")` (alias)
- `(format=:hf_tokenizer_json, tokenizer_json="/.../tokenizer.json")` (alias)
- `(format=:unigram, unigram_tsv="/.../unigram.tsv")` (alias)
"""
function load_tokenizer(spec::NamedTuple; kwargs...)::AbstractSubwordTokenizer
    haskey(spec, :format) || throw(ArgumentError("Tokenizer spec must include :format"))
    format = spec[:format]

    if haskey(spec, :path)
        return load_tokenizer(String(spec[:path]); format=format, kwargs...)
    end

    if haskey(spec, :vocab) && haskey(spec, :merges)
        return load_tokenizer((String(spec[:vocab]), String(spec[:merges])); format=format, kwargs...)
    end

    if haskey(spec, :vocab_json) && haskey(spec, :merges_txt)
        return load_bpe_gpt2(String(spec[:vocab_json]), String(spec[:merges_txt]); kwargs...)
    end

    if haskey(spec, :encoder_json) && haskey(spec, :vocab_bpe)
        return load_bpe_encoder(String(spec[:encoder_json]), String(spec[:vocab_bpe]); kwargs...)
    end

    if haskey(spec, :tokenizer_json)
        return load_hf_tokenizer_json(String(spec[:tokenizer_json]); kwargs...)
    end

    if haskey(spec, :vocab_txt)
        return load_wordpiece(String(spec[:vocab_txt]); kwargs...)
    end

    if haskey(spec, :unigram_tsv)
        return load_unigram(String(spec[:unigram_tsv]); kwargs...)
    end

    if haskey(spec, :model_file)
        return load_sentencepiece(String(spec[:model_file]); kwargs...)
    end

    if haskey(spec, :encoding_file)
        return load_tiktoken(String(spec[:encoding_file]); kwargs...)
    end

    throw(ArgumentError(
        "Tokenizer spec requires one of: :path, (:vocab,:merges), (:vocab_json,:merges_txt), " *
        "(:encoder_json,:vocab_bpe), :vocab_txt, :unigram_tsv, :tokenizer_json, :model_file, or :encoding_file.",
    ))
end

"""
Load tokenizer from a `FilesSpec`.
"""
function load_tokenizer(spec::FilesSpec; kwargs...)::AbstractSubwordTokenizer
    return load_tokenizer(_filespec_to_namedtuple(spec); kwargs...)
end

const _TOKENIZER_CACHE = Dict{Tuple{Symbol,String,Symbol},AbstractSubwordTokenizer}()
const _TOKENIZER_CACHE_LOCK = ReentrantLock()

_cache_format(format::Union{Nothing,Symbol}) = format === nothing ? :auto : _canonical_load_format(format)

function _tokenizer_cache_key(source::Symbol, format::Union{Nothing,Symbol})::Tuple{Symbol,String,Symbol}
    return (:model_key, String(source), _cache_format(format))
end

function _tokenizer_cache_key(source::AbstractString, format::Union{Nothing,Symbol})::Tuple{Symbol,String,Symbol}
    return (:path, normpath(String(source)), _cache_format(format))
end

"""
Return a cached tokenizer for a model key or path, loading and caching on first use.
"""
function get_tokenizer_cached(
    source::Symbol;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::AbstractSubwordTokenizer
    key = _tokenizer_cache_key(source, format)
    lock(_TOKENIZER_CACHE_LOCK) do
        haskey(_TOKENIZER_CACHE, key) && return _TOKENIZER_CACHE[key]
    end

    tok = if format === nothing || format == :auto
        load_tokenizer(source; prefetch=prefetch)
    else
        path = model_path(source; auto_prefetch=prefetch)
        load_tokenizer(path; format=_canonical_load_format(format), model_name=String(source))
    end

    lock(_TOKENIZER_CACHE_LOCK) do
        return get!(_TOKENIZER_CACHE, key, tok)
    end
end

function get_tokenizer_cached(
    source::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::AbstractSubwordTokenizer
    key = _tokenizer_cache_key(source, format)
    lock(_TOKENIZER_CACHE_LOCK) do
        haskey(_TOKENIZER_CACHE, key) && return _TOKENIZER_CACHE[key]
    end

    selected = format === nothing ? :auto : _canonical_load_format(format)
    tok = load_tokenizer(String(source); format=selected)

    lock(_TOKENIZER_CACHE_LOCK) do
        return get!(_TOKENIZER_CACHE, key, tok)
    end
end

"""
Clear the in-session tokenizer cache used by one-call convenience APIs.
"""
function clear_tokenizer_cache!()::Nothing
    lock(_TOKENIZER_CACHE_LOCK) do
        empty!(_TOKENIZER_CACHE)
    end
    return nothing
end

"""
List cache keys for in-session cached tokenizers.
"""
function cached_tokenizers()::Vector{Tuple{Symbol,String,Symbol}}
    lock(_TOKENIZER_CACHE_LOCK) do
        return sort!(collect(keys(_TOKENIZER_CACHE)); by=string)
    end
end

"""
One-call tokenize by model key.
"""
function tokenize(
    source::Symbol,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::Vector{String}
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return tokenize(tok, text)
end

"""
One-call tokenize by tokenizer path/directory.
"""
function tokenize(
    source::AbstractString,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::Vector{String}
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return tokenize(tok, text)
end

"""
One-call encode by model key.
"""
function encode(
    source::Symbol,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::Vector{Int}
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return encode(tok, text; kwargs...)
end

"""
One-call encode by tokenizer path/directory.
"""
function encode(
    source::AbstractString,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::Vector{Int}
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return encode(tok, text; kwargs...)
end

"""
One-call structured encode by model key.
"""
function encode_result(
    source::Symbol,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::TokenizationResult
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return encode_result(tok, text; kwargs...)
end

"""
One-call structured encode by tokenizer path/directory.
"""
function encode_result(
    source::AbstractString,
    text::AbstractString;
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
    kwargs...,
)::TokenizationResult
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return encode_result(tok, text; kwargs...)
end

"""
One-call decode by model key.
"""
function decode(
    source::Symbol,
    ids::AbstractVector{<:Integer};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::String
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return decode(tok, ids)
end

"""
One-call decode by tokenizer path/directory.
"""
function decode(
    source::AbstractString,
    ids::AbstractVector{<:Integer};
    format::Union{Nothing,Symbol}=nothing,
    prefetch::Bool=true,
)::String
    tok = get_tokenizer_cached(source; format=format, prefetch=prefetch)
    return decode(tok, ids)
end

function _filespec_to_namedtuple(spec::FilesSpec)::NamedTuple
    pairs = Pair{Symbol,Any}[:format => spec.format]
    _push_filespec_pair!(pairs, :path, spec.path)
    _push_filespec_pair!(pairs, :vocab, spec.vocab)
    _push_filespec_pair!(pairs, :merges, spec.merges)
    _push_filespec_pair!(pairs, :vocab_json, spec.vocab_json)
    _push_filespec_pair!(pairs, :merges_txt, spec.merges_txt)
    _push_filespec_pair!(pairs, :encoder_json, spec.encoder_json)
    _push_filespec_pair!(pairs, :vocab_bpe, spec.vocab_bpe)
    _push_filespec_pair!(pairs, :vocab_txt, spec.vocab_txt)
    _push_filespec_pair!(pairs, :unigram_tsv, spec.unigram_tsv)
    _push_filespec_pair!(pairs, :tokenizer_json, spec.tokenizer_json)
    _push_filespec_pair!(pairs, :model_file, spec.model_file)
    _push_filespec_pair!(pairs, :encoding_file, spec.encoding_file)
    return _namedtuple_from_pairs(pairs)
end

function _push_filespec_pair!(pairs::Vector{Pair{Symbol,Any}}, key::Symbol, value::Union{Nothing,String})::Nothing
    value === nothing || push!(pairs, key => value)
    return nothing
end

function _namedtuple_from_pairs(pairs::Vector{Pair{Symbol,Any}})::NamedTuple
    return (; pairs...)
end

function encode_result(
    tokenizer::AbstractSubwordTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=true,
    assume_normalized::Bool=false,
    return_offsets::Bool=false,
    return_masks::Bool=false,
)::TokenizationResult
    raw_text = String(text)
    tokenization_text = assume_normalized ? raw_text : tokenization_view(tokenizer, raw_text)

    ids = assume_normalized ?
        _encode_assume_normalized(tokenizer, tokenization_text; add_special_tokens=add_special_tokens) :
        encode(tokenizer, raw_text; add_special_tokens=add_special_tokens)
    tokens = String[id_to_token(tokenizer, id) for id in ids]

    offsets = return_offsets ? _encode_result_offsets(
        tokenizer,
        tokenization_text,
        ids;
        add_special_tokens=add_special_tokens,
    ) : nothing
    attention_mask = return_masks ? fill(1, length(ids)) : nothing
    token_type_ids = return_masks ? fill(0, length(ids)) : nothing
    special_tokens_mask = return_masks ? _special_tokens_mask(tokenizer, ids) : nothing
    info = model_info(tokenizer)

    metadata = (
        format = info.format,
        model_name = info.model_name,
        add_special_tokens = add_special_tokens,
        assume_normalized = assume_normalized,
        offsets_coordinates = offsets_coordinate_system(),
        offsets_reference = assume_normalized ? :input_text : :tokenizer_normalized_text,
    )

    return TokenizationResult(
        ids,
        tokens,
        offsets,
        attention_mask,
        token_type_ids,
        special_tokens_mask,
        metadata,
    )
end

"""
Batch variant of `encode_result`.
"""
function encode_batch_result(
    tokenizer::AbstractSubwordTokenizer,
    texts::AbstractVector{<:AbstractString};
    kwargs...,
)::Vector{TokenizationResult}
    return [encode_result(tokenizer, text; kwargs...) for text in texts]
end

function _encode_assume_normalized(
    tokenizer::AbstractSubwordTokenizer,
    text::String;
    add_special_tokens::Bool,
)::Vector{Int}
    return encode(tokenizer, text; add_special_tokens=add_special_tokens)
end

function _encode_assume_normalized(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String;
    add_special_tokens::Bool,
)::Vector{Int}
    return encode(tokenizer, text; add_special_tokens=add_special_tokens, assume_normalized=true)
end

function _special_tokens_mask(
    tokenizer::AbstractSubwordTokenizer,
    ids::AbstractVector{Int},
)::Vector{Int}
    special_ids = Set(values(special_tokens(tokenizer)))
    return [id in special_ids ? 1 : 0 for id in ids]
end

function _special_tokens_mask(
    tokenizer::HuggingFaceJSONTokenizer,
    ids::AbstractVector{Int},
)::Vector{Int}
    special_ids = Set{Int}()
    union!(special_ids, values(tokenizer.special_token_ids))
    union!(special_ids, values(tokenizer.token_special_ids))
    return [id in special_ids ? 1 : 0 for id in ids]
end

function _encode_result_offsets(
    tokenizer::AbstractSubwordTokenizer,
    text::String,
    ids::Vector{Int};
    add_special_tokens::Bool,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    base_ids = _encode_assume_normalized(tokenizer, text; add_special_tokens=false)
    base_tokens = String[id_to_token(tokenizer, id) for id in base_ids]

    length(base_tokens) == length(base_ids) || return nothing
    base_offsets = _token_offsets_for_tokens(tokenizer, text, base_tokens, base_ids)
    base_offsets === nothing && return nothing

    if !add_special_tokens
        ids == base_ids || return nothing
        return base_offsets
    end

    return _inject_special_offsets(ids, base_ids, base_offsets)
end

function _encode_result_offsets(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    ids::Vector{Int};
    add_special_tokens::Bool,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    base_ids = _encode_assume_normalized(tokenizer, text; add_special_tokens=false)
    segments = _segment_hf_input_with_spans(tokenizer, text)
    base_offsets = _hf_offsets_from_segments(tokenizer, segments, base_ids)
    base_offsets === nothing && return nothing

    if !add_special_tokens
        ids == base_ids || return nothing
        return base_offsets
    end

    return _inject_special_offsets(ids, base_ids, base_offsets)
end

_token_offsets_for_tokens(::AbstractSubwordTokenizer, ::String, ::Vector{String}, ::Vector{Int}) = nothing

function _segment_hf_input_with_spans(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
)::Vector{NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}}
    special_pass = _split_with_added_patterns_with_spans(text, tokenizer.special_added_patterns)
    out = NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}[]

    for seg in special_pass
        if seg.kind == :added
            push!(out, seg)
            continue
        end

        raw_pass = _split_with_added_patterns_with_spans(seg.text, tokenizer.raw_added_patterns)
        for raw_seg in raw_pass
            raw_global = _segment_local_to_global(raw_seg, seg.start)
            if raw_seg.kind == :added
                push!(out, raw_global)
                continue
            end

            normalized_pass = _split_with_added_patterns_with_spans(raw_seg.text, tokenizer.normalized_added_patterns)
            for normalized_seg in normalized_pass
                push!(out, _segment_local_to_global(normalized_seg, raw_global.start))
            end
        end
    end

    return out
end

function _split_with_added_patterns_with_spans(
    text::String,
    patterns::Vector{HFAddedTokenPattern},
)::Vector{NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}}
    segments = NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}[]
    isempty(text) && return segments

    text_start = firstindex(text)
    text_stop = lastindex(text)
    text_stop_exclusive = ncodeunits(text) + 1

    if isempty(patterns)
        push!(segments, (kind=:text, text=text, id=0, start=text_start, stop=text_stop_exclusive))
        return segments
    end

    i = text_start
    chunk_start = i

    while i <= text_stop
        best = _best_added_match(text, i, patterns)
        if best === nothing
            i = nextind(text, i)
            continue
        end

        if chunk_start < best.start
            prev_idx = prevind(text, best.start)
            push!(segments, (
                kind=:text,
                text=String(SubString(text, chunk_start, prev_idx)),
                id=0,
                start=chunk_start,
                stop=best.start,
            ))
        end

        push!(segments, (
            kind=:added,
            text=best.content,
            id=best.id,
            start=best.start,
            stop=nextind(text, best.stop),
        ))

        i = nextind(text, best.stop)
        chunk_start = i
    end

    if chunk_start <= text_stop
        push!(segments, (
            kind=:text,
            text=String(SubString(text, chunk_start, text_stop)),
            id=0,
            start=chunk_start,
            stop=text_stop_exclusive,
        ))
    end

    return segments
end

function _segment_local_to_global(
    segment::NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}},
    global_start::Int,
)::NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}
    return (
        kind = segment.kind,
        text = segment.text,
        id = segment.id,
        start = _advance_codeunits(global_start, segment.start - offsets_index_base()),
        stop = _advance_codeunits(global_start, segment.stop - offsets_index_base()),
    )
end

function _hf_offsets_from_segments(
    tokenizer::HuggingFaceJSONTokenizer,
    segments::Vector{NamedTuple{(:kind, :text, :id, :start, :stop),Tuple{Symbol,String,Int,Int,Int}}},
    base_ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    offsets = Tuple{Int,Int}[]
    token_idx = 1

    for seg in segments
        if seg.kind == :added
            token_idx <= length(base_ids) || return nothing
            base_ids[token_idx] == seg.id || return nothing
            push!(offsets, (seg.start, seg.stop))
            token_idx += 1
            continue
        end

        pretokenized = _apply_hf_pretokenizer(tokenizer.pretokenizer, seg.text)
        seg_ids = _encode_hf_model_segment(tokenizer, pretokenized)
        seg_tokens = String[id_to_token(tokenizer, id) for id in seg_ids]
        local_offsets = _hf_local_model_offsets(tokenizer, seg.text, seg_tokens, seg_ids)
        local_offsets === nothing && return nothing
        length(local_offsets) == length(seg_ids) || return nothing

        for seg_id in seg_ids
            token_idx <= length(base_ids) || return nothing
            base_ids[token_idx] == seg_id || return nothing
            token_idx += 1
        end

        if seg.start == offsets_index_base()
            append!(offsets, local_offsets)
        else
            shifted = _shift_offsets(local_offsets, seg.start - offsets_index_base())
            shifted === nothing && return nothing
            append!(offsets, shifted)
        end
    end

    token_idx == length(base_ids) + 1 || return nothing
    return offsets
end

function _hf_local_model_offsets(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    pre = tokenizer.pretokenizer

    if pre isa HFNoopPreTokenizer || pre isa HFWhitespacePreTokenizer
        return _token_offsets_for_tokens(tokenizer.base, text, tokens, ids)
    elseif pre isa HFByteLevelPreTokenizer
        pretokenized = _apply_hf_pretokenizer(pre, text)
        base_offsets = _token_offsets_for_tokens(tokenizer.base, pretokenized, tokens, ids)
        base_offsets === nothing && return nothing
        shift = pre.add_prefix_space && !isempty(text) && !startswith(text, " ") ? -1 : 0
        return shift == 0 ? base_offsets : _shift_offsets(base_offsets, shift)
    elseif pre isa HFMetaspacePreTokenizer
        return _hf_metaspace_offsets(tokenizer, text, tokens, pre)
    end

    return nothing
end

function _token_offsets_for_tokens(
    tokenizer::WordPieceTokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    _ = ids
    function _strip_wordpiece(piece::String)::String
        if startswith(piece, tokenizer.continuation_prefix)
            start_idx = nextind(piece, firstindex(piece), length(tokenizer.continuation_prefix))
            start_idx > lastindex(piece) && return ""
            return String(SubString(piece, start_idx))
        end
        return piece
    end

    return _wordwise_piece_offsets(
        normalized,
        tokenizer,
        tokens;
        unk_token=tokenizer.unk_token,
        strip_token=_strip_wordpiece,
    )
end

function _token_offsets_for_tokens(
    tokenizer::BPETokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    _ = ids
    marker = tokenizer.end_of_word_marker
    strip_piece = marker === nothing ?
        (piece -> piece) :
        (piece -> replace(piece, marker => ""))

    return _wordwise_piece_offsets(
        normalized,
        tokenizer,
        tokens;
        unk_token=tokenizer.unk_token,
        strip_token=strip_piece,
    )
end

function _token_offsets_for_tokens(
    tokenizer::UnigramTokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    _ = ids
    isempty(tokenizer.whitespace_marker) || return nothing
    return _wordwise_piece_offsets(
        normalized,
        tokenizer,
        tokens;
        unk_token=tokenizer.unk_token,
        strip_token=piece -> piece,
    )
end

function _token_offsets_for_tokens(
    tokenizer::ByteBPETokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    _ = ids
    offsets = Tuple{Int,Int}[]
    token_idx = 1

    for (start_idx, _, word) in _word_spans(normalized)
        pieces = _tokenize_byte_word(tokenizer, word)
        token_idx + length(pieces) - 1 <= length(tokens) || return nothing
        local_end = _advance_codeunits(start_idx, ncodeunits(word))
        cursor = start_idx

        for piece in pieces
            current = tokens[token_idx]
            current == piece || return nothing

            if piece == tokenizer.base.unk_token
                push!(offsets, (start_idx, local_end))
                cursor = local_end
                token_idx += 1
                continue
            end

            piece_bytes = _bytebpe_piece_nbytes(tokenizer, piece)
            piece_bytes === nothing && return nothing
            next_cursor = _advance_codeunits(cursor, piece_bytes)
            push!(offsets, (cursor, next_cursor))
            cursor = next_cursor
            token_idx += 1
        end

        cursor <= local_end || return nothing
    end

    token_idx == length(tokens) + 1 || return nothing
    return offsets
end

function _token_offsets_for_tokens(
    tokenizer::SentencePieceTokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    _ = ids
    unk = _unknown_token(tokenizer)
    unk === nothing && return nothing
    strip_piece = piece -> replace(piece, tokenizer.whitespace_marker => "")

    return _wordwise_piece_offsets(
        normalized,
        tokenizer,
        tokens;
        unk_token=unk,
        strip_token=strip_piece,
    )
end

function _token_offsets_for_tokens(
    tokenizer::TiktokenTokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    length(tokens) == length(ids) || return nothing
    offsets = Tuple{Int,Int}[]
    cursor = firstindex(normalized)

    for id in ids
        1 <= id <= length(tokenizer.id_to_bytes) || return nothing
        piece_bytes = length(tokenizer.id_to_bytes[id])
        next_cursor = _advance_codeunits(cursor, piece_bytes)
        push!(offsets, (cursor, next_cursor))
        cursor = next_cursor
    end

    cursor == ncodeunits(normalized) + 1 || return nothing
    return offsets
end

function _token_offsets_for_tokens(
    tokenizer::HuggingFaceJSONTokenizer,
    normalized::String,
    tokens::Vector{String},
    ids::Vector{Int},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    pre = tokenizer.pretokenizer

    if pre isa HFNoopPreTokenizer || pre isa HFWhitespacePreTokenizer
        return _token_offsets_for_tokens(tokenizer.base, normalized, tokens, ids)
    elseif pre isa HFByteLevelPreTokenizer
        pretokenized = _apply_hf_pretokenizer(pre, normalized)
        base_offsets = _token_offsets_for_tokens(tokenizer.base, pretokenized, tokens, ids)
        base_offsets === nothing && return nothing
        shift = pre.add_prefix_space && !isempty(normalized) && !startswith(normalized, " ") ? -1 : 0
        return shift == 0 ? base_offsets : _shift_offsets(base_offsets, shift)
    elseif pre isa HFMetaspacePreTokenizer
        return _hf_metaspace_offsets(tokenizer, normalized, tokens, pre)
    end

    return nothing
end

function _wordwise_piece_offsets(
    normalized::String,
    tokenizer,
    tokens::Vector{String};
    unk_token::String,
    strip_token::Function,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    offsets = Tuple{Int,Int}[]
    token_idx = 1

    for (start_idx, stop_idx, word) in _word_spans(normalized)
        pieces = tokenize(tokenizer, word)
        token_idx + length(pieces) - 1 <= length(tokens) || return nothing
        local_end = _advance_chars(normalized, start_idx, length(word))
        cursor = start_idx

        for piece in pieces
            current = tokens[token_idx]
            current == piece || return nothing

            if piece == unk_token
                push!(offsets, (start_idx, local_end))
                cursor = local_end
                token_idx += 1
                continue
            end

            clean = strip_token(piece)
            span_len = length(clean)
            next_cursor = span_len == 0 ? cursor : _advance_chars(normalized, cursor, span_len)
            push!(offsets, (cursor, next_cursor))
            cursor = next_cursor
            token_idx += 1
        end

        cursor <= local_end || return nothing
    end

    token_idx == length(tokens) + 1 || return nothing
    return offsets
end

function _hf_metaspace_offsets(
    tokenizer::HuggingFaceJSONTokenizer,
    normalized::String,
    tokens::Vector{String},
    pre::HFMetaspacePreTokenizer,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    replacement_len = length(pre.replacement)
    replacement_len == 0 && return nothing

    unk = _unknown_token(tokenizer.base)
    unk === nothing && return nothing

    offsets = Tuple{Int,Int}[]
    token_idx = 1

    for (start_idx, _, word) in _word_spans(normalized)
        encoded_word = string(pre.replacement, word)
        pieces = tokenize(tokenizer.base, encoded_word)
        token_idx + length(pieces) - 1 <= length(tokens) || return nothing
        local_end = _advance_chars(normalized, start_idx, length(word))
        cursor = start_idx

        for piece in pieces
            current = tokens[token_idx]
            current == piece || return nothing

            if piece == unk
                push!(offsets, (start_idx, local_end))
                cursor = local_end
                token_idx += 1
                continue
            end

            clean = replace(piece, pre.replacement => "")
            span_len = length(clean)
            next_cursor = span_len == 0 ? cursor : _advance_chars(normalized, cursor, span_len)
            push!(offsets, (cursor, next_cursor))
            cursor = next_cursor
            token_idx += 1
        end

        cursor <= local_end || return nothing
    end

    token_idx == length(tokens) + 1 || return nothing
    return offsets
end

function _word_spans(text::String)::Vector{Tuple{Int,Int,String}}
    spans = Tuple{Int,Int,String}[]
    for m in eachmatch(r"\S+", text)
        start_idx = m.offset
        count = length(m.match)
        end_idx = _advance_chars(text, start_idx, count)
        stop_idx = prevind(text, end_idx)
        push!(spans, (start_idx, stop_idx, String(m.match)))
    end
    return spans
end

function _advance_chars(text::String, start_idx::Int, count::Int)::Int
    idx = start_idx
    for _ in 1:count
        idx = nextind(text, idx)
    end
    return idx
end

function _advance_codeunits(start_idx::Int, count::Int)::Int
    return start_idx + count
end

function _shift_offsets(
    offsets::Vector{Tuple{Int,Int}},
    delta::Int,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    shifted = Tuple{Int,Int}[]
    for (start_idx, stop_idx) in offsets
        new_start = start_idx + delta
        new_stop = stop_idx + delta
        new_start >= 1 || return nothing
        new_stop >= new_start || return nothing
        push!(shifted, (new_start, new_stop))
    end
    return shifted
end

function _bytebpe_piece_nbytes(tokenizer::ByteBPETokenizer, piece::String)::Union{Nothing,Int}
    marker = tokenizer.base.end_of_word_marker
    clean = marker === nothing ? piece : replace(piece, marker => "")
    total = 0

    for c in clean
        haskey(tokenizer.unicode_to_byte, c) || return nothing
        total += 1
    end

    return total
end

_unknown_token(::AbstractSubwordTokenizer) = nothing
_unknown_token(tokenizer::WordPieceTokenizer) = tokenizer.unk_token
_unknown_token(tokenizer::BPETokenizer) = tokenizer.unk_token
_unknown_token(tokenizer::UnigramTokenizer) = tokenizer.unk_token
_unknown_token(tokenizer::ByteBPETokenizer) = tokenizer.base.unk_token
_unknown_token(tokenizer::SentencePieceTokenizer) = _unknown_token(tokenizer.inner)

function _inject_special_offsets(
    all_ids::Vector{Int},
    base_ids::Vector{Int},
    base_offsets::Vector{Tuple{Int,Int}},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    sentinel = offsets_sentinel()
    isempty(base_ids) && return [sentinel for _ in all_ids]
    length(base_ids) == length(base_offsets) || return nothing

    match_range = _find_subsequence_range(all_ids, base_ids)
    match_range === nothing && return nothing

    offsets = [sentinel for _ in all_ids]
    start_idx, stop_idx = match_range
    for (i, base_offset) in enumerate(base_offsets)
        offsets[start_idx + i - 1] = base_offset
    end

    stop_idx - start_idx + 1 == length(base_offsets) || return nothing
    return offsets
end

function _find_subsequence_range(
    haystack::Vector{Int},
    needle::Vector{Int},
)::Union{Nothing,Tuple{Int,Int}}
    n = length(needle)
    m = length(haystack)
    n == 0 && return (1, 0)
    n > m && return nothing

    for i in 1:(m - n + 1)
        @inbounds if haystack[i:(i + n - 1)] == needle
            return (i, i + n - 1)
        end
    end

    return nothing
end

"""
Save tokenizer to a canonical on-disk format.

`format=:internal` chooses a tokenizer-family specific default:
- `WordPieceTokenizer` -> `vocab.txt`
- `BPETokenizer` / `ByteBPETokenizer` -> `vocab.txt` + `merges.txt`
- `UnigramTokenizer` -> `unigram.tsv`
- `SentencePieceTokenizer` -> `spm.model`
"""
function save_tokenizer(
    tokenizer::AbstractSubwordTokenizer,
    outdir::AbstractString;
    format::Symbol=:internal,
)::Nothing
    export_tokenizer(tokenizer, outdir; format=format)
    return nothing
end

"""
Export tokenizer to external formats.

Supported `format` values:
- `:internal`
- `:bpe` / `:bpe_gpt2`
- `:wordpiece_vocab`
- `:unigram_tsv`
- `:sentencepiece_model`
"""
function export_tokenizer(
    tokenizer::AbstractSubwordTokenizer,
    outdir::AbstractString;
    format::Symbol,
)::Nothing
    mkpath(outdir)
    target = _canonical_export_format(tokenizer, format)

    if target === :wordpiece_vocab
        _write_wordpiece_vocab(_as_wordpiece(tokenizer), joinpath(outdir, "vocab.txt"))
    elseif target === :bpe_files
        _write_bpe_files(_as_bpe(tokenizer), outdir)
    elseif target === :unigram_tsv
        _write_unigram_tsv(_as_unigram(tokenizer), joinpath(outdir, "unigram.tsv"))
    elseif target === :sentencepiece_model
        _write_sentencepiece_model(tokenizer, joinpath(outdir, "spm.model"))
    else
        throw(ArgumentError("Unsupported export target: $target"))
    end

    return nothing
end

function _canonical_load_format(format::Symbol)::Symbol
    if format in (:bpe, :bpe_gpt2)
        return :bpe
    elseif format in (:bpe_encoder,)
        return :bpe_encoder
    elseif format in (:bytebpe,)
        return :bytebpe
    elseif format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece
    elseif format in (:unigram, :unigram_tsv)
        return :unigram
    elseif format in (:sentencepiece,)
        return :sentencepiece
    elseif format in (:sentencepiece_model,)
        return :sentencepiece_model
    elseif format in (:tiktoken,)
        return :tiktoken
    elseif format in (:hf_tokenizer_json,)
        return :hf_tokenizer_json
    end

    throw(ArgumentError("Unsupported tokenizer format: $format"))
end

function _canonical_export_format(tokenizer::AbstractSubwordTokenizer, format::Symbol)::Symbol
    if format === :internal
        if tokenizer isa WordPieceTokenizer
            return :wordpiece_vocab
        elseif tokenizer isa BPETokenizer || tokenizer isa ByteBPETokenizer
            return :bpe_files
        elseif tokenizer isa UnigramTokenizer
            return :unigram_tsv
        elseif tokenizer isa SentencePieceTokenizer
            return :sentencepiece_model
        end
        throw(ArgumentError("No internal save format is defined for $(typeof(tokenizer))"))
    end

    if format in (:bpe, :bpe_gpt2)
        return :bpe_files
    elseif format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece_vocab
    elseif format in (:unigram, :unigram_tsv)
        return :unigram_tsv
    elseif format in (:sentencepiece, :sentencepiece_model)
        return :sentencepiece_model
    end

    throw(ArgumentError("Unsupported export format: $format"))
end

"""
Inspect a tokenizer directory and return detected candidate files.

Example:
`detect_tokenizer_files("/path/to/model_dir")`
"""
function detect_tokenizer_files(dir::AbstractString)::NamedTuple
    root = normpath(String(dir))
    isdir(root) || throw(ArgumentError("Tokenizer directory does not exist: $root"))
    tokenizer_json = isfile(joinpath(root, "tokenizer.json")) ? joinpath(root, "tokenizer.json") : nothing

    sentencepiece_candidates = String[]
    for filename in ("spm.model", "spiece.model", "tokenizer.model", "tokenizer.model.v3", "sentencepiece.bpe.model")
        path = joinpath(root, filename)
        isfile(path) && push!(sentencepiece_candidates, path)
    end
    if isempty(sentencepiece_candidates) && tokenizer_json === nothing
        any_model = sort(filter(path -> begin
            lower = lowercase(path)
            endswith(lower, ".model") || endswith(lower, ".model.v3")
        end, readdir(root; join=true)))
        length(any_model) == 1 && push!(sentencepiece_candidates, only(any_model))
    end

    tiktoken_files = filter(p -> endswith(lowercase(p), ".tiktoken"), readdir(root; join=true))

    return (
        dir = root,
        tokenizer_json = tokenizer_json,
        vocab_json = isfile(joinpath(root, "vocab.json")) ? joinpath(root, "vocab.json") : nothing,
        merges_txt = isfile(joinpath(root, "merges.txt")) ? joinpath(root, "merges.txt") : nothing,
        encoder_json = isfile(joinpath(root, "encoder.json")) ? joinpath(root, "encoder.json") : nothing,
        vocab_bpe = isfile(joinpath(root, "vocab.bpe")) ? joinpath(root, "vocab.bpe") : nothing,
        vocab_txt = isfile(joinpath(root, "vocab.txt")) ? joinpath(root, "vocab.txt") : nothing,
        unigram_tsv = isfile(joinpath(root, "unigram.tsv")) ? joinpath(root, "unigram.tsv") : nothing,
        sentencepiece_models = sentencepiece_candidates,
        tiktoken_files = tiktoken_files,
    )
end

"""
Detect tokenizer format from a local file or directory.

Returns one of symbols such as `:hf_tokenizer_json`, `:bpe_gpt2`, `:bpe_encoder`,
`:sentencepiece_model`, `:tiktoken`, `:wordpiece`, `:bpe`, or `:unigram`.

Examples:
- `detect_tokenizer_format("/path/to/model_dir")`
- `detect_tokenizer_format("/path/to/tokenizer.model")`
"""
function detect_tokenizer_format(path::AbstractString)::Symbol
    resolved = normpath(String(path))

    if isdir(resolved)
        files = detect_tokenizer_files(resolved)
        if files.tokenizer_json !== nothing
            return :hf_tokenizer_json
        elseif files.vocab_json !== nothing && files.merges_txt !== nothing
            return :bpe_gpt2
        elseif files.encoder_json !== nothing && files.vocab_bpe !== nothing
            return :bpe_encoder
        elseif !isempty(files.sentencepiece_models)
            return :sentencepiece_model
        elseif length(files.tiktoken_files) == 1
            return :tiktoken
        elseif files.vocab_txt !== nothing && files.merges_txt !== nothing
            return :bpe
        elseif files.vocab_txt !== nothing
            return :wordpiece
        elseif files.unigram_tsv !== nothing
            return :unigram
        end

        throw(ArgumentError(
            "Could not infer tokenizer format from directory: $resolved. " *
            "Expected one of tokenizer.json, vocab.json+merges.txt, encoder.json+vocab.bpe, " *
            "SentencePiece .model files, *.tiktoken, vocab.txt(+merges.txt), or unigram.tsv.",
        ))
    end

    isfile(resolved) || throw(ArgumentError("Tokenizer path does not exist: $resolved"))
    lower_path = lowercase(resolved)
    lower_name = lowercase(basename(resolved))

    if endswith(lower_path, ".tiktoken")
        return :tiktoken
    elseif lower_name == "tokenizer.json"
        return :hf_tokenizer_json
    elseif lower_name == "vocab.json"
        sibling_merges = joinpath(dirname(resolved), "merges.txt")
        isfile(sibling_merges) || throw(ArgumentError(
            "Found vocab.json without sibling merges.txt: $resolved. " *
            "Example: load_bpe_gpt2(\"/path/to/vocab.json\", \"/path/to/merges.txt\")",
        ))
        return :bpe_gpt2
    elseif lower_name == "encoder.json"
        sibling_bpe = joinpath(dirname(resolved), "vocab.bpe")
        isfile(sibling_bpe) || throw(ArgumentError(
            "Found encoder.json without sibling vocab.bpe: $resolved. " *
            "Example: load_bpe_encoder(\"/path/to/encoder.json\", \"/path/to/vocab.bpe\")",
        ))
        return :bpe_encoder
    elseif lower_name == "vocab.bpe"
        sibling_encoder = joinpath(dirname(resolved), "encoder.json")
        isfile(sibling_encoder) || throw(ArgumentError(
            "Found vocab.bpe without sibling encoder.json: $resolved. " *
            "Example: load_bpe_encoder(\"/path/to/encoder.json\", \"/path/to/vocab.bpe\")",
        ))
        return :bpe_encoder
    elseif lower_name == "vocab.txt"
        sibling_merges = joinpath(dirname(resolved), "merges.txt")
        return isfile(sibling_merges) ? :bpe : :wordpiece
    elseif lower_name == "unigram.tsv"
        return :unigram
    elseif endswith(lower_path, ".model") || endswith(lower_path, ".model.v3") || lower_name in ("spm.model", "spiece.model", "sentencepiece.bpe.model")
        model_guess = _sniff_model_payload(resolved)
        model_guess !== :unknown && return model_guess
        throw(ArgumentError(
            "Ambiguous .model file at $resolved. Use format override explicitly, e.g. " *
            "load_tokenizer(\"$resolved\"; format=:sentencepiece_model) or format=:tiktoken.",
        ))
    end

    throw(ArgumentError(
        "Could not infer tokenizer format from file: $resolved. " *
        "Use load_tokenizer(path; format=:...) to override.",
    ))
end

_detect_format(path::String)::Symbol = detect_tokenizer_format(path)

function _sample_file_bytes(path::AbstractString, max_bytes::Int=2048)::Vector{UInt8}
    open(path, "r") do io
        return read(io, max_bytes)
    end
end

function _looks_tiktoken_text_payload(bytes::Vector{UInt8})::Bool
    isempty(bytes) && return false
    any(==(0x00), bytes) && return false
    isvalid(String, bytes) || return false

    sample = String(bytes)
    lines = split(sample, '\n')
    checked = 0
    matched = 0

    for line in lines
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, '#') && continue
        checked += 1
        fields = split(stripped)
        if length(fields) == 2 &&
           occursin(r"^[A-Za-z0-9+/=]+$", fields[1]) &&
           occursin(r"^-?[0-9]+$", fields[2])
            matched += 1
        end
        checked >= 6 && break
    end

    return checked > 0 && matched == checked
end

function _sniff_model_payload(path::AbstractString)::Symbol
    sample = _sample_file_bytes(path)
    _looks_tiktoken_text_payload(sample) && return :tiktoken

    # SentencePiece text probes may fail on truncated multibyte samples,
    # so inspect full bytes for this check.
    full_bytes = read(path)
    if _looks_text_sentencepiece(full_bytes)
        return :sentencepiece_model
    end

    if !isvalid(String, sample) || any(==(0x00), sample)
        return :sentencepiece_model
    end

    return :unknown
end

function _has_gpt2_assets(path::AbstractString)::Bool
    files = detect_tokenizer_files(path)
    return (
        (files.vocab_json !== nothing && files.merges_txt !== nothing) ||
        (files.encoder_json !== nothing && files.vocab_bpe !== nothing)
    )
end

function _as_wordpiece(tokenizer::AbstractSubwordTokenizer)::WordPieceTokenizer
    tokenizer isa WordPieceTokenizer || throw(ArgumentError("Format :wordpiece_vocab requires WordPieceTokenizer, got $(typeof(tokenizer))"))
    return tokenizer
end

function _as_bpe(tokenizer::AbstractSubwordTokenizer)::BPETokenizer
    if tokenizer isa BPETokenizer
        return tokenizer
    elseif tokenizer isa ByteBPETokenizer
        return tokenizer.base
    elseif tokenizer isa SentencePieceTokenizer && tokenizer.inner isa BPETokenizer
        return tokenizer.inner
    end

    throw(ArgumentError("Requested BPE export from incompatible tokenizer: $(typeof(tokenizer))"))
end

function _as_unigram(tokenizer::AbstractSubwordTokenizer)::UnigramTokenizer
    if tokenizer isa UnigramTokenizer
        return tokenizer
    elseif tokenizer isa SentencePieceTokenizer && tokenizer.inner isa UnigramTokenizer
        return tokenizer.inner
    end

    throw(ArgumentError("Requested Unigram export from incompatible tokenizer: $(typeof(tokenizer))"))
end

function _write_wordpiece_vocab(tokenizer::WordPieceTokenizer, path::String)::Nothing
    _write_token_lines(path, tokenizer.vocab.id_to_token)
    return nothing
end

function _write_bpe_files(tokenizer::BPETokenizer, outdir::AbstractString)::Nothing
    vocab_path = joinpath(outdir, "vocab.txt")
    merges_path = joinpath(outdir, "merges.txt")

    _write_token_lines(vocab_path, tokenizer.vocab.id_to_token)

    pairs = _sorted_merge_pairs(tokenizer)
    open(merges_path, "w") do io
        println(io, "# merges")
        for (left, right) in pairs
            println(io, left, " ", right)
        end
    end

    return nothing
end

function _write_unigram_tsv(tokenizer::UnigramTokenizer, path::String)::Nothing
    open(path, "w") do io
        for id in 1:length(tokenizer.vocab.id_to_token)
            token = tokenizer.vocab.id_to_token[id]
            score = tokenizer.logprobs[id]
            special = _special_symbol_for_id(tokenizer.vocab, id)
            if special === nothing
                println(io, token, "\t", score)
            else
                println(io, token, "\t", score, "\t", special)
            end
        end
    end

    return nothing
end

function _write_sentencepiece_model(tokenizer::AbstractSubwordTokenizer, path::String)::Nothing
    if tokenizer isa SentencePieceTokenizer
        _write_sentencepiece_model_from_parts(tokenizer.inner, tokenizer.whitespace_marker, path)
        return nothing
    elseif tokenizer isa UnigramTokenizer
        marker = isempty(tokenizer.whitespace_marker) ? "▁" : tokenizer.whitespace_marker
        _write_sentencepiece_model_from_parts(tokenizer, marker, path)
        return nothing
    elseif tokenizer isa BPETokenizer
        _write_sentencepiece_model_from_parts(tokenizer, "▁", path)
        return nothing
    elseif tokenizer isa ByteBPETokenizer
        _write_sentencepiece_model_from_parts(tokenizer.base, "▁", path)
        return nothing
    end

    throw(ArgumentError("Cannot export $(typeof(tokenizer)) as :sentencepiece_model"))
end

function _write_sentencepiece_model_from_parts(inner::UnigramTokenizer, marker::String, path::String)::Nothing
    open(path, "w") do io
        println(io, "type=unigram")
        println(io, "whitespace_marker=", marker)
        println(io, "unk_token=", inner.unk_token)

        for id in 1:length(inner.vocab.id_to_token)
            token = inner.vocab.id_to_token[id]
            score = inner.logprobs[id]
            special = _special_symbol_for_id(inner.vocab, id)
            if special === nothing
                println(io, "piece\t", token, "\t", score)
            else
                println(io, "piece\t", token, "\t", score, "\t", special)
            end
        end
    end

    return nothing
end

function _write_sentencepiece_model_from_parts(inner::BPETokenizer, marker::String, path::String)::Nothing
    open(path, "w") do io
        println(io, "type=bpe")
        println(io, "whitespace_marker=", marker)
        println(io, "unk_token=", inner.unk_token)

        for id in 1:length(inner.vocab.id_to_token)
            token = inner.vocab.id_to_token[id]
            special = _special_symbol_for_id(inner.vocab, id)
            if special === nothing
                println(io, "piece\t", token, "\t", -1.0)
            else
                println(io, "piece\t", token, "\t", -1.0, "\t", special)
            end
        end

        for (left, right) in _sorted_merge_pairs(inner)
            println(io, "merge\t", left, "\t", right)
        end
    end

    return nothing
end

function _sorted_merge_pairs(tokenizer::BPETokenizer)::Vector{Tuple{String,String}}
    ranked = collect(tokenizer.pair_ranks)
    sort!(ranked; by=last)
    return [pair for (pair, _) in ranked]
end

function _special_symbol_for_id(vocab::SubwordVocabulary, id::Int)::Union{Nothing,Symbol}
    for (symbol, sid) in vocab.special_token_ids
        sid == id && return symbol
    end
    return nothing
end

function _write_token_lines(path::String, tokens::Vector{String})::Nothing
    open(path, "w") do io
        for token in tokens
            println(io, token)
        end
    end
    return nothing
end

"""
Load GPT-2 / RoBERTa style BPE from `vocab.json` + `merges.txt`.

Example:
`load_bpe_gpt2("/path/to/vocab.json", "/path/to/merges.txt")`
"""
function load_bpe_gpt2(
    vocab_json::AbstractString,
    merges_txt::AbstractString;
    byte_level::Bool=true,
    unk_token::AbstractString="<unk>",
    model_name::Union{Nothing,AbstractString}=nothing,
    kwargs...,
)::AbstractSubwordTokenizer
    _ = kwargs
    vocab_json_path = normpath(String(vocab_json))
    merges_path = normpath(String(merges_txt))
    isfile(vocab_json_path) || throw(ArgumentError(
        "Missing GPT2 vocab JSON file: $vocab_json_path. " *
        "Expected files: vocab.json + merges.txt. " *
        "Example: load_bpe_gpt2(\"/path/to/vocab.json\", \"/path/to/merges.txt\")",
    ))
    isfile(merges_path) || throw(ArgumentError(
        "Missing GPT2 merges file: $merges_path. " *
        "Expected files: vocab.json + merges.txt. " *
        "Example: load_bpe_gpt2(\"/path/to/vocab.json\", \"/path/to/merges.txt\")",
    ))
    tokens = _read_gpt2_vocab_json(vocab_json_path)

    if !(String(unk_token) in tokens)
        pushfirst!(tokens, String(unk_token))
    end

    special_map = detect_special_tokens(tokens, String(unk_token))
    vocab = build_vocab(tokens; special_tokens=special_map)
    pairs = _read_merge_pairs(merges_path)
    pair_ranks = Dict{Tuple{String,String},Int}()
    for (i, pair) in enumerate(pairs)
        pair_ranks[pair] = i
    end

    name = model_name === nothing ? basename(vocab_json_path) : String(model_name)
    base_metadata = TokenizerMetadata(:bpe_gpt2, name, v"0.2.0", :none)
    base = BPETokenizer(vocab, pair_ranks, String(unk_token), nothing, base_metadata)

    byte_level || return base

    b2u, u2b = _byte_unicode_tables()
    metadata = TokenizerMetadata(:bytebpe, name, v"0.2.0", :none)
    return ByteBPETokenizer(base, b2u, u2b, metadata)
end

"""
Load GPT-2 encoder variant from `encoder.json` + `vocab.bpe`.

Example:
`load_bpe_encoder("/path/to/encoder.json", "/path/to/vocab.bpe")`
"""
function load_bpe_encoder(
    encoder_json::AbstractString,
    vocab_bpe::AbstractString;
    byte_level::Bool=true,
    kwargs...,
)::AbstractSubwordTokenizer
    encoder_path = normpath(String(encoder_json))
    merges_path = normpath(String(vocab_bpe))
    isfile(encoder_path) || throw(ArgumentError(
        "Missing encoder.json file: $encoder_path. " *
        "Expected files: encoder.json + vocab.bpe. " *
        "Example: load_bpe_encoder(\"/path/to/encoder.json\", \"/path/to/vocab.bpe\")",
    ))
    isfile(merges_path) || throw(ArgumentError(
        "Missing vocab.bpe file: $merges_path. " *
        "Expected files: encoder.json + vocab.bpe. " *
        "Example: load_bpe_encoder(\"/path/to/encoder.json\", \"/path/to/vocab.bpe\")",
    ))
    return load_bpe_gpt2(encoder_path, merges_path; byte_level=byte_level, kwargs...)
end

function _load_bpe_gpt2(
    path::AbstractString;
    kwargs...,
)::AbstractSubwordTokenizer
    vocab_json_path, merges_path = _resolve_gpt2_paths(path)
    return load_bpe_gpt2(vocab_json_path, merges_path; kwargs...)
end

function _resolve_gpt2_paths(path::AbstractString)::Tuple{String,String}
    resolved = normpath(String(path))
    if isdir(resolved)
        files = detect_tokenizer_files(resolved)
        if files.vocab_json !== nothing && files.merges_txt !== nothing
            return (files.vocab_json, files.merges_txt)
        elseif files.encoder_json !== nothing && files.vocab_bpe !== nothing
            return (files.encoder_json, files.vocab_bpe)
        end

        throw(ArgumentError(
            "Missing GPT2 BPE files in directory: $resolved. " *
            "Expected one of vocab.json+merges.txt or encoder.json+vocab.bpe. " *
            "Example: load_tokenizer(\"$resolved\"; format=:bpe_gpt2)",
        ))
    end

    isfile(resolved) || throw(ArgumentError("GPT2 BPE path does not exist: $resolved"))
    lower_name = lowercase(basename(resolved))

    if lower_name == "vocab.json"
        merges = joinpath(dirname(resolved), "merges.txt")
        isfile(merges) || throw(ArgumentError(
            "Missing merges.txt next to $resolved. " *
            "Example: load_bpe_gpt2(\"$resolved\", \"$merges\")",
        ))
        return (resolved, merges)
    elseif lower_name == "encoder.json"
        merges = joinpath(dirname(resolved), "vocab.bpe")
        isfile(merges) || throw(ArgumentError(
            "Missing vocab.bpe next to $resolved. " *
            "Example: load_bpe_encoder(\"$resolved\", \"$merges\")",
        ))
        return (resolved, merges)
    elseif lower_name == "vocab.bpe"
        vocab = joinpath(dirname(resolved), "encoder.json")
        isfile(vocab) || throw(ArgumentError(
            "Missing encoder.json next to $resolved. " *
            "Example: load_bpe_encoder(\"$vocab\", \"$resolved\")",
        ))
        return (vocab, resolved)
    end

    throw(ArgumentError(
        "GPT2 BPE path must point to vocab.json, encoder.json, vocab.bpe, or a model directory: $resolved",
    ))
end

function _resolve_bpe_encoder_paths(path::AbstractString)::Tuple{String,String}
    resolved = normpath(String(path))
    if isdir(resolved)
        files = detect_tokenizer_files(resolved)
        if files.encoder_json !== nothing && files.vocab_bpe !== nothing
            return (files.encoder_json, files.vocab_bpe)
        end
        throw(ArgumentError(
            "Missing encoder.json + vocab.bpe in directory: $resolved. " *
            "Example: load_tokenizer(\"$resolved\"; format=:bpe_encoder)",
        ))
    end

    isfile(resolved) || throw(ArgumentError("BPE encoder path does not exist: $resolved"))
    lower_name = lowercase(basename(resolved))
    if lower_name == "encoder.json"
        vocab_bpe = joinpath(dirname(resolved), "vocab.bpe")
        isfile(vocab_bpe) || throw(ArgumentError(
            "Missing vocab.bpe next to $resolved. " *
            "Example: load_bpe_encoder(\"$resolved\", \"$vocab_bpe\")",
        ))
        return (resolved, vocab_bpe)
    elseif lower_name == "vocab.bpe"
        encoder_json = joinpath(dirname(resolved), "encoder.json")
        isfile(encoder_json) || throw(ArgumentError(
            "Missing encoder.json next to $resolved. " *
            "Example: load_bpe_encoder(\"$encoder_json\", \"$resolved\")",
        ))
        return (encoder_json, resolved)
    end

    throw(ArgumentError(
        "BPE encoder path must point to encoder.json, vocab.bpe, or a directory containing both: $resolved",
    ))
end

function _read_gpt2_vocab_json(path::AbstractString)::Vector{String}
    pair_re = r"\"((?:\\.|[^\"])*)\"\s*:\s*(\d+)"
    pairs = Tuple{String,Int}[]

    content = read(path, String)
    for m in eachmatch(pair_re, content)
        token_raw = m.captures[1]
        id = parse(Int, m.captures[2])
        token = Base.unescape_string(token_raw)
        push!(pairs, (token, id))
    end

    isempty(pairs) && throw(ArgumentError("No token->id entries found in vocab.json: $path"))
    max_id = maximum(last(pair) for pair in pairs)
    tokens = Vector{Union{Nothing,String}}(undef, max_id + 1)
    fill!(tokens, nothing)

    for (token, id) in pairs
        idx = id + 1
        (1 <= idx <= length(tokens)) || throw(ArgumentError("Out-of-range token id $id in $path"))
        tokens[idx] === nothing || throw(ArgumentError("Duplicate token id $id in $path"))
        tokens[idx] = token
    end

    any(t -> t === nothing, tokens) && throw(ArgumentError("vocab.json has missing token ids in 0:$(max_id): $path"))
    return String[t::String for t in tokens]
end
