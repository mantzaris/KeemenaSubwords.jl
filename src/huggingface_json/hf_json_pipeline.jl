(tokenizer::HuggingFaceJSONTokenizer)(text::AbstractString)::Vector{String} = tokenize(tokenizer, text)

function normalize(
    tokenizer::HuggingFaceJSONTokenizer,
    text::AbstractString,
)::String
    return _apply_hf_normalizer(tokenizer.normalizer, String(text))
end

requires_tokenizer_normalization(::HuggingFaceJSONTokenizer)::Bool = true

function tokenize(tokenizer::HuggingFaceJSONTokenizer, text::AbstractString)::Vector{String}
    ids = encode(tokenizer, text; add_special_tokens=false)
    return String[id_to_token(tokenizer, id) for id in ids]
end

function encode(
    tokenizer::HuggingFaceJSONTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
    assume_normalized::Bool=false,
)::Vector{Int}
    segments = _segment_hf_input(tokenizer, String(text); assume_normalized=assume_normalized)
    ids = Int[]

    for seg in segments
        if seg.kind == :added
            push!(ids, seg.id)
        else
            append!(ids, _encode_hf_text_segment(tokenizer, seg.text))
        end
    end

    add_special_tokens || return ids
    return _apply_hf_postprocessor(tokenizer.postprocessor, ids, tokenizer)
end

function _encode_hf_text_segment(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
)::Vector{Int}
    effective_pre = _hf_effective_pretokenizer_for_offsets(tokenizer.pretokenizer)
    if effective_pre isa HFByteLevelPreTokenizer
        return _encode_hf_bytelevel_segment(tokenizer, text, effective_pre)
    end

    pretokenized = _apply_hf_pretokenizer(tokenizer.pretokenizer, text)
    return _encode_hf_model_segment(tokenizer, pretokenized)
end

function decode(tokenizer::HuggingFaceJSONTokenizer, ids::AbstractVector{Int})::String
    filtered_ids = _hf_decode_filter_ids(tokenizer, ids)
    text = decode(tokenizer.base, filtered_ids)
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

function _hf_decode_filter_ids(
    tokenizer::HuggingFaceJSONTokenizer,
    ids::AbstractVector{Int},
)::Vector{Int}
    skip_ids = Set{Int}()
    for symbol in (:pad, :cls, :sep, :bos, :eos)
        special_id = get(tokenizer.special_token_ids, symbol, nothing)
        special_id === nothing || push!(skip_ids, special_id)
    end

    isempty(skip_ids) && return Int[id for id in ids]
    return Int[id for id in ids if !(id in skip_ids)]
end

function _segment_hf_input(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    ;
    assume_normalized::Bool=false,
)::Vector{NamedTuple{(:kind, :text, :id),Tuple{Symbol,String,Int}}}
    special_pass = _split_with_added_patterns(text, tokenizer.special_added_patterns)
    out = NamedTuple{(:kind, :text, :id),Tuple{Symbol,String,Int}}[]

    for seg in special_pass
        if seg.kind == :added
            push!(out, seg)
            continue
        end

        raw_pass = _split_with_added_patterns(seg.text, tokenizer.raw_added_patterns)
        for raw_seg in raw_pass
            if raw_seg.kind == :added
                push!(out, raw_seg)
                continue
            end

            normalized = assume_normalized ? raw_seg.text : _apply_hf_normalizer(tokenizer.normalizer, raw_seg.text)
            normalized_pass = _split_with_added_patterns(normalized, tokenizer.normalized_added_patterns)
            append!(out, normalized_pass)
        end
    end

    return out
end

function _split_with_added_patterns(
    text::String,
    patterns::Vector{HFAddedTokenPattern},
)::Vector{NamedTuple{(:kind, :text, :id),Tuple{Symbol,String,Int}}}
    isempty(text) && return NamedTuple{(:kind, :text, :id),Tuple{Symbol,String,Int}}[]
    isempty(patterns) && return [(kind=:text, text=text, id=0)]

    spans = NamedTuple{(:kind, :text, :id),Tuple{Symbol,String,Int}}[]
    i = firstindex(text)
    stop = lastindex(text)
    chunk_start = i

    while i <= stop
        best = _best_added_match(text, i, patterns)
        if best === nothing
            i = nextind(text, i)
            continue
        end

        if chunk_start < best.start
            prev_idx = prevind(text, best.start)
            push!(spans, (kind=:text, text=String(SubString(text, chunk_start, prev_idx)), id=0))
        end

        push!(spans, (kind=:added, text=best.content, id=best.id))
        i = nextind(text, best.stop)
        chunk_start = i
    end

    if chunk_start <= stop
        push!(spans, (kind=:text, text=String(SubString(text, chunk_start, stop)), id=0))
    end

    return spans
end

function _best_added_match(
    text::String,
    idx::Int,
    patterns::Vector{HFAddedTokenPattern},
)::Union{Nothing,NamedTuple}
    best = nothing
    for pattern in patterns
        match = _match_added_pattern(text, idx, pattern)
        match === nothing && continue

        if best === nothing
            best = match
            continue
        end

        if match.stop > best.stop || (match.stop == best.stop && length(match.content) > length(best.content))
            best = match
        end
    end
    return best
end

function _match_added_pattern(
    text::String,
    idx::Int,
    pattern::HFAddedTokenPattern,
)::Union{Nothing,NamedTuple}
    stop = lastindex(text)
    core_start = idx

    if pattern.lstrip
        while core_start <= stop && isspace(text[core_start])
            core_start = nextind(text, core_start)
        end
    end

    core_start <= stop || return nothing
    core_stop = _match_literal_at(text, core_start, pattern.content)
    core_stop === nothing && return nothing

    if pattern.single_word && !_is_single_word_span(text, core_start, core_stop)
        return nothing
    end

    span_stop = core_stop
    if pattern.rstrip
        next_pos = nextind(text, span_stop)
        while next_pos <= stop && isspace(text[next_pos])
            span_stop = next_pos
            next_pos = nextind(text, next_pos)
        end
    end

    return (start=idx, stop=span_stop, content=pattern.content, id=pattern.id)
end

function _match_literal_at(
    text::String,
    start_idx::Int,
    literal::String,
)::Union{Nothing,Int}
    isempty(literal) && return nothing

    t_idx = start_idx
    t_stop = lastindex(text)
    for c in literal
        t_idx <= t_stop || return nothing
        text[t_idx] == c || return nothing
        t_idx = nextind(text, t_idx)
    end

    return prevind(text, t_idx)
end

function _is_word_char(c::Char)::Bool
    return isletter(c) || isnumeric(c) || c == '_'
end

function _is_single_word_span(text::String, start_idx::Int, stop_idx::Int)::Bool
    first_idx = firstindex(text)
    last_idx = lastindex(text)

    if start_idx > first_idx
        left = text[prevind(text, start_idx)]
        _is_word_char(left) && return false
    end

    if stop_idx < last_idx
        right = text[nextind(text, stop_idx)]
        _is_word_char(right) && return false
    end

    return true
end

function _encode_hf_model_segment(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
)::Vector{Int}
    model = tokenizer.model
    base = tokenizer.base

    if model isa HFBPEModelSpec
        return _encode_hf_bpe_segment(base, model, text)
    elseif model isa HFUnigramModelSpec
        return _encode_hf_unigram_segment(base, model, text)
    end

    return encode(base, text; add_special_tokens=false)
end

function _encode_hf_bytelevel_segment(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    pre::HFByteLevelPreTokenizer,
)::Vector{Int}
    if !pre.use_regex
        pretokenized = _apply_hf_pretokenizer(pre, text)
        return _encode_hf_model_segment(tokenizer, pretokenized)
    end

    if tokenizer.base isa ByteBPETokenizer
        base = tokenizer.base::ByteBPETokenizer
        if !_hf_bytelevel_can_split(base)
            pretokenized = _apply_hf_pretokenizer(pre, text)
            return _encode_hf_model_segment(tokenizer, pretokenized)
        end

        raw_splits, _ = _hf_bytelevel_raw_splits_with_work_spans(
            text;
            add_prefix_space=pre.add_prefix_space,
            use_regex=pre.use_regex,
        )
        ids = Int[]
        for split in raw_splits
            append!(ids, _encode_hf_bytelevel_piece(base, split.piece))
        end
        return ids
    end

    pretokenized = _apply_hf_pretokenizer(pre, text)
    return _encode_hf_model_segment(tokenizer, pretokenized)
end

function _hf_bytelevel_can_split(tokenizer::ByteBPETokenizer)::Bool
    byte_space = string(tokenizer.byte_to_unicode[Int(UInt8(' ')) + 1])
    return haskey(tokenizer.base.vocab.token_to_id, byte_space)
end

function _encode_hf_bytelevel_piece(
    tokenizer::ByteBPETokenizer,
    piece::String,
)::Vector{Int}
    isempty(piece) && return Int[]
    merged_tokens = _tokenize_byte_word(tokenizer, piece)
    return Int[token_to_id(tokenizer, token) for token in merged_tokens]
end

function _encode_hf_bpe_segment(
    base::AbstractSubwordTokenizer,
    model::HFBPEModelSpec,
    text::String,
)::Vector{Int}
    if base isa ByteBPETokenizer || !model.byte_fallback
        return encode(base, text; add_special_tokens=false)
    end

    base_bpe = base::BPETokenizer
    normalized = normalize_text(text)
    ids = Int[]

    for word in eachsplit(normalized)
        pieces = _tokenize_bpe_word(base_bpe, String(word); append_end_marker=true)
        if length(pieces) == 1 && pieces[1] == base_bpe.unk_token
            fallback = _byte_fallback_tokens(base_bpe.vocab, String(word), base_bpe.end_of_word_marker, base_bpe.unk_token)
            append!(ids, Int[token_to_id(base_bpe, piece) for piece in fallback])
        else
            append!(ids, Int[token_to_id(base_bpe, piece) for piece in pieces])
        end
    end

    return ids
end

function _encode_hf_unigram_segment(
    base::AbstractSubwordTokenizer,
    model::HFUnigramModelSpec,
    text::String,
)::Vector{Int}
    if !(base isa UnigramTokenizer) || !model.byte_fallback
        return encode(base, text; add_special_tokens=false)
    end

    uni = base::UnigramTokenizer
    normalized = normalize_text(text)
    ids = Int[]

    for word in eachsplit(normalized)
        input = isempty(uni.whitespace_marker) ? String(word) : string(uni.whitespace_marker, String(word))
        pieces = _viterbi_segment(uni, input)
        if length(pieces) == 1 && pieces[1] == uni.unk_token
            fallback = _byte_fallback_tokens(uni.vocab, String(word), nothing, uni.unk_token)
            append!(ids, Int[token_to_id(uni, piece) for piece in fallback])
        else
            append!(ids, Int[token_to_id(uni, piece) for piece in pieces])
        end
    end

    return ids
end

function _byte_fallback_tokens(
    vocab::SubwordVocabulary,
    word::String,
    end_of_word_marker::Union{Nothing,String},
    unk_token::String,
)::Vector{String}
    b2u, _ = _byte_unicode_tables()
    unicode_tokens = String[]
    for b in codeunits(word)
        piece = string(b2u[Int(b) + 1])
        haskey(vocab.token_to_id, piece) || (unicode_tokens = String[]; break)
        push!(unicode_tokens, piece)
    end

    if !isempty(unicode_tokens)
        if end_of_word_marker !== nothing && haskey(vocab.token_to_id, end_of_word_marker)
            push!(unicode_tokens, end_of_word_marker)
        end
        return unicode_tokens
    end

    hex_tokens = String[]
    for b in codeunits(word)
        piece = "<0x$(uppercase(string(Int(b), base=16, pad=2)))>"
        haskey(vocab.token_to_id, piece) || return String[unk_token]
        push!(hex_tokens, piece)
    end
    return isempty(hex_tokens) ? String[unk_token] : hex_tokens
end

function _is_hf_bert_chinese_char(c::Char)::Bool
    codepoint = Int(c)
    return (0x4E00 <= codepoint <= 0x9FFF) ||
           (0x3400 <= codepoint <= 0x4DBF) ||
           (0x20000 <= codepoint <= 0x2A6DF) ||
           (0x2A700 <= codepoint <= 0x2B73F) ||
           (0x2B740 <= codepoint <= 0x2B81F) ||
           (0x2B820 <= codepoint <= 0x2CEAF) ||
           (0xF900 <= codepoint <= 0xFAFF) ||
           (0x2F800 <= codepoint <= 0x2FA1F)
end

function _is_hf_bert_control(c::Char)::Bool
    isspace(c) && return false
    char_string = string(c)
    return occursin(r"^\p{Cc}$", char_string) || occursin(r"^\p{Cf}$", char_string)
end

function _apply_hf_bert_clean_text(text::String)::String
    buffer = IOBuffer()
    for c in text
        if c == '\0' || c == '\ufffd'
            continue
        end
        _is_hf_bert_control(c) && continue
        if isspace(c)
            write(buffer, ' ')
        else
            write(buffer, c)
        end
    end
    return String(take!(buffer))
end

function _apply_hf_bert_chinese_spacing(text::String)::String
    buffer = IOBuffer()
    for c in text
        if _is_hf_bert_chinese_char(c)
            write(buffer, ' ')
            write(buffer, c)
            write(buffer, ' ')
        else
            write(buffer, c)
        end
    end
    return String(take!(buffer))
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

function _apply_hf_normalizer(::HFNFDNormalizer, text::String)::String
    return Base.Unicode.normalize(text, :NFD)
end

function _apply_hf_normalizer(::HFNFKCNormalizer, text::String)::String
    return Base.Unicode.normalize(text, :NFKC)
end

function _apply_hf_normalizer(::HFStripAccentsNormalizer, text::String)::String
    decomposed = Base.Unicode.normalize(text, :NFD)
    return replace(decomposed, r"\p{M}+" => "")
end

function _apply_hf_normalizer(normalizer::HFBertNormalizer, text::String)::String
    out = text

    if normalizer.clean_text
        out = _apply_hf_bert_clean_text(out)
    end

    if normalizer.handle_chinese_chars
        out = _apply_hf_bert_chinese_spacing(out)
    end

    if normalizer.lowercase
        out = lowercase(out)
    end

    if effective_strip_accents(normalizer)
        decomposed = Base.Unicode.normalize(out, :NFD)
        out = replace(decomposed, r"\p{Mn}+" => "")
    end

    return out
end

function _apply_hf_normalizer(normalizer::HFReplaceNormalizer, text::String)::String
    return replace(text, normalizer.pattern => normalizer.replacement)
end

function _apply_hf_normalizer(normalizer::HFPrependNormalizer, text::String)::String
    return normalizer.prefix * text
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

function _is_hf_bert_punctuation(c::Char)::Bool
    codepoint = Int(c)
    return (0x21 <= codepoint <= 0x2F) ||
           (0x3A <= codepoint <= 0x40) ||
           (0x5B <= codepoint <= 0x60) ||
           (0x7B <= codepoint <= 0x7E) ||
           ispunct(c)
end

function _hf_bert_pretokenize_with_spans(
    text::String,
)::Vector{NamedTuple{(:piece, :start, :stop),Tuple{String,Int,Int}}}
    pieces = NamedTuple{(:piece, :start, :stop),Tuple{String,Int,Int}}[]
    isempty(text) && return pieces

    idx = firstindex(text)
    stop = lastindex(text)

    while idx <= stop
        current = text[idx]
        if isspace(current)
            idx = nextind(text, idx)
            continue
        end

        start_idx = idx
        if _is_hf_bert_punctuation(current)
            next_idx = nextind(text, idx)
            push!(pieces, (piece=string(current), start=start_idx, stop=next_idx))
            idx = next_idx
            continue
        end

        end_idx = idx
        while end_idx <= stop
            c = text[end_idx]
            if isspace(c) || _is_hf_bert_punctuation(c)
                break
            end
            end_idx = nextind(text, end_idx)
        end

        token_stop = end_idx
        token = String(SubString(text, start_idx, prevind(text, token_stop)))
        push!(pieces, (piece=token, start=start_idx, stop=token_stop))
        idx = token_stop
    end

    return pieces
end

function _apply_hf_pretokenizer(::HFBertPreTokenizer, text::String)::String
    pieces = _hf_bert_pretokenize_with_spans(text)
    isempty(pieces) && return ""
    return join([piece.piece for piece in pieces], " ")
end

const _HF_BYTELEVEL_GPT2_REGEX = Regex(
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
)

function _hf_bytelevel_raw_splits_with_work_spans(
    text::String;
    add_prefix_space::Bool,
    use_regex::Bool,
)::Tuple{
    Vector{NamedTuple{(:piece, :start, :stop),Tuple{String,Int,Int}}},
    Bool,
}
    prefix_added = add_prefix_space && !isempty(text) && !startswith(text, " ")
    working = prefix_added ? " " * text : text
    splits = NamedTuple{(:piece, :start, :stop),Tuple{String,Int,Int}}[]

    isempty(working) && return (splits, prefix_added)

    if use_regex
        for match in eachmatch(_HF_BYTELEVEL_GPT2_REGEX, working)
            piece = String(match.match)
            start_idx = match.offset
            stop_idx = start_idx + ncodeunits(piece)
            push!(splits, (piece=piece, start=start_idx, stop=stop_idx))
        end
    else
        stop_idx = ncodeunits(working) + offsets_index_base()
        push!(splits, (piece=working, start=offsets_index_base(), stop=stop_idx))
    end

    return (splits, prefix_added)
end

function _hf_bytelevel_map_piece(piece::String)::String
    byte_to_unicode, _ = _byte_unicode_tables()
    mapped = IOBuffer()
    for byte in codeunits(piece)
        write(mapped, byte_to_unicode[Int(byte) + 1])
    end
    return String(take!(mapped))
end

function _hf_bytelevel_work_span_to_original(
    start_idx::Int,
    stop_idx::Int;
    prefix_added::Bool,
    max_stop::Int,
)::Tuple{Int,Int}
    shift = prefix_added ? 1 : 0
    base = offsets_index_base()

    mapped_start = clamp(start_idx - shift, base, max_stop)
    mapped_stop = clamp(stop_idx - shift, base, max_stop)
    mapped_stop < mapped_start && (mapped_stop = mapped_start)
    return (mapped_start, mapped_stop)
end

function _hf_bytelevel_pretokenize_with_spans(
    text::String;
    add_prefix_space::Bool,
    use_regex::Bool,
)::Vector{Tuple{String,Tuple{Int,Int}}}
    raw_splits, prefix_added = _hf_bytelevel_raw_splits_with_work_spans(
        text;
        add_prefix_space=add_prefix_space,
        use_regex=use_regex,
    )
    max_stop = ncodeunits(text) + offsets_index_base()
    out = Tuple{String,Tuple{Int,Int}}[]

    for split in raw_splits
        mapped_piece = _hf_bytelevel_map_piece(split.piece)
        mapped_span = _hf_bytelevel_work_span_to_original(
            split.start,
            split.stop;
            prefix_added=prefix_added,
            max_stop=max_stop,
        )
        push!(out, (mapped_piece, mapped_span))
    end

    return out
end

function _apply_hf_pretokenizer(pre::HFByteLevelPreTokenizer, text::String)::String
    if pre.add_prefix_space && !isempty(text) && !startswith(text, " ")
        return " " * text
    end
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
        return replace(text, pre.pattern => (m -> " " * String(m) * " "))
    elseif pre.behavior == :removed
        return replace(text, pre.pattern => " ")
    end

    return text
end

function _apply_hf_pretokenizer(pre::HFDigitsPreTokenizer, text::String)::String
    if pre.individual_digits
        return replace(text, r"(\d)" => s" \1 ")
    end
    return replace(text, r"(\d+)" => s" \1 ")
end

function _apply_hf_pretokenizer(pre::HFPunctuationPreTokenizer, text::String)::String
    if pre.behavior == :isolated
        return replace(text, r"([[:punct:]])" => s" \1 ")
    elseif pre.behavior == :removed
        return replace(text, r"[[:punct:]]" => " ")
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
    post::HFBertProcessingPostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    _ = tokenizer
    return Int[post.cls_id; ids; post.sep_id]
end

function _apply_hf_postprocessor(
    post::HFRobertaProcessingPostProcessor,
    ids::Vector{Int},
    tokenizer::HuggingFaceJSONTokenizer,
)::Vector{Int}
    _ = tokenizer
    return Int[post.cls_id; ids; post.sep_id]
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
    _ = tokenizer
    return replace(text, decoder.prefix => "")
end

function _apply_hf_decoder(
    decoder::HFBPEDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = tokenizer
    return strip(replace(text, decoder.suffix => " "))
end

function _apply_hf_decoder(
    decoder::HFMetaspaceDecoder,
    text::String,
    tokenizer::HuggingFaceJSONTokenizer,
)::String
    _ = tokenizer
    decoded = replace(text, decoder.replacement => " ")
    if decoder.add_prefix_space
        return strip(decoded)
    end
    return decoded
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

function _hf_bert_offsets(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    tokens::Vector{String},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    offsets = Tuple{Int,Int}[]
    token_idx = 1

    for piece in _hf_bert_pretokenize_with_spans(text)
        piece_ids = _encode_hf_model_segment(tokenizer, piece.piece)
        piece_tokens = String[id_to_token(tokenizer, id) for id in piece_ids]
        piece_offsets = _token_offsets_for_tokens(
            tokenizer.base,
            piece.piece,
            piece_tokens,
            piece_ids,
        )
        piece_offsets === nothing && return nothing
        length(piece_offsets) == length(piece_tokens) || return nothing
        token_idx + length(piece_tokens) - 1 <= length(tokens) || return nothing

        for (i, piece_token) in enumerate(piece_tokens)
            tokens[token_idx] == piece_token || return nothing
            start_idx, stop_idx = piece_offsets[i]
            push!(offsets, (
                _advance_codeunits(piece.start, start_idx - 1),
                _advance_codeunits(piece.start, stop_idx - 1),
            ))
            token_idx += 1
        end
    end

    token_idx == length(tokens) + 1 || return nothing
    return offsets
end

function _hf_bytelevel_piece_offsets(
    tokenizer::ByteBPETokenizer,
    piece::String,
    piece_tokens::Vector{String},
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    offsets = Tuple{Int,Int}[]
    cursor = offsets_index_base()
    piece_end = ncodeunits(piece) + offsets_index_base()

    for token in piece_tokens
        if token == tokenizer.base.unk_token
            push!(offsets, (offsets_index_base(), piece_end))
            cursor = piece_end
            continue
        end

        piece_bytes = _bytebpe_piece_nbytes(tokenizer, token)
        piece_bytes === nothing && return nothing
        next_cursor = _advance_codeunits(cursor, piece_bytes)
        next_cursor <= piece_end || return nothing
        push!(offsets, (cursor, next_cursor))
        cursor = next_cursor
    end

    cursor <= piece_end || return nothing
    return offsets
end

function _hf_bytelevel_offsets(
    tokenizer::HuggingFaceJSONTokenizer,
    text::String,
    tokens::Vector{String},
    ids::Vector{Int},
    pre::HFByteLevelPreTokenizer,
)::Union{Nothing,Vector{Tuple{Int,Int}}}
    if !pre.use_regex
        pretokenized = _apply_hf_pretokenizer(pre, text)
        base_offsets = _token_offsets_for_tokens(tokenizer.base, pretokenized, tokens, ids)
        base_offsets === nothing && return nothing
        shift = pre.add_prefix_space && !isempty(text) && !startswith(text, " ") ? -1 : 0
        return shift == 0 ? base_offsets : _shift_offsets(base_offsets, shift)
    end

    if tokenizer.base isa ByteBPETokenizer
        base = tokenizer.base::ByteBPETokenizer
        if !_hf_bytelevel_can_split(base)
            pretokenized = _apply_hf_pretokenizer(pre, text)
            base_offsets = _token_offsets_for_tokens(base, pretokenized, tokens, ids)
            base_offsets === nothing && return nothing
            shift = pre.add_prefix_space && !isempty(text) && !startswith(text, " ") ? -1 : 0
            return shift == 0 ? base_offsets : _shift_offsets(base_offsets, shift)
        end
    end

    raw_splits, prefix_added = _hf_bytelevel_raw_splits_with_work_spans(
        text;
        add_prefix_space=pre.add_prefix_space,
        use_regex=pre.use_regex,
    )

    offsets = Tuple{Int,Int}[]
    token_idx = 1
    max_stop = ncodeunits(text) + offsets_index_base()

    for split in raw_splits
        piece_ids, piece_tokens, local_offsets = if tokenizer.base isa ByteBPETokenizer
            bytebpe = tokenizer.base::ByteBPETokenizer
            split_ids = _encode_hf_bytelevel_piece(bytebpe, split.piece)
            split_tokens = String[id_to_token(tokenizer, id) for id in split_ids]
            split_offsets = _hf_bytelevel_piece_offsets(bytebpe, split.piece, split_tokens)
            split_offsets === nothing && return nothing
            (split_ids, split_tokens, split_offsets)
        else
            pretokenized = _apply_hf_pretokenizer(pre, split.piece)
            split_ids = _encode_hf_model_segment(tokenizer, pretokenized)
            split_tokens = String[id_to_token(tokenizer, id) for id in split_ids]
            split_offsets = _token_offsets_for_tokens(
                tokenizer.base,
                pretokenized,
                split_tokens,
                split_ids,
            )
            split_offsets === nothing && return nothing
            (split_ids, split_tokens, split_offsets)
        end

        length(piece_ids) == length(piece_tokens) == length(local_offsets) || return nothing
        token_idx + length(piece_ids) - 1 <= length(tokens) || return nothing

        for i in eachindex(piece_ids)
            tokens[token_idx] == piece_tokens[i] || return nothing
            ids[token_idx] == piece_ids[i] || return nothing

            local_start, local_stop = local_offsets[i]
            work_start = split.start + local_start - offsets_index_base()
            work_stop = split.start + local_stop - offsets_index_base()
            global_offset = _hf_bytelevel_work_span_to_original(
                work_start,
                work_stop;
                prefix_added=prefix_added,
                max_stop=max_stop,
            )
            push!(offsets, global_offset)
            token_idx += 1
        end
    end

    token_idx == length(tokens) + 1 || return nothing
    return offsets
end

function _hf_bytelevel_trim_options(post::HFJSONPostProcessor)::Tuple{Bool,Bool}
    return (false, false)
end

function _hf_bytelevel_trim_options(post::HFByteLevelPostProcessor)::Tuple{Bool,Bool}
    return (post.trim_offsets, post.add_prefix_space)
end

function _hf_bytelevel_trim_options(post::HFRobertaProcessingPostProcessor)::Tuple{Bool,Bool}
    return (post.trim_offsets, post.add_prefix_space)
end

function _hf_bytelevel_trim_options(post::HFSequencePostProcessor)::Tuple{Bool,Bool}
    for item in post.items
        enabled, add_prefix_space = _hf_bytelevel_trim_options(item)
        enabled && return (enabled, add_prefix_space)
    end
    return (false, false)
end

function _hf_bytelevel_space_char(tokenizer::HuggingFaceJSONTokenizer)::Char
    if tokenizer.base isa ByteBPETokenizer
        bytebpe = tokenizer.base::ByteBPETokenizer
        return bytebpe.byte_to_unicode[Int(UInt8(' ')) + 1]
    end

    byte_to_unicode, _ = _byte_unicode_tables()
    return byte_to_unicode[Int(UInt8(' ')) + 1]
end

function _hf_bytelevel_is_trim_space(c::Char, byte_space_char::Char)::Bool
    return c == byte_space_char || isspace(c)
end

function _hf_bytelevel_trim_offsets!(
    tokens::Vector{String},
    offsets::Vector{Tuple{Int,Int}};
    add_prefix_space::Bool,
    byte_space_char::Char,
    original_text::Union{Nothing,String}=nothing,
)::Nothing
    length(tokens) == length(offsets) || return nothing
    base = offsets_index_base()
    first_input_is_space = original_text !== nothing &&
        !isempty(original_text) &&
        isspace(original_text[firstindex(original_text)])

    for i in eachindex(tokens)
        start_idx, stop_idx = offsets[i]
        has_span((start_idx, stop_idx)) || continue

        token = tokens[i]
        token_chars = collect(token)
        isempty(token_chars) && continue

        leading = 0
        trailing = 0

        while leading < length(token_chars) &&
              _hf_bytelevel_is_trim_space(token_chars[leading + 1], byte_space_char)
            leading += 1
        end

        while trailing < (length(token_chars) - leading) &&
              _hf_bytelevel_is_trim_space(token_chars[length(token_chars) - trailing], byte_space_char)
            trailing += 1
        end

        if add_prefix_space &&
           i == 1 &&
           leading > 0 &&
           start_idx == base &&
           !first_input_is_space &&
           token_chars[1] == byte_space_char
            leading -= 1
        end

        trimmed_start = min(start_idx + leading, stop_idx)
        trimmed_stop = max(stop_idx - trailing, trimmed_start)
        offsets[i] = (trimmed_start, trimmed_stop)
    end

    return nothing
end
