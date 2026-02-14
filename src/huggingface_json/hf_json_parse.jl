using JSON3

function _resolve_hf_json_path(path::AbstractString)::String
    if isdir(path)
        candidate = joinpath(path, "tokenizer.json")
        isfile(candidate) || throw(ArgumentError(
            "No tokenizer.json found in directory: $path. " *
            "Expected files: tokenizer.json. Example: load_hf_tokenizer_json(\"/path/to/tokenizer.json\")",
        ))
        return candidate
    end

    isfile(path) || throw(ArgumentError(
        "tokenizer.json path does not exist: $path. " *
        "Example: load_hf_tokenizer_json(\"/path/to/tokenizer.json\")",
    ))
    return String(path)
end

function parse_hf_tokenizer_json(path::AbstractString)::HFJSONSpec
    resolved_path = _resolve_hf_json_path(path)
    root = try
        JSON3.read(read(resolved_path, String))
    catch err
        throw(ArgumentError("Failed to parse tokenizer.json at $resolved_path: $(sprint(showerror, err))"))
    end

    _json_is_object(root) || throw(ArgumentError("tokenizer.json root must be an object: $resolved_path"))

    model = _parse_hf_model(_json_get_required(root, "model", "\$.model"), "\$.model")
    normalizer = _parse_hf_normalizer(_json_get(root, "normalizer"), "\$.normalizer")
    pretokenizer = _parse_hf_pretokenizer(_json_get(root, "pre_tokenizer"), "\$.pre_tokenizer")
    postprocessor = _parse_hf_postprocessor(_json_get(root, "post_processor"), "\$.post_processor")
    decoder = _parse_hf_decoder(_json_get(root, "decoder"), "\$.decoder")

    added_tokens = _parse_added_tokens(_json_get(root, "added_tokens"), "\$.added_tokens")
    added_token_ids = Dict{String,Int}(token.content => token.id for token in added_tokens)
    special_token_ids = Dict{String,Int}(token.content => token.id for token in added_tokens if token.special)
    _merge_template_special_ids!(special_token_ids, postprocessor)
    truncation = _parse_hf_truncation(_json_get(root, "truncation"), "\$.truncation")
    padding = _parse_hf_padding(_json_get(root, "padding"), "\$.padding")

    return HFJSONSpec(
        model,
        normalizer,
        pretokenizer,
        postprocessor,
        decoder,
        added_tokens,
        added_token_ids,
        special_token_ids,
        truncation,
        padding,
        resolved_path,
    )
end

function _parse_hf_model(model_obj, path::String)::HFJSONModelSpec
    _json_is_object(model_obj) || throw(ArgumentError("Tokenizer model block must be an object at $path"))
    type_name = lowercase(_json_get_required_string(model_obj, "type", "$path.type"))

    if type_name == "bpe"
        vocab = _parse_vocab_map(_json_get_required(model_obj, "vocab", "$path.vocab"), "$path.vocab")
        merges = _parse_merges(_json_get_required(model_obj, "merges", "$path.merges"), "$path.merges")
        unk_token = _json_get_string(model_obj, "unk_token", "<unk>")
        continuing_prefix = _json_get_optional_string(model_obj, "continuing_subword_prefix")
        end_suffix = _json_get_string(model_obj, "end_of_word_suffix", "")
        suffix = isempty(end_suffix) ? nothing : end_suffix
        fuse_unk = _json_get_bool(model_obj, "fuse_unk", false)
        byte_fallback = _json_get_bool(model_obj, "byte_fallback", false)
        dropout = _json_get_optional_float(model_obj, "dropout")
        if dropout !== nothing && dropout > 0
            throw(ArgumentError(
                "Unsupported model setting: dropout=$(dropout) at $path.dropout. " *
                "Workaround: set dropout to 0/null or export a deterministic tokenizer.json.",
            ))
        end
        return HFBPEModelSpec(
            vocab,
            merges,
            unk_token,
            false,
            continuing_prefix,
            suffix,
            fuse_unk,
            byte_fallback,
            dropout,
        )
    elseif type_name == "wordpiece"
        vocab = _parse_vocab_map(_json_get_required(model_obj, "vocab", "$path.vocab"), "$path.vocab")
        unk_token = _json_get_string(model_obj, "unk_token", "[UNK]")
        continuation = _json_get_string(model_obj, "continuing_subword_prefix", "##")
        max_input_chars = _json_get_int(model_obj, "max_input_chars_per_word", 100)
        return HFWordPieceModelSpec(vocab, unk_token, continuation, max_input_chars)
    elseif type_name == "unigram"
        return _parse_unigram_model(model_obj, path)
    end

    _unsupported_hf_component("model", type_name, path)
end

function _parse_unigram_model(model_obj, path::String)::HFUnigramModelSpec
    vocab_entries = _json_get_required(model_obj, "vocab", "$path.vocab")
    _json_is_array(vocab_entries) || throw(ArgumentError("Unigram vocab must be an array at $path.vocab"))

    tokens = String[]
    scores = Float64[]
    for (i, row) in enumerate(vocab_entries)
        row_path = "$path.vocab[$i]"
        _json_is_array(row) || throw(ArgumentError("Unigram vocab row must be an array at $row_path"))
        length(row) >= 2 || throw(ArgumentError("Unigram vocab row must contain [token, score] at $row_path"))
        token = String(row[1])
        score = _as_float(row[2], "$row_path[2]")
        push!(tokens, token)
        push!(scores, score)
    end

    unk_id_zero = _json_get_int(model_obj, "unk_id", 0)
    0 <= unk_id_zero < length(tokens) || throw(ArgumentError("Unigram unk_id out of range at $path.unk_id"))
    byte_fallback = _json_get_bool(model_obj, "byte_fallback", false)
    return HFUnigramModelSpec(tokens, scores, unk_id_zero + 1, byte_fallback)
end

function _parse_hf_normalizer(normalizer_obj, path::String)::HFJSONNormalizer
    normalizer_obj === nothing && return HFNoopNormalizer()

    _json_is_object(normalizer_obj) || throw(ArgumentError("Normalizer block must be an object at $path"))
    type_name = _json_get_required_string(normalizer_obj, "type", "$path.type")

    if type_name == "Lowercase"
        return HFLowercaseNormalizer()
    elseif type_name == "NFC"
        return HFNFCNormalizer()
    elseif type_name == "NFD"
        return HFNFDNormalizer()
    elseif type_name == "NFKC"
        return HFNFKCNormalizer()
    elseif type_name == "StripAccents"
        return HFStripAccentsNormalizer()
    elseif type_name == "Replace"
        pattern = _parse_split_regex(_json_get_required(normalizer_obj, "pattern", "$path.pattern"), "$path.pattern")
        replacement = _json_get_string(normalizer_obj, "content", "")
        return HFReplaceNormalizer(pattern, replacement)
    elseif type_name == "Prepend"
        prefix = _json_get_required_string(normalizer_obj, "prepend", "$path.prepend")
        return HFPrependNormalizer(prefix)
    elseif type_name == "Sequence"
        items_any = _json_get_required(normalizer_obj, "normalizers", "$path.normalizers")
        _json_is_array(items_any) || throw(ArgumentError("Normalizer sequence must be an array at $path.normalizers"))
        items = HFJSONNormalizer[]
        for (i, item) in enumerate(items_any)
            push!(items, _parse_hf_normalizer(item, "$path.normalizers[$i]"))
        end
        return HFSequenceNormalizer(items)
    end

    _unsupported_hf_component("normalizer", type_name, path)
end

function _parse_hf_pretokenizer(pre_obj, path::String)::HFJSONPreTokenizer
    pre_obj === nothing && return HFNoopPreTokenizer()

    _json_is_object(pre_obj) || throw(ArgumentError("Pre-tokenizer block must be an object at $path"))
    type_name = _json_get_required_string(pre_obj, "type", "$path.type")

    if type_name == "ByteLevel"
        add_prefix_space = _json_get_bool(pre_obj, "add_prefix_space", false)
        return HFByteLevelPreTokenizer(add_prefix_space)
    elseif type_name in ("Whitespace", "WhitespaceSplit")
        return HFWhitespacePreTokenizer()
    elseif type_name == "Metaspace"
        replacement = _json_get_string(pre_obj, "replacement", "▁")
        add_prefix_space = _json_get_bool(pre_obj, "add_prefix_space", false)
        prepend_scheme = lowercase(_json_get_string(pre_obj, "prepend_scheme", ""))
        prepend_scheme == "always" && (add_prefix_space = true)
        return HFMetaspacePreTokenizer(replacement, add_prefix_space)
    elseif type_name == "Split"
        regex = _parse_split_regex(_json_get_required(pre_obj, "pattern", "$path.pattern"), "$path.pattern")
        behavior_raw = lowercase(_json_get_string(pre_obj, "behavior", "isolated"))
        behavior_raw in ("isolated", "removed") || throw(ArgumentError("Unsupported Split behavior '$behavior_raw' at $path.behavior"))
        invert = _json_get_bool(pre_obj, "invert", false)
        invert && throw(ArgumentError("Unsupported Split invert=true at $path.invert"))
        return HFSplitPreTokenizer(regex, Symbol(behavior_raw))
    elseif type_name == "Digits"
        individual_digits = _json_get_bool(pre_obj, "individual_digits", false)
        return HFDigitsPreTokenizer(individual_digits)
    elseif type_name == "Punctuation"
        behavior_raw = lowercase(_json_get_string(pre_obj, "behavior", "isolated"))
        behavior_raw in ("isolated", "removed") || throw(ArgumentError("Unsupported Punctuation behavior '$behavior_raw' at $path.behavior"))
        return HFPunctuationPreTokenizer(Symbol(behavior_raw))
    elseif type_name == "Sequence"
        items_any = _json_get_required(pre_obj, "pretokenizers", "$path.pretokenizers")
        _json_is_array(items_any) || throw(ArgumentError("Pre-tokenizer sequence must be an array at $path.pretokenizers"))
        items = HFJSONPreTokenizer[]
        for (i, item) in enumerate(items_any)
            push!(items, _parse_hf_pretokenizer(item, "$path.pretokenizers[$i]"))
        end
        return HFSequencePreTokenizer(items)
    end

    _unsupported_hf_component("pre_tokenizer", type_name, path)
end

function _parse_hf_postprocessor(post_obj, path::String)::HFJSONPostProcessor
    post_obj === nothing && return HFNoopPostProcessor()

    _json_is_object(post_obj) || throw(ArgumentError("Post-processor block must be an object at $path"))
    type_name = _json_get_required_string(post_obj, "type", "$path.type")

    if type_name == "TemplateProcessing"
        single = _parse_template_items(_json_get_required(post_obj, "single", "$path.single"), "$path.single")
        pair_any = _json_get(post_obj, "pair")
        pair = pair_any === nothing ? String[] : _parse_template_items(pair_any, "$path.pair")
        specials = _parse_template_special_tokens(
            _json_get(post_obj, "special_tokens"),
            "$path.special_tokens",
        )
        return HFTemplateProcessingPostProcessor(single, pair, specials)
    elseif type_name == "ByteLevel"
        return HFByteLevelPostProcessor()
    elseif type_name == "BertProcessing"
        cls_token, cls_id = _parse_token_id_pair(_json_get_required(post_obj, "cls", "$path.cls"), "$path.cls")
        sep_token, sep_id = _parse_token_id_pair(_json_get_required(post_obj, "sep", "$path.sep"), "$path.sep")
        return HFBertProcessingPostProcessor(cls_token, cls_id, sep_token, sep_id)
    elseif type_name == "RobertaProcessing"
        cls_token, cls_id = _parse_token_id_pair(_json_get_required(post_obj, "cls", "$path.cls"), "$path.cls")
        sep_token, sep_id = _parse_token_id_pair(_json_get_required(post_obj, "sep", "$path.sep"), "$path.sep")
        return HFRobertaProcessingPostProcessor(cls_token, cls_id, sep_token, sep_id)
    elseif type_name == "Sequence"
        items_any = _json_get_required(post_obj, "processors", "$path.processors")
        _json_is_array(items_any) || throw(ArgumentError("Post-processor sequence must be an array at $path.processors"))
        items = HFJSONPostProcessor[]
        for (i, item) in enumerate(items_any)
            push!(items, _parse_hf_postprocessor(item, "$path.processors[$i]"))
        end
        return HFSequencePostProcessor(items)
    end

    _unsupported_hf_component("post_processor", type_name, path)
end

function _parse_hf_decoder(decoder_obj, path::String)::HFJSONDecoder
    decoder_obj === nothing && return HFNoopDecoder()

    _json_is_object(decoder_obj) || throw(ArgumentError("Decoder block must be an object at $path"))
    type_name = _json_get_required_string(decoder_obj, "type", "$path.type")

    if type_name == "ByteLevel"
        return HFByteLevelDecoder()
    elseif type_name == "WordPiece"
        prefix = _json_get_string(decoder_obj, "prefix", "##")
        return HFWordPieceDecoder(prefix)
    elseif type_name == "BPEDecoder"
        suffix = _json_get_string(decoder_obj, "suffix", "</w>")
        return HFBPEDecoder(suffix)
    elseif type_name == "Metaspace"
        replacement = _json_get_string(decoder_obj, "replacement", "▁")
        return HFMetaspaceDecoder(replacement)
    elseif type_name == "Sequence"
        items_any = _json_get_required(decoder_obj, "decoders", "$path.decoders")
        _json_is_array(items_any) || throw(ArgumentError("Decoder sequence must be an array at $path.decoders"))
        items = HFJSONDecoder[]
        for (i, item) in enumerate(items_any)
            push!(items, _parse_hf_decoder(item, "$path.decoders[$i]"))
        end
        return HFSequenceDecoder(items)
    end

    _unsupported_hf_component("decoder", type_name, path)
end

function _parse_template_items(items_any, path::String)::Vector{String}
    _json_is_array(items_any) || throw(ArgumentError("Template items must be an array at $path"))
    items = String[]
    for (i, item) in enumerate(items_any)
        push!(items, _parse_template_item(item, "$path[$i]"))
    end
    return items
end

function _parse_template_item(item, path::String)::String
    item isa AbstractString && return String(item)

    if _json_is_object(item)
        if _json_haskey(item, "Sequence")
            seq_obj = _json_get_required(item, "Sequence", "$path.Sequence")
            seq_id = uppercase(_json_get_required_string(seq_obj, "id", "$path.Sequence.id"))
            return "\$" * seq_id
        elseif _json_haskey(item, "SpecialToken")
            token_obj = _json_get_required(item, "SpecialToken", "$path.SpecialToken")
            return _json_get_required_string(token_obj, "id", "$path.SpecialToken.id")
        end
    end

    throw(ArgumentError("Unsupported TemplateProcessing item at $path"))
end

function _parse_template_special_tokens(tokens_any, path::String)::Dict{String,Int}
    tokens_any === nothing && return Dict{String,Int}()
    result = Dict{String,Int}()
    if _json_is_array(tokens_any)
        for (i, entry) in enumerate(tokens_any)
            entry_path = "$path[$i]"
            token_name, token_id = _parse_template_special_token_entry(entry, entry_path)
            result[token_name] = token_id
        end
        return result
    elseif _json_is_object(tokens_any)
        for (entry_key_any, entry) in tokens_any
            entry_key = String(entry_key_any)
            entry_path = "$path.$entry_key"
            token_name, token_id = _parse_template_special_token_entry(
                entry,
                entry_path;
                fallback_token=entry_key,
            )
            result[token_name] = token_id
        end
        return result
    end

    throw(ArgumentError("Template special_tokens must be an array or object at $path"))
end

function _parse_template_special_token_entry(
    entry,
    entry_path::String;
    fallback_token::Union{Nothing,String}=nothing,
)::Tuple{String,Int}
    _json_is_object(entry) || throw(ArgumentError("Template special_tokens entry must be an object at $entry_path"))

    token_name = if _json_haskey(entry, "id")
        _json_get_required_string(entry, "id", "$entry_path.id")
    elseif _json_haskey(entry, "tokens")
        tokens = _json_get_required(entry, "tokens", "$entry_path.tokens")
        _json_is_array(tokens) && !isempty(tokens) || throw(ArgumentError("Template tokens array empty at $entry_path.tokens"))
        String(tokens[1])
    elseif fallback_token !== nothing
        fallback_token
    else
        throw(ArgumentError("Template special token entry missing token id at $entry_path"))
    end

    ids_any = _json_get_required(entry, "ids", "$entry_path.ids")
    _json_is_array(ids_any) && !isempty(ids_any) || throw(ArgumentError("Template special token ids must be non-empty array at $entry_path.ids"))
    token_id_zero = _as_int(ids_any[1], "$entry_path.ids[1]")
    return (token_name, token_id_zero + 1)
end

function _parse_added_tokens(tokens_any, path::String)::Vector{HFAddedToken}
    tokens_any === nothing && return HFAddedToken[]
    _json_is_array(tokens_any) || throw(ArgumentError("added_tokens must be an array at $path"))

    result = HFAddedToken[]
    for (i, entry) in enumerate(tokens_any)
        entry_path = "$path[$i]"
        _json_is_object(entry) || continue
        _json_haskey(entry, "content") || continue
        _json_haskey(entry, "id") || continue

        content = _json_get_required_string(entry, "content", "$entry_path.content")
        id_zero = _as_int(_json_get_required(entry, "id", "$entry_path.id"), "$entry_path.id")
        push!(result, HFAddedToken(
            content,
            id_zero + 1,
            _json_get_bool(entry, "special", false),
            _json_get_bool(entry, "single_word", false),
            _json_get_bool(entry, "lstrip", false),
            _json_get_bool(entry, "rstrip", false),
            _json_get_bool(entry, "normalized", true),
        ))
    end
    return result
end

function _parse_hf_truncation(obj, path::String)::Union{Nothing,NamedTuple}
    obj === nothing && return nothing
    _json_is_object(obj) || throw(ArgumentError("truncation must be an object/null at $path"))
    return (
        max_length = _json_get_int(obj, "max_length", 0),
        strategy = _json_get_string(obj, "strategy", "longest_first"),
        stride = _json_get_int(obj, "stride", 0),
        direction = _json_get_string(obj, "direction", "right"),
    )
end

function _parse_hf_padding(obj, path::String)::Union{Nothing,NamedTuple}
    obj === nothing && return nothing
    _json_is_object(obj) || throw(ArgumentError("padding must be an object/null at $path"))
    return (
        strategy = _json_get_string(obj, "strategy", "batch_longest"),
        direction = _json_get_string(obj, "direction", "right"),
        pad_id = _json_get_int(obj, "pad_id", 0) + 1,
        pad_type_id = _json_get_int(obj, "pad_type_id", 0),
        pad_token = _json_get_string(obj, "pad_token", "[PAD]"),
        length = _json_get_int(obj, "length", 0),
    )
end

function _merge_template_special_ids!(
    all_specials::Dict{String,Int},
    post::HFJSONPostProcessor,
)::Nothing
    if post isa HFTemplateProcessingPostProcessor
        for (token, id) in post.special_tokens
            all_specials[token] = id
        end
    elseif post isa HFBertProcessingPostProcessor
        all_specials[post.cls_token] = post.cls_id
        all_specials[post.sep_token] = post.sep_id
    elseif post isa HFRobertaProcessingPostProcessor
        all_specials[post.cls_token] = post.cls_id
        all_specials[post.sep_token] = post.sep_id
    elseif post isa HFSequencePostProcessor
        for item in post.items
            _merge_template_special_ids!(all_specials, item)
        end
    end
    return nothing
end

function _parse_vocab_map(vocab_any, path::String)::Vector{String}
    _json_is_object(vocab_any) || throw(ArgumentError("Vocabulary map must be an object at $path"))
    pairs = Tuple{String,Int}[]
    for (token, id_any) in vocab_any
        push!(pairs, (String(token), _as_int(id_any, path)))
    end
    return _ordered_tokens_from_id_pairs(pairs, path)
end

function _ordered_tokens_from_id_pairs(
    pairs::Vector{Tuple{String,Int}},
    path::String,
)::Vector{String}
    isempty(pairs) && throw(ArgumentError("Vocabulary map is empty at $path"))
    max_id = maximum(last(pair) for pair in pairs)
    max_id >= 0 || throw(ArgumentError("Vocabulary ids must be non-negative at $path"))

    tokens = Vector{Union{Nothing,String}}(undef, max_id + 1)
    fill!(tokens, nothing)

    for (token, id_zero) in pairs
        id_zero >= 0 || throw(ArgumentError("Vocabulary id must be non-negative at $path"))
        index = id_zero + 1
        tokens[index] === nothing || throw(ArgumentError("Duplicate vocabulary id $id_zero at $path"))
        tokens[index] = token
    end

    any(t -> t === nothing, tokens) && throw(ArgumentError("Vocabulary ids must be contiguous starting at 0 at $path"))
    return String[t::String for t in tokens]
end

function _parse_merges(merges_any, path::String)::Vector{Tuple{String,String}}
    _json_is_array(merges_any) || throw(ArgumentError("BPE merges must be an array at $path"))

    merges = Tuple{String,String}[]
    for (i, entry) in enumerate(merges_any)
        entry_path = "$path[$i]"
        if entry isa AbstractString
            fields = split(String(entry))
            length(fields) == 2 || throw(ArgumentError("Invalid merge entry at $entry_path"))
            push!(merges, (fields[1], fields[2]))
        elseif _json_is_array(entry)
            length(entry) == 2 || throw(ArgumentError("Invalid merge entry at $entry_path"))
            push!(merges, (String(entry[1]), String(entry[2])))
        else
            throw(ArgumentError("Invalid merge entry type at $entry_path"))
        end
    end

    return merges
end

function _parse_split_regex(pattern_any, path::String)::Regex
    pattern = if pattern_any isa AbstractString
        String(pattern_any)
    elseif _json_is_object(pattern_any)
        if _json_haskey(pattern_any, "Regex")
            _json_get_required_string(pattern_any, "Regex", "$path.Regex")
        elseif _json_haskey(pattern_any, "String")
            _json_get_required_string(pattern_any, "String", "$path.String")
        else
            throw(ArgumentError("Unsupported split pattern object at $path"))
        end
    else
        throw(ArgumentError("Unsupported split pattern type at $path"))
    end

    try
        return Regex(pattern)
    catch err
        throw(ArgumentError("Invalid split regex at $path: $(sprint(showerror, err))"))
    end
end

function _parse_token_id_pair(value, path::String)::Tuple{String,Int}
    if _json_is_array(value)
        length(value) == 2 || throw(ArgumentError("Expected [token, id] pair at $path"))
        token = String(value[1])
        id = _as_int(value[2], "$path[2]") + 1
        return (token, id)
    elseif _json_is_object(value)
        token = _json_get_required_string(value, "token", "$path.token")
        id = _as_int(_json_get_required(value, "id", "$path.id"), "$path.id") + 1
        return (token, id)
    end
    throw(ArgumentError("Expected token/id pair at $path"))
end

function _unsupported_hf_component(component::String, type_name::String, path::String)
    throw(ArgumentError(
        "Unsupported $component type: $type_name at $path. " *
        "Workaround: use vocab.json + merges.txt if present, or export a simplified tokenizer.json.",
    ))
end

function _json_is_object(value)::Bool
    return value !== nothing && (value isa AbstractDict || string(typeof(value)) == "JSON3.Object")
end

function _json_is_array(value)::Bool
    return value !== nothing && (value isa AbstractVector || occursin("JSON3.Array", string(typeof(value))))
end

function _json_haskey(obj, key::String)::Bool
    if obj isa AbstractDict
        return haskey(obj, key) || haskey(obj, Symbol(key))
    end
    return haskey(obj, Symbol(key))
end

function _json_get(obj, key::String)
    if obj isa AbstractDict
        if haskey(obj, key)
            return obj[key]
        elseif haskey(obj, Symbol(key))
            return obj[Symbol(key)]
        end
    else
        if haskey(obj, Symbol(key))
            return obj[Symbol(key)]
        end
    end
    return nothing
end

function _json_get_required(obj, key::String, path::String)
    value = _json_get(obj, key)
    value === nothing && throw(ArgumentError("Missing required field at $path"))
    return value
end

function _json_get_string(obj, key::String, default::String)::String
    value = _json_get(obj, key)
    value === nothing && return default
    return String(value)
end

function _json_get_optional_string(obj, key::String)::Union{Nothing,String}
    value = _json_get(obj, key)
    value === nothing && return nothing
    value isa AbstractString || throw(ArgumentError("Expected string at field '$key'"))
    text = String(value)
    isempty(text) && return nothing
    return text
end

function _json_get_required_string(obj, key::String, path::String)::String
    value = _json_get_required(obj, key, path)
    value isa AbstractString || throw(ArgumentError("Expected string at $path"))
    return String(value)
end

function _json_get_bool(obj, key::String, default::Bool)::Bool
    value = _json_get(obj, key)
    value === nothing && return default
    value isa Bool || throw(ArgumentError("Expected boolean at field '$key'"))
    return value
end

function _json_get_int(obj, key::String, default::Int)::Int
    value = _json_get(obj, key)
    value === nothing && return default
    return _as_int(value, key)
end

function _json_get_optional_float(obj, key::String)::Union{Nothing,Float64}
    value = _json_get(obj, key)
    value === nothing && return nothing
    return _as_float(value, key)
end

function _as_int(value, path::String)::Int
    if value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("Expected finite integer at $path"))
        floor(Int, value) == value || throw(ArgumentError("Expected integer value at $path"))
        return Int(value)
    end
    throw(ArgumentError("Expected integer value at $path"))
end

function _as_float(value, path::String)::Float64
    if value isa Real
        return Float64(value)
    end
    throw(ArgumentError("Expected numeric value at $path"))
end
