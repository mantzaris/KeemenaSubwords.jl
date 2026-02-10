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
"""
function load_tokenizer(
    paths::Tuple{<:AbstractString,<:AbstractString};
    format::Symbol=:bpe,
    kwargs...,
)::AbstractSubwordTokenizer
    vocab_path, merges_path = paths
    selected_format = _canonical_load_format(format)

    if selected_format === :bpe
        return load_bpe(vocab_path, merges_path; kwargs...)
    elseif selected_format === :bytebpe
        return load_bytebpe(vocab_path, merges_path; kwargs...)
    end

    throw(ArgumentError("Tuple model path loading supports only :bpe/:bpe_gpt2 and :bytebpe, got: $format"))
end

"""
Load tokenizer from a named specification.

Examples:
- `(format=:wordpiece, path="/.../vocab.txt")`
- `(format=:bpe_gpt2, vocab="/.../vocab.txt", merges="/.../merges.txt")`
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

    if haskey(spec, :model_file)
        return load_sentencepiece(String(spec[:model_file]); kwargs...)
    end

    if haskey(spec, :encoding_file)
        return load_tiktoken(String(spec[:encoding_file]); kwargs...)
    end

    throw(ArgumentError(
        "Tokenizer spec requires one of: :path, (:vocab,:merges), (:vocab_json,:merges_txt), " *
        "(:encoder_json,:vocab_bpe), :tokenizer_json, :model_file, or :encoding_file.",
    ))
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

    sentencepiece_candidates = String[]
    for filename in ("spm.model", "spiece.model", "tokenizer.model", "tokenizer.model.v3", "sentencepiece.bpe.model")
        path = joinpath(root, filename)
        isfile(path) && push!(sentencepiece_candidates, path)
    end

    tiktoken_files = filter(p -> endswith(lowercase(p), ".tiktoken"), readdir(root; join=true))

    return (
        dir = root,
        tokenizer_json = isfile(joinpath(root, "tokenizer.json")) ? joinpath(root, "tokenizer.json") : nothing,
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
