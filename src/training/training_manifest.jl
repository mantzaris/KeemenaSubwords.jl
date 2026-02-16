using JSON3

const TRAINING_MANIFEST_FILENAME = "keemena_training_manifest.json"
const TRAINING_MANIFEST_SCHEMA_VERSION = 1

struct TrainingManifestV1
    schema_version::Int
    trainer::Symbol
    load_format::Symbol
    export_format::Symbol
    files::Dict{String,String}
    load_kwargs::Dict{String,Any}
    training_config::Dict{String,Any}
    metadata::Dict{String,Any}
    warnings::Vector{String}
end

"""
    write_training_manifest(outdir, manifest)

Write a `TrainingManifestV1` to `outdir/keemena_training_manifest.json`.
"""
function write_training_manifest(
    outdir::AbstractString,
    manifest::TrainingManifestV1,
)::Nothing
    bundle_dir = normpath(String(outdir))
    isdir(bundle_dir) || throw(ArgumentError(
        "Bundle directory does not exist: $bundle_dir",
    ))

    payload = _manifest_payload(manifest)
    manifest_path = joinpath(bundle_dir, TRAINING_MANIFEST_FILENAME)
    open(manifest_path, "w") do io
        JSON3.write(io, payload)
    end
    return nothing
end

"""
    read_training_manifest(outdir) -> TrainingManifestV1

Read `outdir/keemena_training_manifest.json`.
"""
function read_training_manifest(
    outdir::AbstractString,
)::TrainingManifestV1
    bundle_dir = normpath(String(outdir))
    manifest_path = joinpath(bundle_dir, TRAINING_MANIFEST_FILENAME)
    isfile(manifest_path) || throw(ArgumentError(
        "Training manifest not found: $manifest_path",
    ))

    root = try
        JSON3.read(read(manifest_path, String))
    catch err
        throw(ArgumentError(
            "Failed to parse training manifest at $manifest_path: $(sprint(showerror, err))",
        ))
    end

    _manifest_is_object(root) || throw(ArgumentError(
        "Training manifest root must be an object: $manifest_path",
    ))

    schema_version = _manifest_as_int(
        _manifest_get_required(root, "schema_version", "\$.schema_version"),
        "\$.schema_version",
    )
    schema_version == TRAINING_MANIFEST_SCHEMA_VERSION || throw(ArgumentError(
        "Unsupported training manifest schema_version=$schema_version at $manifest_path. " *
        "Supported schema_version=$(TRAINING_MANIFEST_SCHEMA_VERSION).",
    ))

    trainer = Symbol(_manifest_as_string(
        _manifest_get_required(root, "trainer", "\$.trainer"),
        "\$.trainer",
    ))
    load_format = Symbol(_manifest_as_string(
        _manifest_get_required(root, "load_format", "\$.load_format"),
        "\$.load_format",
    ))
    export_format = Symbol(_manifest_as_string(
        _manifest_get_required(root, "export_format", "\$.export_format"),
        "\$.export_format",
    ))

    files = _manifest_as_string_dict(
        _manifest_get_required(root, "files", "\$.files"),
        "\$.files",
    )
    load_kwargs = _manifest_as_any_dict(
        _manifest_get_required(root, "load_kwargs", "\$.load_kwargs"),
        "\$.load_kwargs",
    )
    training_config = _manifest_as_any_dict(
        _manifest_get_required(root, "training_config", "\$.training_config"),
        "\$.training_config",
    )
    metadata = _manifest_as_any_dict(
        _manifest_get_required(root, "metadata", "\$.metadata"),
        "\$.metadata",
    )
    warnings = _manifest_as_string_vector(
        _manifest_get_required(root, "warnings", "\$.warnings"),
        "\$.warnings",
    )

    return TrainingManifestV1(
        schema_version,
        trainer,
        load_format,
        export_format,
        files,
        load_kwargs,
        training_config,
        metadata,
        warnings,
    )
end

"""
    save_training_bundle(result, outdir; export_format=:auto, overwrite=false)

Export a trained tokenizer result and write a deterministic v1 manifest into
`outdir` so the bundle can be reloaded later with `load_training_bundle`.
"""
function save_training_bundle(
    result::TrainingResult,
    outdir::AbstractString;
    export_format::Symbol=:auto,
    overwrite::Bool=false,
)::Nothing
    bundle_dir = normpath(String(outdir))

    if isfile(bundle_dir)
        throw(ArgumentError("Cannot write training bundle: path is a file: $bundle_dir"))
    elseif isdir(bundle_dir)
        overwrite || throw(ArgumentError(
            "Training bundle directory already exists: $bundle_dir (set overwrite=true to reuse it)",
        ))
    else
        mkpath(bundle_dir)
    end

    target_export_format = _resolve_bundle_export_format(result.tokenizer, export_format)
    _bundle_export_tokenizer(result.tokenizer, bundle_dir; format=target_export_format)

    manifest = _manifest_from_result(result, bundle_dir, target_export_format)
    write_training_manifest(bundle_dir, manifest)
    return nothing
end

"""
    load_training_bundle(outdir) -> AbstractSubwordTokenizer

Load a tokenizer bundle previously written by `save_training_bundle`.
"""
function load_training_bundle(
    outdir::AbstractString,
)::AbstractSubwordTokenizer
    bundle_dir = normpath(String(outdir))
    manifest = read_training_manifest(bundle_dir)

    files = _resolve_manifest_file_paths(bundle_dir, manifest.files)
    load_kwargs = copy(manifest.load_kwargs)
    special_tokens = _manifest_special_tokens_spec(get(load_kwargs, "special_tokens", nothing))
    delete!(load_kwargs, "special_tokens")
    kwargs = _manifest_kwargs_pairs(load_kwargs)

    tokenizer = if manifest.load_format in (:bpe, :bytebpe)
        vocab_path = get(files, "vocab_txt", nothing)
        merges_path = get(files, "merges_txt", nothing)
        vocab_path === nothing && throw(ArgumentError("Manifest files missing key 'vocab_txt'"))
        merges_path === nothing && throw(ArgumentError("Manifest files missing key 'merges_txt'"))
        _bundle_load_tokenizer((vocab_path, merges_path); format=manifest.load_format, kwargs...)
    else
        primary_key = _manifest_primary_file_key(manifest.load_format)
        path = get(files, primary_key, nothing)
        path === nothing && throw(ArgumentError(
            "Manifest files missing required key '$primary_key' for load_format=$(manifest.load_format)",
        ))
        _bundle_load_tokenizer(path; format=manifest.load_format, kwargs...)
    end

    _apply_manifest_special_tokens!(tokenizer, special_tokens)
    return tokenizer
end

function _manifest_from_result(
    result::TrainingResult,
    outdir::AbstractString,
    export_format::Symbol,
)::TrainingManifestV1
    _ = outdir
    trainer = _manifest_trainer_symbol(result.config)
    load_format, files, load_kwargs = _manifest_loader_spec(result.tokenizer, export_format)
    training_config, warnings = _manifest_training_config(result.config)

    metadata = Dict{String,Any}(
        "model_name" => String(result.config.model_name),
        "version" => string(result.config.version),
        "tokenizer_type" => string(typeof(result.tokenizer)),
        "config_type" => string(typeof(result.config)),
    )

    return TrainingManifestV1(
        TRAINING_MANIFEST_SCHEMA_VERSION,
        trainer,
        load_format,
        export_format,
        files,
        load_kwargs,
        training_config,
        metadata,
        warnings,
    )
end

function _resolve_bundle_export_format(
    tokenizer::AbstractSubwordTokenizer,
    export_format::Symbol,
)::Symbol
    if export_format in (:auto, :internal)
        if tokenizer isa WordPieceTokenizer
            return :wordpiece_vocab
        elseif tokenizer isa BPETokenizer || tokenizer isa ByteBPETokenizer
            return :bpe
        elseif tokenizer isa UnigramTokenizer
            return :unigram_tsv
        elseif tokenizer isa SentencePieceTokenizer
            return :sentencepiece_model
        elseif tokenizer isa HuggingFaceJSONTokenizer
            return :hf_tokenizer_json
        end

        throw(ArgumentError(
            "No default training bundle export format for tokenizer type $(typeof(tokenizer))",
        ))
    end

    if export_format in (:bpe, :bpe_gpt2)
        return :bpe
    elseif export_format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece_vocab
    elseif export_format in (:unigram, :unigram_tsv)
        return :unigram_tsv
    elseif export_format in (:sentencepiece, :sentencepiece_model)
        return :sentencepiece_model
    elseif export_format == :hf_tokenizer_json
        return :hf_tokenizer_json
    end

    throw(ArgumentError("Unsupported training bundle export_format=$export_format"))
end

function _manifest_loader_spec(
    tokenizer::AbstractSubwordTokenizer,
    export_format::Symbol,
)::Tuple{Symbol,Dict{String,String},Dict{String,Any}}
    if export_format == :hf_tokenizer_json
        return (
            :hf_tokenizer_json,
            Dict{String,String}("tokenizer_json" => "tokenizer.json"),
            Dict{String,Any}(),
        )
    elseif export_format == :bpe
        files = Dict{String,String}(
            "vocab_txt" => "vocab.txt",
            "merges_txt" => "merges.txt",
        )

        if tokenizer isa ByteBPETokenizer
            bytebpe = tokenizer::ByteBPETokenizer
            return (
                :bytebpe,
                files,
                Dict{String,Any}(
                    "unk_token" => bytebpe.base.unk_token,
                    "end_of_word_marker" => bytebpe.base.end_of_word_marker,
                    "special_tokens" => _manifest_special_tokens_map(tokenizer),
                ),
            )
        elseif tokenizer isa BPETokenizer
            bpe = tokenizer::BPETokenizer
            return (
                :bpe,
                files,
                Dict{String,Any}(
                    "unk_token" => bpe.unk_token,
                    "end_of_word_marker" => bpe.end_of_word_marker,
                    "special_tokens" => _manifest_special_tokens_map(tokenizer),
                ),
            )
        end

        throw(ArgumentError(
            "Export format :bpe requires BPETokenizer or ByteBPETokenizer, got $(typeof(tokenizer))",
        ))
    elseif export_format == :wordpiece_vocab
        tokenizer isa WordPieceTokenizer || throw(ArgumentError(
            "Export format :wordpiece_vocab requires WordPieceTokenizer, got $(typeof(tokenizer))",
        ))
        wordpiece = tokenizer::WordPieceTokenizer
        return (
            :wordpiece,
            Dict{String,String}("vocab_txt" => "vocab.txt"),
            Dict{String,Any}(
                "continuation_prefix" => wordpiece.continuation_prefix,
                "unk_token" => wordpiece.unk_token,
                "max_input_chars_per_word" => wordpiece.max_input_chars_per_word,
                "special_tokens" => _manifest_special_tokens_map(tokenizer),
            ),
        )
    elseif export_format == :unigram_tsv
        tokenizer isa UnigramTokenizer || throw(ArgumentError(
            "Export format :unigram_tsv requires UnigramTokenizer, got $(typeof(tokenizer))",
        ))
        unigram = tokenizer::UnigramTokenizer
        return (
            :unigram,
            Dict{String,String}("unigram_tsv" => "unigram.tsv"),
            Dict{String,Any}(
                "unk_token" => unigram.unk_token,
                "whitespace_marker" => unigram.whitespace_marker,
                "special_tokens" => _manifest_special_tokens_map(tokenizer),
            ),
        )
    elseif export_format == :sentencepiece_model
        return (
            :sentencepiece_model,
            Dict{String,String}("spm_model" => "spm.model"),
            Dict{String,Any}(
                "special_tokens" => _manifest_special_tokens_map(tokenizer),
            ),
        )
    end

    throw(ArgumentError("Unsupported bundle export format: $export_format"))
end

function _manifest_training_config(
    config::AbstractTrainingConfig,
)::Tuple{Dict{String,Any},Vector{String}}
    config_dict = Dict{String,Any}()
    warnings = String[]

    for field in fieldnames(typeof(config))
        value = getfield(config, field)
        key = String(field)

        if field == :pretokenizer
            if value !== nothing
                push!(warnings, "pretokenizer was used for training but is not persisted at runtime")
            end
            continue
        end

        if value isa Function
            push!(warnings, "training config field '$key' is a function and was omitted from manifest")
            continue
        end

        config_dict[key] = _training_value_to_json_safe(value)
    end

    unique!(warnings)
    sort!(warnings)
    return (config_dict, warnings)
end

function _training_value_to_json_safe(value)
    if value === nothing || value isa Bool || value isa Integer || value isa AbstractFloat
        return value
    elseif value isa AbstractString
        return String(value)
    elseif value isa Symbol
        return String(value)
    elseif value isa VersionNumber
        return string(value)
    elseif value isa Tuple || value isa AbstractVector || value isa Set
        return Any[_training_value_to_json_safe(item) for item in value]
    elseif value isa NamedTuple
        return Dict{String,Any}(
            String(key) => _training_value_to_json_safe(getfield(value, key))
            for key in keys(value)
        )
    elseif value isa AbstractDict
        out = Dict{String,Any}()
        for (k, v) in value
            out[String(k)] = _training_value_to_json_safe(v)
        end
        return out
    end

    return string(value)
end

function _manifest_payload(manifest::TrainingManifestV1)
    return (
        schema_version = manifest.schema_version,
        trainer = String(manifest.trainer),
        load_format = String(manifest.load_format),
        export_format = String(manifest.export_format),
        files = _canonical_json(manifest.files),
        load_kwargs = _canonical_json(manifest.load_kwargs),
        training_config = _canonical_json(manifest.training_config),
        metadata = _canonical_json(manifest.metadata),
        warnings = String[manifest.warnings...],
    )
end

function _canonical_json(value)
    if value === nothing || value isa Bool || value isa Integer || value isa AbstractFloat
        return value
    elseif value isa AbstractString
        return String(value)
    elseif value isa Symbol
        return String(value)
    elseif value isa AbstractVector || value isa Tuple || value isa Set
        return Any[_canonical_json(v) for v in value]
    elseif value isa NamedTuple
        items = Pair{String,Any}[]
        for key in keys(value)
            push!(items, String(key) => _canonical_json(getfield(value, key)))
        end
        return _canonical_json_pairs(items)
    elseif value isa AbstractDict
        items = Pair{String,Any}[]
        for (k, v) in value
            push!(items, String(k) => _canonical_json(v))
        end
        return _canonical_json_pairs(items)
    end

    return string(value)
end

function _canonical_json_pairs(
    pairs::Vector{Pair{String,Any}},
)
    sort!(pairs; by=first)
    names = Tuple(Symbol(pair.first) for pair in pairs)
    values = Tuple(pair.second for pair in pairs)
    return NamedTuple{names}(values)
end

function _manifest_is_object(value)::Bool
    return value !== nothing && (value isa AbstractDict || string(typeof(value)) == "JSON3.Object")
end

function _manifest_is_array(value)::Bool
    return value !== nothing && (value isa AbstractVector || occursin("JSON3.Array", string(typeof(value))))
end

function _manifest_haskey(obj, key::String)::Bool
    if obj isa AbstractDict
        return haskey(obj, key) || haskey(obj, Symbol(key))
    end
    return haskey(obj, Symbol(key))
end

function _manifest_get(obj, key::String)
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

function _manifest_get_required(obj, key::String, path::String)
    value = _manifest_get(obj, key)
    value === nothing && throw(ArgumentError("Missing required manifest field at $path"))
    return value
end

function _manifest_as_string(value, path::String)::String
    value isa AbstractString || throw(ArgumentError("Expected string at $path"))
    return String(value)
end

function _manifest_as_int(value, path::String)::Int
    if value isa Integer
        return Int(value)
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("Expected finite integer at $path"))
        floor(Int, value) == value || throw(ArgumentError("Expected integer value at $path"))
        return Int(value)
    end

    throw(ArgumentError("Expected integer at $path"))
end

function _manifest_as_string_dict(value, path::String)::Dict{String,String}
    _manifest_is_object(value) || throw(ArgumentError("Expected object at $path"))
    out = Dict{String,String}()
    for (k, v) in pairs(value)
        out[String(k)] = _manifest_as_string(v, "$path.$(String(k))")
    end
    return out
end

function _manifest_as_any_dict(value, path::String)::Dict{String,Any}
    _manifest_is_object(value) || throw(ArgumentError("Expected object at $path"))
    out = Dict{String,Any}()
    for (k, v) in pairs(value)
        out[String(k)] = _manifest_json_to_any(v, "$path.$(String(k))")
    end
    return out
end

function _manifest_as_string_vector(value, path::String)::Vector{String}
    _manifest_is_array(value) || throw(ArgumentError("Expected array at $path"))
    out = String[]
    for (i, item) in enumerate(value)
        push!(out, _manifest_as_string(item, "$path[$i]"))
    end
    return out
end

function _manifest_json_to_any(value, path::String)
    if value === nothing || value isa Bool || value isa Integer || value isa AbstractFloat
        return value
    elseif value isa AbstractString
        return String(value)
    elseif _manifest_is_array(value)
        out = Any[]
        for (i, item) in enumerate(value)
            push!(out, _manifest_json_to_any(item, "$path[$i]"))
        end
        return out
    elseif _manifest_is_object(value)
        out = Dict{String,Any}()
        for (k, v) in pairs(value)
            out[String(k)] = _manifest_json_to_any(v, "$path.$(String(k))")
        end
        return out
    end

    throw(ArgumentError("Unsupported JSON value at $path"))
end

function _manifest_kwargs_pairs(
    kwargs_dict::Dict{String,Any},
)::Vector{Pair{Symbol,Any}}
    entries = collect(kwargs_dict)
    sort!(entries; by=first)
    return [Symbol(key) => value for (key, value) in entries]
end

function _resolve_manifest_file_paths(
    bundle_dir::String,
    files::Dict{String,String},
)::Dict{String,String}
    bundle_root = abspath(normpath(bundle_dir))
    resolved = Dict{String,String}()

    for (key, raw_relpath) in files
        relpath = String(raw_relpath)
        if isabspath(relpath)
            throw(ArgumentError(
                "Manifest file entry '$key' must be relative, got absolute path: $relpath",
            ))
        end

        absolute = abspath(normpath(joinpath(bundle_root, relpath)))
        _path_within_dir(absolute, bundle_root) || throw(ArgumentError(
            "Manifest file entry '$key' escapes bundle directory: $relpath",
        ))

        isfile(absolute) || throw(ArgumentError(
            "Manifest file entry '$key' points to missing file: $absolute",
        ))
        resolved[key] = absolute
    end

    return resolved
end

function _path_within_dir(path::String, root::String)::Bool
    relative = relpath(path, root)
    isabspath(relative) && return false
    return !(
        relative == ".." ||
        startswith(relative, "../") ||
        startswith(relative, "..\\")
    )
end

function _manifest_primary_file_key(load_format::Symbol)::String
    if load_format == :wordpiece
        return "vocab_txt"
    elseif load_format == :unigram
        return "unigram_tsv"
    elseif load_format in (:sentencepiece, :sentencepiece_model)
        return "spm_model"
    elseif load_format == :hf_tokenizer_json
        return "tokenizer_json"
    end

    throw(ArgumentError("Unsupported manifest load_format=$load_format"))
end

function _manifest_special_tokens_spec(value)::Dict{Symbol,String}
    value === nothing && return Dict{Symbol,String}()
    value isa AbstractDict || throw(ArgumentError(
        "Manifest load_kwargs.special_tokens must be an object when present",
    ))

    out = Dict{Symbol,String}()
    for (raw_symbol, raw_token) in value
        out[Symbol(String(raw_symbol))] = String(raw_token)
    end
    return out
end

function _manifest_special_tokens_map(
    tokenizer::AbstractSubwordTokenizer,
)::Dict{String,Any}
    mapping = Dict{String,Any}()
    pairs = collect(special_tokens(tokenizer))
    sort!(pairs; by=first)

    for (symbol, id) in pairs
        1 <= id <= vocab_size(tokenizer) || continue
        mapping[String(symbol)] = id_to_token(tokenizer, id)
    end

    return mapping
end

function _apply_manifest_special_tokens!(
    tokenizer::AbstractSubwordTokenizer,
    special_tokens_map::Dict{Symbol,String},
)::Nothing
    isempty(special_tokens_map) && return nothing
    target_ids = _mutable_special_token_ids(tokenizer)
    target_ids === nothing && return nothing

    for (symbol, token) in special_tokens_map
        token_id = token_to_id(tokenizer, token)
        token_id < 1 && continue
        token_id > vocab_size(tokenizer) && continue
        id_to_token(tokenizer, token_id) == token || continue
        target_ids[symbol] = token_id
    end

    return nothing
end

_mutable_special_token_ids(::AbstractSubwordTokenizer) = nothing
_mutable_special_token_ids(tokenizer::BPETokenizer) = tokenizer.vocab.special_token_ids
_mutable_special_token_ids(tokenizer::ByteBPETokenizer) = tokenizer.base.vocab.special_token_ids
_mutable_special_token_ids(tokenizer::WordPieceTokenizer) = tokenizer.vocab.special_token_ids
_mutable_special_token_ids(tokenizer::UnigramTokenizer) = tokenizer.vocab.special_token_ids
_mutable_special_token_ids(tokenizer::SentencePieceTokenizer) = _mutable_special_token_ids(tokenizer.inner)
_mutable_special_token_ids(tokenizer::HuggingFaceJSONTokenizer) = tokenizer.special_token_ids

function _bundle_export_tokenizer(
    tokenizer::AbstractSubwordTokenizer,
    outdir::AbstractString;
    format::Symbol,
)::Nothing
    parent = parentmodule(@__MODULE__)
    export_fn = getfield(parent, :export_tokenizer)
    export_fn(tokenizer, outdir; format=format)
    return nothing
end

function _bundle_load_tokenizer(args...; kwargs...)::AbstractSubwordTokenizer
    parent = parentmodule(@__MODULE__)
    load_fn = getfield(parent, :load_tokenizer)
    return load_fn(args...; kwargs...)
end

_manifest_trainer_symbol(::BPETrainingConfig) = :bpe
_manifest_trainer_symbol(::ByteBPETrainingConfig) = :bytebpe
_manifest_trainer_symbol(::UnigramTrainingConfig) = :unigram
_manifest_trainer_symbol(::WordPieceTrainingConfig) = :wordpiece
_manifest_trainer_symbol(::SentencePieceTrainingConfig) = :sentencepiece
_manifest_trainer_symbol(::BertWordPieceTrainingConfig) = :hf_bert_wordpiece
_manifest_trainer_symbol(::RobertaByteBPETrainingConfig) = :hf_roberta_bytebpe
_manifest_trainer_symbol(::GPT2ByteBPETrainingConfig) = :hf_gpt2_bytebpe
