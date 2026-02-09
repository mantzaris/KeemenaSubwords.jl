import Pkg.Artifacts

const _PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ARTIFACTS_TOML = joinpath(_PACKAGE_ROOT, "Artifacts.toml")

struct BuiltinModelEntry
    format::Symbol
    artifact_name::Union{Nothing,String}
    artifact_subpath::Union{Nothing,String}
    fallback_path::String
    description::String
end

_entry(
    format::Symbol,
    artifact_name::Union{Nothing,String},
    artifact_subpath::Union{Nothing,String},
    fallback_path::String,
    description::String,
) = BuiltinModelEntry(format, artifact_name, artifact_subpath, fallback_path, description)

const _MODEL_REGISTRY = Dict{Symbol,BuiltinModelEntry}(
    :core_bpe_en => _entry(
        :bpe,
        "core_bpe_en",
        "core_bpe_en",
        joinpath(_PACKAGE_ROOT, "models", "core_bpe_en"),
        "Tiny built-in English classic BPE model (vocab.txt + merges.txt).",
    ),
    :core_wordpiece_en => _entry(
        :wordpiece,
        "core_wordpiece_en",
        "core_wordpiece_en/vocab.txt",
        joinpath(_PACKAGE_ROOT, "models", "core_wordpiece_en", "vocab.txt"),
        "Tiny built-in English WordPiece model.",
    ),
    :core_sentencepiece_unigram_en => _entry(
        :sentencepiece,
        "core_sentencepiece_unigram_en",
        "core_sentencepiece_unigram_en/spm.model",
        joinpath(_PACKAGE_ROOT, "models", "core_sentencepiece_unigram_en", "spm.model"),
        "Tiny built-in SentencePiece Unigram model (.model).",
    ),
    :tiktoken_o200k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/o200k_base/o200k_base.tiktoken",
        joinpath(_PACKAGE_ROOT, "models", "tiktoken", "o200k_base", "o200k_base.tiktoken"),
        "OpenAI tiktoken o200k_base encoding.",
    ),
    :tiktoken_cl100k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/cl100k_base/cl100k_base.tiktoken",
        joinpath(_PACKAGE_ROOT, "models", "tiktoken", "cl100k_base", "cl100k_base.tiktoken"),
        "OpenAI tiktoken cl100k_base encoding.",
    ),
    :tiktoken_r50k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/r50k_base/r50k_base.tiktoken",
        joinpath(_PACKAGE_ROOT, "models", "tiktoken", "r50k_base", "r50k_base.tiktoken"),
        "OpenAI tiktoken r50k_base encoding.",
    ),
    :tiktoken_p50k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/p50k_base/p50k_base.tiktoken",
        joinpath(_PACKAGE_ROOT, "models", "tiktoken", "p50k_base", "p50k_base.tiktoken"),
        "OpenAI tiktoken p50k_base encoding.",
    ),
    :openai_gpt2_bpe => _entry(
        :bpe_gpt2,
        "keemena_public_tokenizer_assets_v1",
        "bpe/openai_gpt2",
        joinpath(_PACKAGE_ROOT, "models", "bpe", "openai_gpt2"),
        "OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe).",
    ),
    :bert_base_uncased_wordpiece => _entry(
        :wordpiece_vocab,
        "keemena_public_tokenizer_assets_v1",
        "wordpiece/bert_base_uncased/vocab.txt",
        joinpath(_PACKAGE_ROOT, "models", "wordpiece", "bert_base_uncased", "vocab.txt"),
        "Hugging Face bert-base-uncased WordPiece vocabulary.",
    ),
    :t5_small_sentencepiece_unigram => _entry(
        :sentencepiece_model,
        "keemena_public_tokenizer_assets_v1",
        "sentencepiece/t5_small/spiece.model",
        joinpath(_PACKAGE_ROOT, "models", "sentencepiece", "t5_small", "spiece.model"),
        "Hugging Face google-t5/t5-small SentencePiece model (Unigram).",
    ),
)

const _MODEL_UPSTREAM_FILES = Dict{
    Symbol,
    Vector{NamedTuple{(:relative_path, :url, :sha256),Tuple{String,String,Union{Nothing,String}}}},
}(
    :tiktoken_o200k_base => [
        (
            relative_path = "tiktoken/o200k_base/o200k_base.tiktoken",
            url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
            sha256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        ),
    ],
    :tiktoken_cl100k_base => [
        (
            relative_path = "tiktoken/cl100k_base/cl100k_base.tiktoken",
            url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
            sha256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        ),
    ],
    :tiktoken_r50k_base => [
        (
            relative_path = "tiktoken/r50k_base/r50k_base.tiktoken",
            url = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
            sha256 = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        ),
    ],
    :tiktoken_p50k_base => [
        (
            relative_path = "tiktoken/p50k_base/p50k_base.tiktoken",
            url = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
            sha256 = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        ),
    ],
    :openai_gpt2_bpe => [
        (
            relative_path = "bpe/openai_gpt2/encoder.json",
            url = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
            sha256 = "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        ),
        (
            relative_path = "bpe/openai_gpt2/vocab.bpe",
            url = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
            sha256 = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
    ],
    :bert_base_uncased_wordpiece => [
        (
            relative_path = "wordpiece/bert_base_uncased/vocab.txt",
            url = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt",
            sha256 = nothing,
        ),
    ],
    :t5_small_sentencepiece_unigram => [
        (
            relative_path = "sentencepiece/t5_small/spiece.model",
            url = "https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model",
            sha256 = nothing,
        ),
    ],
)

"""
List available built-in model names.
"""
function available_models()::Vector{Symbol}
    names = collect(keys(_MODEL_REGISTRY))
    sort!(names)
    return names
end

"""
Ensure artifact-backed built-in models are present on disk.

Returns a dictionary of `key => is_available`.
"""
function prefetch_models(
    keys::AbstractVector{Symbol}=available_models();
    force::Bool=false,
)::Dict{Symbol,Bool}
    availability = Dict{Symbol,Bool}()

    for key in keys
        haskey(_MODEL_REGISTRY, key) || throw(ArgumentError("Unknown built-in model: $key"))
        entry = _MODEL_REGISTRY[key]

        if entry.artifact_name !== nothing
            _ensure_artifact_installed(entry.artifact_name; force=force)
        end

        availability[key] = ispath(_resolve_model_path(entry))
    end

    return availability
end

"""
Describe a built-in model.
"""
function describe_model(name::Symbol)::NamedTuple
    haskey(_MODEL_REGISTRY, name) || throw(ArgumentError("Unknown built-in model: $name"))
    entry = _MODEL_REGISTRY[name]
    resolved = _resolve_model_path(entry)
    source = _resolve_model_source(entry, resolved)
    upstream = get(_MODEL_UPSTREAM_FILES, name, NamedTuple[])
    files = _resolved_model_files(entry, resolved)
    return (
        name = name,
        format = entry.format,
        path = resolved,
        files = files,
        exists = ispath(resolved),
        source = source,
        description = entry.description,
        artifact_name = entry.artifact_name,
        upstream_files = upstream,
        provenance_urls = [f.url for f in upstream],
    )
end

"""
Resolve built-in model name to on-disk path.
"""
function model_path(name::Symbol; auto_prefetch::Bool=true)::String
    info = describe_model(name)

    if !info.exists && auto_prefetch
        prefetch_models([name])
        info = describe_model(name)
    end

    info.exists || throw(ArgumentError("Model asset missing on disk for $name at $(info.path)"))
    return info.path
end

function _resolve_model_source(entry::BuiltinModelEntry, resolved::String)::Symbol
    if entry.artifact_name !== nothing
        artifact_root = _artifact_root(entry.artifact_name)
        if artifact_root !== nothing
            expected = entry.artifact_subpath === nothing ? artifact_root : joinpath(artifact_root, entry.artifact_subpath)
            if resolved == expected
                return :artifact
            end
        end
    end

    if resolved == entry.fallback_path
        return :in_repo
    end

    return :unresolved
end

function _resolve_model_path(entry::BuiltinModelEntry)::String
    artifact_root = _artifact_root(entry.artifact_name)
    if artifact_root !== nothing
        if entry.artifact_subpath === nothing
            return artifact_root
        end
        candidate = joinpath(artifact_root, entry.artifact_subpath)
        if ispath(candidate)
            return candidate
        end
    end
    return entry.fallback_path
end

function _resolved_model_files(entry::BuiltinModelEntry, resolved::String)::Vector{String}
    if entry.format == :bpe || entry.format == :bytebpe
        if isdir(resolved)
            return String[
                joinpath(resolved, "vocab.txt"),
                joinpath(resolved, "merges.txt"),
            ]
        end
        return String[resolved]
    elseif entry.format == :bpe_gpt2
        if isdir(resolved)
            encoder = joinpath(resolved, "encoder.json")
            vocab_bpe = joinpath(resolved, "vocab.bpe")
            if isfile(encoder) && isfile(vocab_bpe)
                return String[encoder, vocab_bpe]
            end

            vocab_json = joinpath(resolved, "vocab.json")
            merges_txt = joinpath(resolved, "merges.txt")
            return String[vocab_json, merges_txt]
        end
        return String[resolved]
    elseif entry.format in (:wordpiece, :wordpiece_vocab)
        if isdir(resolved)
            return String[joinpath(resolved, "vocab.txt")]
        end
        return String[resolved]
    elseif entry.format in (:sentencepiece, :sentencepiece_model)
        if isdir(resolved)
            return String[joinpath(resolved, "spm.model")]
        end
        return String[resolved]
    elseif entry.format == :tiktoken
        if isdir(resolved)
            matches = filter(p -> endswith(lowercase(p), ".tiktoken"), readdir(resolved; join=true))
            return String[matches...]
        end
        return String[resolved]
    end

    return String[resolved]
end

function _ensure_artifact_installed(artifact_name::String; force::Bool=false)::Bool
    isfile(_ARTIFACTS_TOML) || return false

    hash = Artifacts.artifact_hash(artifact_name, _ARTIFACTS_TOML)
    hash === nothing && return false

    if !Artifacts.artifact_exists(hash) || force
        try
            Base.invokelatest(Artifacts.ensure_artifact_installed, artifact_name, _ARTIFACTS_TOML)
        catch
            # Keep failure non-fatal; callers can still use in-repo fallback assets.
        end
    end

    return Artifacts.artifact_exists(hash)
end

function _artifact_root(artifact_name::Union{Nothing,String})::Union{Nothing,String}
    artifact_name === nothing && return nothing
    isfile(_ARTIFACTS_TOML) || return nothing

    hash = Artifacts.artifact_hash(artifact_name, _ARTIFACTS_TOML)
    hash === nothing && return nothing
    Artifacts.artifact_exists(hash) || return nothing
    return Artifacts.artifact_path(hash)
end
