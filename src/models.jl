import Pkg.Artifacts

const _PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ARTIFACTS_TOML = joinpath(_PACKAGE_ROOT, "Artifacts.toml")

const _MODEL_REGISTRY = Dict{
    Symbol,
    NamedTuple{
        (:format, :artifact_name, :artifact_subpath, :fallback_path, :description),
        Tuple{Symbol,Union{Nothing,String},Union{Nothing,String},String,String},
    },
}(
    :core_bpe_en => (
        format = :bpe,
        artifact_name = "core_bpe_en",
        artifact_subpath = "core_bpe_en",
        fallback_path = joinpath(_PACKAGE_ROOT, "models", "core_bpe_en"),
        description = "Tiny built-in English classic BPE model (vocab.txt + merges.txt).",
    ),
    :core_wordpiece_en => (
        format = :wordpiece,
        artifact_name = "core_wordpiece_en",
        artifact_subpath = "core_wordpiece_en/vocab.txt",
        fallback_path = joinpath(_PACKAGE_ROOT, "models", "core_wordpiece_en", "vocab.txt"),
        description = "Tiny built-in English WordPiece model.",
    ),
    :core_sentencepiece_unigram_en => (
        format = :sentencepiece,
        artifact_name = "core_sentencepiece_unigram_en",
        artifact_subpath = "core_sentencepiece_unigram_en/spm.model",
        fallback_path = joinpath(_PACKAGE_ROOT, "models", "core_sentencepiece_unigram_en", "spm.model"),
        description = "Tiny built-in SentencePiece Unigram model (.model).",
    ),
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
Describe a built-in model.
"""
function describe_model(name::Symbol)::NamedTuple
    haskey(_MODEL_REGISTRY, name) || throw(ArgumentError("Unknown built-in model: $name"))
    entry = _MODEL_REGISTRY[name]
    resolved = _resolve_model_path(entry)
    source = _resolve_model_source(entry, resolved)
    return (
        name = name,
        format = entry.format,
        path = resolved,
        exists = ispath(resolved),
        source = source,
        description = entry.description,
    )
end

"""
Resolve built-in model name to on-disk path.
"""
function model_path(name::Symbol)::String
    info = describe_model(name)
    info.exists || throw(ArgumentError("Model asset missing on disk for $name at $(info.path)"))
    return info.path
end

function _resolve_model_source(entry, resolved::String)::Symbol
    if resolved == entry.fallback_path
        return :in_repo
    end
    return :artifact
end

function _resolve_model_path(entry)::String
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

function _artifact_root(artifact_name::Union{Nothing,String})::Union{Nothing,String}
    artifact_name === nothing && return nothing
    isfile(_ARTIFACTS_TOML) || return nothing

    hash = Artifacts.artifact_hash(artifact_name, _ARTIFACTS_TOML)
    hash === nothing && return nothing
    Artifacts.artifact_exists(hash) || return nothing
    return Artifacts.artifact_path(hash)
end
