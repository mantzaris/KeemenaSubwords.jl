import Pkg.Artifacts

const _PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ARTIFACTS_TOML = joinpath(_PACKAGE_ROOT, "Artifacts.toml")
const _ARTIFACT_ONLY_ROOT = joinpath(_PACKAGE_ROOT, "models", "_artifacts_only")

struct BuiltinModelEntry
    format::Symbol
    artifact_name::Union{Nothing,String}
    artifact_subpath::Union{Nothing,String}
    fallback_path::String
    description::String
    license::String
    upstream_ref::String
end

_entry(
    format::Symbol,
    artifact_name::Union{Nothing,String},
    artifact_subpath::Union{Nothing,String},
    fallback_path::String,
    description::String,
    license::String,
    upstream_ref::String,
) = BuiltinModelEntry(
    format,
    artifact_name,
    artifact_subpath,
    fallback_path,
    description,
    license,
    upstream_ref,
)

const _MODEL_REGISTRY = Dict{Symbol,BuiltinModelEntry}(
    :core_bpe_en => _entry(
        :bpe,
        "core_bpe_en",
        "core_bpe_en",
        joinpath(_PACKAGE_ROOT, "models", "core_bpe_en"),
        "Tiny built-in English classic BPE model (vocab.txt + merges.txt).",
        "MIT",
        "in-repo:core",
    ),
    :core_wordpiece_en => _entry(
        :wordpiece_vocab,
        "core_wordpiece_en",
        "core_wordpiece_en/vocab.txt",
        joinpath(_PACKAGE_ROOT, "models", "core_wordpiece_en", "vocab.txt"),
        "Tiny built-in English WordPiece model.",
        "MIT",
        "in-repo:core",
    ),
    :core_sentencepiece_unigram_en => _entry(
        :sentencepiece_model,
        "core_sentencepiece_unigram_en",
        "core_sentencepiece_unigram_en/spm.model",
        joinpath(_PACKAGE_ROOT, "models", "core_sentencepiece_unigram_en", "spm.model"),
        "Tiny built-in SentencePiece Unigram model (.model).",
        "MIT",
        "in-repo:core",
    ),
    :tiktoken_o200k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/o200k_base/o200k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "o200k_base", "o200k_base.tiktoken"),
        "OpenAI tiktoken o200k_base encoding.",
        "MIT",
        "openaipublic:encodings/o200k_base.tiktoken",
    ),
    :tiktoken_cl100k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/cl100k_base/cl100k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "cl100k_base", "cl100k_base.tiktoken"),
        "OpenAI tiktoken cl100k_base encoding.",
        "MIT",
        "openaipublic:encodings/cl100k_base.tiktoken",
    ),
    :tiktoken_r50k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/r50k_base/r50k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "r50k_base", "r50k_base.tiktoken"),
        "OpenAI tiktoken r50k_base encoding.",
        "MIT",
        "openaipublic:encodings/r50k_base.tiktoken",
    ),
    :tiktoken_p50k_base => _entry(
        :tiktoken,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/p50k_base/p50k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "p50k_base", "p50k_base.tiktoken"),
        "OpenAI tiktoken p50k_base encoding.",
        "MIT",
        "openaipublic:encodings/p50k_base.tiktoken",
    ),
    :openai_gpt2_bpe => _entry(
        :bpe_gpt2,
        "keemena_public_tokenizer_assets_v1",
        "bpe/openai_gpt2",
        joinpath(_ARTIFACT_ONLY_ROOT, "bpe", "openai_gpt2"),
        "OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe).",
        "MIT",
        "openaipublic:gpt-2/encodings/main",
    ),
    :bert_base_uncased_wordpiece => _entry(
        :wordpiece_vocab,
        "keemena_public_tokenizer_assets_v1",
        "wordpiece/bert_base_uncased/vocab.txt",
        joinpath(_ARTIFACT_ONLY_ROOT, "wordpiece", "bert_base_uncased", "vocab.txt"),
        "Hugging Face bert-base-uncased WordPiece vocabulary.",
        "Apache-2.0",
        "huggingface:bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594",
    ),
    :t5_small_sentencepiece_unigram => _entry(
        :sentencepiece_model,
        "keemena_public_tokenizer_assets_v1",
        "sentencepiece/t5_small/spiece.model",
        joinpath(_ARTIFACT_ONLY_ROOT, "sentencepiece", "t5_small", "spiece.model"),
        "Hugging Face google-t5/t5-small SentencePiece model (Unigram).",
        "Apache-2.0",
        "huggingface:google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    ),
    :mistral_v1_sentencepiece => _entry(
        :sentencepiece_model,
        "mistral_v1_sentencepiece",
        "mistral_v1_sentencepiece",
        joinpath(_ARTIFACT_ONLY_ROOT, "mistral_v1_sentencepiece"),
        "Mistral/Mixtral tokenizer.model SentencePiece model.",
        "Apache-2.0",
        "huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
    ),
    :mistral_v3_sentencepiece => _entry(
        :sentencepiece_model,
        "mistral_v3_sentencepiece",
        "mistral_v3_sentencepiece",
        joinpath(_ARTIFACT_ONLY_ROOT, "mistral_v3_sentencepiece"),
        "Mistral-7B-Instruct-v0.3 tokenizer.model.v3 SentencePiece model.",
        "Apache-2.0",
        "huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    :phi2_bpe => _entry(
        :bpe_gpt2,
        "phi2_bpe",
        "phi2_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "phi2_bpe"),
        "Microsoft Phi-2 GPT2-style tokenizer files (vocab.json + merges.txt).",
        "MIT",
        "huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0",
    ),
    :roberta_base_bpe => _entry(
        :bpe_gpt2,
        "roberta_base_bpe",
        "roberta_base_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "roberta_base_bpe"),
        "RoBERTa-base byte-level BPE tokenizer files (vocab.json + merges.txt).",
        "MIT",
        "huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    ),
    :xlm_roberta_base_sentencepiece_bpe => _entry(
        :sentencepiece_model,
        "xlm_roberta_base_sentencepiece_bpe",
        "xlm_roberta_base_sentencepiece_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "xlm_roberta_base_sentencepiece_bpe"),
        "XLM-RoBERTa-base sentencepiece.bpe.model file.",
        "MIT",
        "huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
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
            url = "https://huggingface.co/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt",
            sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        ),
    ],
    :t5_small_sentencepiece_unigram => [
        (
            relative_path = "sentencepiece/t5_small/spiece.model",
            url = "https://huggingface.co/google-t5/t5-small/resolve/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/spiece.model",
            sha256 = "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
        ),
    ],
    :mistral_v1_sentencepiece => [
        (
            relative_path = "mistral_v1_sentencepiece/tokenizer.model",
            url = "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/eba92302a2861cdc0098cc54bc9f17cb2c47eb61/tokenizer.model",
            sha256 = "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
        ),
    ],
    :mistral_v3_sentencepiece => [
        (
            relative_path = "mistral_v3_sentencepiece/tokenizer.model.v3",
            url = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/c170c708c41dac9275d15a8fff4eca08d52bab71/tokenizer.model.v3",
            sha256 = "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
        ),
    ],
    :phi2_bpe => [
        (
            relative_path = "phi2_bpe/vocab.json",
            url = "https://huggingface.co/microsoft/phi-2/resolve/810d367871c1d460086d9f82db8696f2e0a0fcd0/vocab.json",
            sha256 = "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
        ),
        (
            relative_path = "phi2_bpe/merges.txt",
            url = "https://huggingface.co/microsoft/phi-2/resolve/810d367871c1d460086d9f82db8696f2e0a0fcd0/merges.txt",
            sha256 = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
    ],
    :roberta_base_bpe => [
        (
            relative_path = "roberta_base_bpe/vocab.json",
            url = "https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/vocab.json",
            sha256 = "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
        ),
        (
            relative_path = "roberta_base_bpe/merges.txt",
            url = "https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/merges.txt",
            sha256 = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        ),
    ],
    :xlm_roberta_base_sentencepiece_bpe => [
        (
            relative_path = "xlm_roberta_base_sentencepiece_bpe/sentencepiece.bpe.model",
            url = "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model",
            sha256 = "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
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

        resolved = _resolve_model_path(entry)
        files = _resolved_model_files(entry, resolved)
        availability[key] = !isempty(files) && all(ispath, files)
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
    exists = !isempty(files) && all(ispath, files)

    return (
        name = name,
        format = entry.format,
        path = resolved,
        files = files,
        exists = exists,
        source = source,
        description = entry.description,
        license = entry.license,
        upstream_ref = entry.upstream_ref,
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
            if normpath(resolved) == normpath(expected)
                return :artifact
            end
        end
    end

    if ispath(resolved) && startswith(normpath(resolved), normpath(joinpath(_PACKAGE_ROOT, "models")))
        return :in_repo
    end

    return :missing
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

function _sentencepiece_candidates(dir::String)::Vector{String}
    names = (
        "spm.model",
        "tokenizer.model",
        "tokenizer.model.v3",
        "sentencepiece.bpe.model",
    )
    found = String[]
    for name in names
        candidate = joinpath(dir, name)
        isfile(candidate) && push!(found, candidate)
    end
    return found
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
            vocab_json = joinpath(resolved, "vocab.json")
            merges_txt = joinpath(resolved, "merges.txt")
            if isfile(vocab_json) && isfile(merges_txt)
                return String[vocab_json, merges_txt]
            end

            encoder = joinpath(resolved, "encoder.json")
            vocab_bpe = joinpath(resolved, "vocab.bpe")
            return String[encoder, vocab_bpe]
        end
        return String[resolved]
    elseif entry.format in (:wordpiece, :wordpiece_vocab)
        if isdir(resolved)
            return String[joinpath(resolved, "vocab.txt")]
        end
        return String[resolved]
    elseif entry.format in (:sentencepiece, :sentencepiece_model)
        if isdir(resolved)
            return _sentencepiece_candidates(resolved)
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
            # Keep failure non-fatal; callers can still rely on existing artifacts.
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
