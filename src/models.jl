import Pkg.Artifacts
using Downloads
using TOML

const _PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ARTIFACTS_TOML = joinpath(_PACKAGE_ROOT, "Artifacts.toml")
const _ARTIFACT_ONLY_ROOT = joinpath(_PACKAGE_ROOT, "models", "_artifacts_only")
const _DEFAULT_CACHE_ROOT = isempty(DEPOT_PATH) ?
    joinpath(homedir(), ".julia", "keemena_subwords") :
    joinpath(first(DEPOT_PATH), "keemena_subwords")
const _CACHE_ROOT = normpath(get(ENV, "KEEMENA_SUBWORDS_CACHE_DIR", _DEFAULT_CACHE_ROOT))
const _LOCAL_MODELS_TOML = joinpath(_CACHE_ROOT, "local_models.toml")
const _LOCAL_MODELS_LOADED = Ref(false)
const _PERSISTED_LOCAL_KEYS = Set{Symbol}()
const _LOCAL_MODEL_RESOLVED_FILES = Dict{Symbol,Vector{String}}()
const _LOCAL_MODEL_NOTES = Dict{Symbol,String}()
const _UPSTREAM_FILE_INFO = NamedTuple{(:relative_path, :url, :sha256),Tuple{String,String,Union{Nothing,String}}}
const _VALID_MODEL_DISTRIBUTIONS = Set{Symbol}((
    :shipped,
    :artifact_public,
    :installable_gated,
    :user_local,
))

struct BuiltinModelEntry
    format::Symbol
    family::Symbol
    distribution::Symbol
    artifact_name::Union{Nothing,String}
    artifact_subpath::Union{Nothing,String}
    fallback_path::String
    description::String
    license::String
    upstream_repo::String
    upstream_ref::String
end

function _validate_distribution(distribution::Symbol)::Symbol
    distribution in _VALID_MODEL_DISTRIBUTIONS || throw(ArgumentError("Unsupported model distribution: $distribution"))
    return distribution
end

_entry(
    format::Symbol,
    family::Symbol,
    distribution::Symbol,
    artifact_name::Union{Nothing,String},
    artifact_subpath::Union{Nothing,String},
    fallback_path::String,
    description::String,
    license::String,
    upstream_repo::String,
    upstream_ref::String,
) = BuiltinModelEntry(
    format,
    family,
    _validate_distribution(distribution),
    artifact_name,
    artifact_subpath,
    fallback_path,
    description,
    license,
    upstream_repo,
    upstream_ref,
)

const _MODEL_REGISTRY = Dict{Symbol,BuiltinModelEntry}(
    :core_bpe_en => _entry(
        :bpe,
        :core,
        :shipped,
        "core_bpe_en",
        "core_bpe_en",
        joinpath(_PACKAGE_ROOT, "models", "core_bpe_en"),
        "Tiny built-in English classic BPE model (vocab.txt + merges.txt).",
        "MIT",
        "in-repo/core",
        "in-repo:core",
    ),
    :core_wordpiece_en => _entry(
        :wordpiece_vocab,
        :core,
        :shipped,
        "core_wordpiece_en",
        "core_wordpiece_en/vocab.txt",
        joinpath(_PACKAGE_ROOT, "models", "core_wordpiece_en", "vocab.txt"),
        "Tiny built-in English WordPiece model.",
        "MIT",
        "in-repo/core",
        "in-repo:core",
    ),
    :core_sentencepiece_unigram_en => _entry(
        :sentencepiece_model,
        :core,
        :shipped,
        "core_sentencepiece_unigram_en",
        "core_sentencepiece_unigram_en/spm.model",
        joinpath(_PACKAGE_ROOT, "models", "core_sentencepiece_unigram_en", "spm.model"),
        "Tiny built-in SentencePiece Unigram model (.model).",
        "MIT",
        "in-repo/core",
        "in-repo:core",
    ),
    :tiktoken_o200k_base => _entry(
        :tiktoken,
        :openai,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/o200k_base/o200k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "o200k_base", "o200k_base.tiktoken"),
        "OpenAI tiktoken o200k_base encoding.",
        "MIT",
        "openaipublic/encodings",
        "openaipublic:encodings/o200k_base.tiktoken",
    ),
    :tiktoken_cl100k_base => _entry(
        :tiktoken,
        :openai,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/cl100k_base/cl100k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "cl100k_base", "cl100k_base.tiktoken"),
        "OpenAI tiktoken cl100k_base encoding.",
        "MIT",
        "openaipublic/encodings",
        "openaipublic:encodings/cl100k_base.tiktoken",
    ),
    :tiktoken_r50k_base => _entry(
        :tiktoken,
        :openai,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/r50k_base/r50k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "r50k_base", "r50k_base.tiktoken"),
        "OpenAI tiktoken r50k_base encoding.",
        "MIT",
        "openaipublic/encodings",
        "openaipublic:encodings/r50k_base.tiktoken",
    ),
    :tiktoken_p50k_base => _entry(
        :tiktoken,
        :openai,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "tiktoken/p50k_base/p50k_base.tiktoken",
        joinpath(_ARTIFACT_ONLY_ROOT, "tiktoken", "p50k_base", "p50k_base.tiktoken"),
        "OpenAI tiktoken p50k_base encoding.",
        "MIT",
        "openaipublic/encodings",
        "openaipublic:encodings/p50k_base.tiktoken",
    ),
    :openai_gpt2_bpe => _entry(
        :bpe_gpt2,
        :openai,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "bpe/openai_gpt2",
        joinpath(_ARTIFACT_ONLY_ROOT, "bpe", "openai_gpt2"),
        "OpenAI GPT-2 byte-level BPE assets (encoder.json + vocab.bpe).",
        "MIT",
        "openaipublic/gpt-2",
        "openaipublic:gpt-2/encodings/main",
    ),
    :bert_base_uncased_wordpiece => _entry(
        :wordpiece_vocab,
        :bert,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "wordpiece/bert_base_uncased/vocab.txt",
        joinpath(_ARTIFACT_ONLY_ROOT, "wordpiece", "bert_base_uncased", "vocab.txt"),
        "Hugging Face bert-base-uncased WordPiece vocabulary.",
        "Apache-2.0",
        "bert-base-uncased",
        "huggingface:bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594",
    ),
    :t5_small_sentencepiece_unigram => _entry(
        :sentencepiece_model,
        :t5,
        :artifact_public,
        "keemena_public_tokenizer_assets_v1",
        "sentencepiece/t5_small/spiece.model",
        joinpath(_ARTIFACT_ONLY_ROOT, "sentencepiece", "t5_small", "spiece.model"),
        "Hugging Face google-t5/t5-small SentencePiece model (Unigram).",
        "Apache-2.0",
        "google-t5/t5-small",
        "huggingface:google-t5/t5-small@df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    ),
    :mistral_v1_sentencepiece => _entry(
        :sentencepiece_model,
        :mistral,
        :artifact_public,
        "mistral_v1_sentencepiece",
        "mistral_v1_sentencepiece",
        joinpath(_ARTIFACT_ONLY_ROOT, "mistral_v1_sentencepiece"),
        "Mistral/Mixtral tokenizer.model SentencePiece model.",
        "Apache-2.0",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
    ),
    :mistral_v3_sentencepiece => _entry(
        :sentencepiece_model,
        :mistral,
        :artifact_public,
        "mistral_v3_sentencepiece",
        "mistral_v3_sentencepiece",
        joinpath(_ARTIFACT_ONLY_ROOT, "mistral_v3_sentencepiece"),
        "Mistral-7B-Instruct-v0.3 tokenizer.model.v3 SentencePiece model.",
        "Apache-2.0",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    :phi2_bpe => _entry(
        :bpe_gpt2,
        :phi,
        :artifact_public,
        "phi2_bpe",
        "phi2_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "phi2_bpe"),
        "Microsoft Phi-2 GPT2-style tokenizer files (vocab.json + merges.txt).",
        "MIT",
        "microsoft/phi-2",
        "huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0",
    ),
    :qwen2_5_bpe => _entry(
        :hf_tokenizer_json,
        :qwen,
        :artifact_public,
        "qwen2_5_bpe",
        "qwen2_5_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "qwen2_5_bpe"),
        "Qwen2.5 BPE tokenizer assets (tokenizer.json with vocab/merges fallback).",
        "Apache-2.0",
        "Qwen/Qwen2.5-7B",
        "huggingface:Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796",
    ),
    :bert_base_multilingual_cased_wordpiece => _entry(
        :wordpiece_vocab,
        :bert,
        :artifact_public,
        "bert_base_multilingual_cased_wordpiece",
        "bert_base_multilingual_cased_wordpiece/vocab.txt",
        joinpath(_ARTIFACT_ONLY_ROOT, "bert_base_multilingual_cased_wordpiece", "vocab.txt"),
        "Hugging Face bert-base-multilingual-cased WordPiece vocabulary.",
        "Apache-2.0",
        "google-bert/bert-base-multilingual-cased",
        "huggingface:google-bert/bert-base-multilingual-cased@3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    ),
    :roberta_base_bpe => _entry(
        :bpe_gpt2,
        :roberta,
        :artifact_public,
        "roberta_base_bpe",
        "roberta_base_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "roberta_base_bpe"),
        "RoBERTa-base byte-level BPE tokenizer files (vocab.json + merges.txt).",
        "MIT",
        "FacebookAI/roberta-base",
        "huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    ),
    :xlm_roberta_base_sentencepiece_bpe => _entry(
        :sentencepiece_model,
        :xlm_roberta,
        :artifact_public,
        "xlm_roberta_base_sentencepiece_bpe",
        "xlm_roberta_base_sentencepiece_bpe",
        joinpath(_ARTIFACT_ONLY_ROOT, "xlm_roberta_base_sentencepiece_bpe"),
        "XLM-RoBERTa-base sentencepiece.bpe.model file.",
        "MIT",
        "FacebookAI/xlm-roberta-base",
        "huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    ),
    :llama2_tokenizer => _entry(
        :sentencepiece_model,
        :llama,
        :installable_gated,
        nothing,
        nothing,
        joinpath(_CACHE_ROOT, "install", "llama2_tokenizer"),
        "Meta Llama 2 tokenizer (gated; install with install_model!).",
        "Llama-2-Community-License",
        "meta-llama/Llama-2-7b-hf",
        "huggingface:meta-llama/Llama-2-7b-hf@main",
    ),
    :llama3_8b_tokenizer => _entry(
        :hf_tokenizer_json,
        :llama,
        :installable_gated,
        nothing,
        nothing,
        joinpath(_CACHE_ROOT, "install", "llama3_8b_tokenizer"),
        "Meta Llama 3 8B tokenizer (gated; install with install_model!).",
        "Llama-3.1-Community-License",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "huggingface:meta-llama/Meta-Llama-3-8B-Instruct@main",
    ),
)

const _MODEL_UPSTREAM_FILES = Dict{
    Symbol,
    Vector{_UPSTREAM_FILE_INFO},
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
    :qwen2_5_bpe => [
        (
            relative_path = "qwen2_5_bpe/vocab.json",
            url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796/vocab.json",
            sha256 = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
        (
            relative_path = "qwen2_5_bpe/merges.txt",
            url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796/merges.txt",
            sha256 = "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        (
            relative_path = "qwen2_5_bpe/tokenizer.json",
            url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796/tokenizer.json",
            sha256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        (
            relative_path = "qwen2_5_bpe/tokenizer_config.json",
            url = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796/tokenizer_config.json",
            sha256 = "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
        ),
    ],
    :bert_base_multilingual_cased_wordpiece => [
        (
            relative_path = "bert_base_multilingual_cased_wordpiece/vocab.txt",
            url = "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/3f076fdb1ab68d5b2880cb87a0886f315b8146f8/vocab.txt",
            sha256 = "fe0fda7c425b48c516fc8f160d594c8022a0808447475c1a7c6d6479763f310c",
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
    :llama2_tokenizer => [
        (
            relative_path = "tokenizer.model",
            url = "https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.model",
            sha256 = nothing,
        ),
    ],
    :llama3_8b_tokenizer => [
        (
            relative_path = "tokenizer.json",
            url = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/tokenizer.json",
            sha256 = nothing,
        ),
        (
            relative_path = "tokenizer_config.json",
            url = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/tokenizer_config.json",
            sha256 = nothing,
        ),
    ],
)

"""
List available built-in model names.
"""
function available_models(
    ;
    format::Union{Nothing,Symbol}=nothing,
    family::Union{Nothing,Symbol}=nothing,
    distribution::Union{Nothing,Symbol}=nothing,
    shipped::Union{Nothing,Bool}=nothing,
)::Vector{Symbol}
    _ensure_local_models_loaded()
    names = collect(keys(_MODEL_REGISTRY))
    if format !== nothing
        names = [name for name in names if _matches_format(_MODEL_REGISTRY[name].format, format)]
    end
    if family !== nothing
        names = [name for name in names if _MODEL_REGISTRY[name].family == family]
    end
    if distribution !== nothing
        names = [name for name in names if _MODEL_REGISTRY[name].distribution == distribution]
    end
    if shipped !== nothing
        names = [name for name in names if _is_shipped_entry(_MODEL_REGISTRY[name]) == shipped]
    end
    sort!(names)
    return names
end

"""
Internal helper for registering/updating model entries.
"""
function _register_model_entry!(
    key::Symbol,
    path::AbstractString;
    format::Symbol,
    description::AbstractString,
    family::Symbol,
    license::AbstractString,
    upstream_repo::AbstractString,
    upstream_ref::AbstractString,
    distribution::Symbol,
    upstream_files::Vector{_UPSTREAM_FILE_INFO}=Vector{_UPSTREAM_FILE_INFO}(),
    persist::Bool=false,
)::Nothing
    resolved_path = normpath(String(path))
    _MODEL_REGISTRY[key] = _entry(
        format,
        family,
        distribution,
        nothing,
        nothing,
        resolved_path,
        String(description),
        String(license),
        String(upstream_repo),
        String(upstream_ref),
    )

    _MODEL_UPSTREAM_FILES[key] = Vector{_UPSTREAM_FILE_INFO}(upstream_files)

    if persist
        push!(_PERSISTED_LOCAL_KEYS, key)
        _write_local_models_registry()
    end
    return nothing
end

function _canonical_registry_format(format::Symbol)::Symbol
    if format in (:wordpiece, :wordpiece_vocab)
        return :wordpiece_vocab
    elseif format in (:sentencepiece, :sentencepiece_model)
        return :sentencepiece_model
    elseif format in (:bpe_encoder, :bpe_gpt2)
        return :bpe_gpt2
    elseif format in (:bpe, :bytebpe, :tiktoken, :hf_tokenizer_json, :unigram)
        return format
    end
    throw(ArgumentError("Unsupported tokenizer format for model registration: $format"))
end

_local_family_symbol(family::Nothing) = :local
_local_family_symbol(family::Symbol) = family

function _copy_local_spec_file!(
    source::AbstractString,
    target_dir::AbstractString,
    target_name::AbstractString,
)::String
    source_path = normpath(String(source))
    isfile(source_path) || throw(ArgumentError("Local model file does not exist: $source_path"))
    mkpath(target_dir)
    target_path = joinpath(target_dir, String(target_name))
    cp(source_path, target_path; force=true)
    return target_path
end

function _materialize_local_spec!(
    key::Symbol,
    spec::NamedTuple,
    default_format::Symbol,
)::Tuple{String,Symbol,Vector{String}}
    if haskey(spec, :path)
        path = normpath(String(spec[:path]))
        ispath(path) || throw(ArgumentError("Local model path does not exist: $path"))
        fmt = default_format === :auto ? detect_tokenizer_format(path) : default_format
        return (path, _canonical_registry_format(fmt), String[])
    end

    local_spec_dir = joinpath(_CACHE_ROOT, "local_specs", String(key))
    resolved_files = String[]

    if haskey(spec, :vocab) && haskey(spec, :merges)
        vocab = String(spec[:vocab])
        merges = String(spec[:merges])
        fmt = default_format === :auto ? :bpe_gpt2 : default_format
        canonical = _canonical_registry_format(fmt)

        if canonical == :bpe_gpt2
            vocab_name = lowercase(basename(vocab)) == "encoder.json" ? "encoder.json" : "vocab.json"
            merges_name = lowercase(basename(merges)) == "vocab.bpe" ? "vocab.bpe" : "merges.txt"
            push!(resolved_files, _copy_local_spec_file!(vocab, local_spec_dir, vocab_name))
            push!(resolved_files, _copy_local_spec_file!(merges, local_spec_dir, merges_name))
        else
            push!(resolved_files, _copy_local_spec_file!(vocab, local_spec_dir, "vocab.txt"))
            push!(resolved_files, _copy_local_spec_file!(merges, local_spec_dir, "merges.txt"))
        end
        return (local_spec_dir, canonical, resolved_files)
    end

    if haskey(spec, :encoder_json) && haskey(spec, :vocab_bpe)
        push!(resolved_files, _copy_local_spec_file!(String(spec[:encoder_json]), local_spec_dir, "encoder.json"))
        push!(resolved_files, _copy_local_spec_file!(String(spec[:vocab_bpe]), local_spec_dir, "vocab.bpe"))
        return (local_spec_dir, :bpe_gpt2, resolved_files)
    end

    if haskey(spec, :vocab_json) && haskey(spec, :merges_txt)
        push!(resolved_files, _copy_local_spec_file!(String(spec[:vocab_json]), local_spec_dir, "vocab.json"))
        push!(resolved_files, _copy_local_spec_file!(String(spec[:merges_txt]), local_spec_dir, "merges.txt"))
        return (local_spec_dir, :bpe_gpt2, resolved_files)
    end

    if haskey(spec, :vocab_txt)
        push!(resolved_files, _copy_local_spec_file!(String(spec[:vocab_txt]), local_spec_dir, "vocab.txt"))
        return (local_spec_dir, :wordpiece_vocab, resolved_files)
    end

    if haskey(spec, :tokenizer_json)
        push!(resolved_files, _copy_local_spec_file!(String(spec[:tokenizer_json]), local_spec_dir, "tokenizer.json"))
        if haskey(spec, :tokenizer_config_json)
            push!(resolved_files, _copy_local_spec_file!(String(spec[:tokenizer_config_json]), local_spec_dir, "tokenizer_config.json"))
        end
        if haskey(spec, :special_tokens_map_json)
            push!(resolved_files, _copy_local_spec_file!(String(spec[:special_tokens_map_json]), local_spec_dir, "special_tokens_map.json"))
        end
        return (local_spec_dir, :hf_tokenizer_json, resolved_files)
    end

    if haskey(spec, :model_file)
        source = String(spec[:model_file])
        target_name = endswith(lowercase(source), ".model.v3") ? "tokenizer.model.v3" : "tokenizer.model"
        push!(resolved_files, _copy_local_spec_file!(source, local_spec_dir, target_name))
        return (local_spec_dir, :sentencepiece_model, resolved_files)
    end

    if haskey(spec, :encoding_file)
        source = String(spec[:encoding_file])
        target_name = endswith(lowercase(source), ".tiktoken") ? basename(source) : "tokenizer.tiktoken"
        push!(resolved_files, _copy_local_spec_file!(source, local_spec_dir, target_name))
        return (local_spec_dir, :tiktoken, resolved_files)
    end

    throw(ArgumentError(
        "Unsupported local model spec for $key. Supported NamedTuple keys include " *
        "(:path), (:vocab,:merges), (:encoder_json,:vocab_bpe), (:vocab_json,:merges_txt), " *
        "(:vocab_txt), (:tokenizer_json), (:model_file), (:encoding_file).",
    ))
end

"""
Register a local tokenizer path under a symbolic key and persist it in the cache registry.
"""
function register_local_model!(
    key::Symbol,
    path_or_dir::AbstractString;
    format::Symbol=:auto,
    description::AbstractString="User-supplied local tokenizer",
    family::Union{Nothing,Symbol}=nothing,
    license::AbstractString="external-user-supplied",
    upstream_repo::AbstractString="user-supplied",
    upstream_ref::AbstractString="user-supplied",
    distribution::Symbol=:user_local,
    upstream_files::Vector{_UPSTREAM_FILE_INFO}=Vector{_UPSTREAM_FILE_INFO}(),
    notes::AbstractString="",
)::Nothing
    _ensure_local_models_loaded()
    path = normpath(String(path_or_dir))
    ispath(path) || throw(ArgumentError("Local model path does not exist: $path"))
    detected_format = format === :auto ? detect_tokenizer_format(path) : format
    canonical_format = _canonical_registry_format(detected_format)
    family_symbol = _local_family_symbol(family)

    _register_model_entry!(
        key,
        path;
        format=canonical_format,
        description=description,
        family=family_symbol,
        license=license,
        upstream_repo=upstream_repo,
        upstream_ref=upstream_ref,
        distribution=distribution,
        upstream_files=upstream_files,
        persist=false,
    )
    _LOCAL_MODEL_RESOLVED_FILES[key] = _resolved_model_files(_MODEL_REGISTRY[key], path)
    isempty(notes) || (_LOCAL_MODEL_NOTES[key] = String(notes))
    push!(_PERSISTED_LOCAL_KEYS, key)
    _write_local_models_registry()
    return nothing
end

"""
Register local model files by explicit specification.
"""
function register_local_model!(
    key::Symbol,
    spec::NamedTuple;
    format::Symbol=:auto,
    description::AbstractString="User-supplied local tokenizer",
    family::Union{Nothing,Symbol}=nothing,
    license::AbstractString="external-user-supplied",
    upstream_repo::AbstractString="user-supplied",
    upstream_ref::AbstractString="user-supplied",
    distribution::Symbol=:user_local,
    notes::AbstractString="",
)::Nothing
    _ensure_local_models_loaded()
    default_format = haskey(spec, :format) ? Symbol(spec[:format]) : format
    path, canonical_format, resolved_files = _materialize_local_spec!(key, spec, default_format)
    family_symbol = _local_family_symbol(family)

    _register_model_entry!(
        key,
        path;
        format=canonical_format,
        description=description,
        family=family_symbol,
        license=license,
        upstream_repo=upstream_repo,
        upstream_ref=upstream_ref,
        distribution=distribution,
        persist=false,
    )
    _LOCAL_MODEL_RESOLVED_FILES[key] = isempty(resolved_files) ? _resolved_model_files(_MODEL_REGISTRY[key], path) : resolved_files
    isempty(notes) || (_LOCAL_MODEL_NOTES[key] = String(notes))
    push!(_PERSISTED_LOCAL_KEYS, key)
    _write_local_models_registry()
    return nothing
end

"""
Register local model files from a `FilesSpec`.
"""
function register_local_model!(
    key::Symbol,
    spec::FilesSpec;
    kwargs...,
)::Nothing
    return register_local_model!(key, _filespec_to_namedtuple(spec); kwargs...)
end

"""
Deprecated alias kept for compatibility. Use `register_local_model!` instead.
"""
function register_external_model!(
    key::Symbol,
    path::AbstractString;
    format::Symbol=:auto,
    description::AbstractString="User-supplied external tokenizer",
    family::Symbol=:external,
    license::AbstractString="external-user-supplied",
    upstream_repo::AbstractString="user-supplied",
    upstream_ref::AbstractString="user-supplied",
)::Nothing
    Base.depwarn(
        "register_external_model! is deprecated; use register_local_model! instead.",
        :register_external_model!,
    )
    resolved_path = normpath(String(path))
    ispath(resolved_path) || throw(ArgumentError("Local model path does not exist: $resolved_path"))
    detected_format = format === :auto ? detect_tokenizer_format(resolved_path) : format
    canonical_format = _canonical_registry_format(detected_format)

    _register_model_entry!(
        key,
        resolved_path;
        format=canonical_format,
        description=description,
        family=family,
        license=license,
        upstream_repo=upstream_repo,
        upstream_ref=upstream_ref,
        distribution=:user_local,
        persist=false,
    )
    _LOCAL_MODEL_RESOLVED_FILES[key] = _resolved_model_files(_MODEL_REGISTRY[key], resolved_path)
    return nothing
end

"""
Download selected files from a Hugging Face repository revision into cache.

This helper is opt-in and useful for user-managed / gated tokenizers.
"""
function download_hf_files(
    repo_id::AbstractString,
    filenames::AbstractVector{<:AbstractString};
    revision::AbstractString="main",
    outdir::Union{Nothing,AbstractString}=nothing,
    token::Union{Nothing,AbstractString}=nothing,
    force::Bool=false,
)::Vector{String}
    _ensure_local_models_loaded()
    target_dir = outdir === nothing ?
        joinpath(_CACHE_ROOT, "hf", _sanitize_cache_segment(repo_id), _sanitize_cache_segment(String(revision))) :
        String(outdir)
    mkpath(target_dir)

    headers = Pair{String,String}[]
    token !== nothing && push!(headers, "Authorization" => "Bearer $(token)")

    outputs = String[]
    for file in filenames
        relative = String(file)
        url = "https://huggingface.co/$(repo_id)/resolve/$(revision)/$(relative)"
        local_path = joinpath(target_dir, splitpath(relative)...)
        mkpath(dirname(local_path))

        if !force && isfile(local_path)
            push!(outputs, local_path)
            continue
        end

        try
            Downloads.download(url, local_path; headers=headers)
        catch err
            throw(ArgumentError("Failed to download Hugging Face file $(repo_id)/$(relative)@$(revision): $(sprint(showerror, err))"))
        end

        push!(outputs, local_path)
    end

    return outputs
end

"""
Install an installable-gated tokenizer into the user cache and register it by key.
"""
function install_model!(
    key::Symbol;
    token::Union{Nothing,AbstractString}=nothing,
    revision::AbstractString="main",
    force::Bool=false,
)::NamedTuple
    _ensure_local_models_loaded()
    haskey(_MODEL_REGISTRY, key) || throw(ArgumentError("Unknown model key: $key"))
    entry = _MODEL_REGISTRY[key]
    entry.distribution == :installable_gated || throw(ArgumentError(
        "Model $key is not installable_gated (distribution=$(entry.distribution)). Use prefetch_models for shipped/artifact models.",
    ))

    token_value = token === nothing ? nothing : String(token)
    (token_value === nothing || isempty(strip(token_value))) && throw(ArgumentError(
        "Model $key is gated. Provide an access token (for example token=ENV[\"HF_TOKEN\"]) after accepting the upstream license.",
    ))

    upstream_files = get(_MODEL_UPSTREAM_FILES, key, Vector{_UPSTREAM_FILE_INFO}())
    isempty(upstream_files) && throw(ArgumentError("No upstream files are registered for installable model $key"))

    repo_id = entry.upstream_repo
    isempty(repo_id) && throw(ArgumentError("Installable model $key does not define an upstream repository."))

    filenames = [f.relative_path for f in upstream_files]
    install_dir = joinpath(_CACHE_ROOT, "install", String(key))
    download_hf_files(
        repo_id,
        filenames;
        revision=revision,
        outdir=install_dir,
        token=token_value,
        force=force,
    )

    register_local_model!(
        key,
        install_dir;
        format=entry.format,
        description=entry.description,
        family=entry.family,
        license=entry.license,
        upstream_repo=repo_id,
        upstream_ref="huggingface:$(repo_id)@$(revision)",
        distribution=:installable_gated,
        upstream_files=upstream_files,
    )

    return describe_model(key)
end

"""
Install the gated LLaMA 2 tokenizer files into local cache and register them.

This is a convenience wrapper over `install_model!(:llama2_tokenizer; ...)`.
"""
install_llama2_tokenizer!(; kwargs...) = install_model!(:llama2_tokenizer; kwargs...)

"""
Install the gated LLaMA 3 8B tokenizer files into local cache and register them.

This is a convenience wrapper over `install_model!(:llama3_8b_tokenizer; ...)`.
"""
install_llama3_8b_tokenizer!(; kwargs...) = install_model!(:llama3_8b_tokenizer; kwargs...)

"""
Recommended built-in keys for LLM-oriented default prefetching.
"""
function recommended_defaults_for_llms()::Vector{Symbol}
    candidates = [
        :tiktoken_cl100k_base,
        :tiktoken_o200k_base,
        :mistral_v3_sentencepiece,
        :phi2_bpe,
        :qwen2_5_bpe,
        :roberta_base_bpe,
        :xlm_roberta_base_sentencepiece_bpe,
    ]
    return [key for key in candidates if haskey(_MODEL_REGISTRY, key)]
end

"""
Ensure artifact-backed built-in models are present on disk.

Returns a dictionary of `key => is_available`.
"""
function prefetch_models(
    keys::AbstractVector{Symbol}=available_models(shipped=true);
    force::Bool=false,
)::Dict{Symbol,Bool}
    _ensure_local_models_loaded()
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
    _ensure_local_models_loaded()
    haskey(_MODEL_REGISTRY, name) || throw(ArgumentError("Unknown built-in model: $name"))
    entry = _MODEL_REGISTRY[name]
    resolved = _resolve_model_path(entry)
    source = _resolve_model_source(entry, resolved)
    upstream = get(_MODEL_UPSTREAM_FILES, name, Vector{_UPSTREAM_FILE_INFO}())
    files = _resolved_model_files(entry, resolved)
    exists = !isempty(files) && all(ispath, files)
    resolved_files = get(_LOCAL_MODEL_RESOLVED_FILES, name, files)
    notes = get(_LOCAL_MODEL_NOTES, name, "")

    return (
        name = name,
        format = entry.format,
        path = resolved,
        files = files,
        resolved_files = resolved_files,
        expected_files = _expected_model_files(entry),
        exists = exists,
        source = source,
        shipped = _is_shipped_entry(entry),
        description = entry.description,
        family = entry.family,
        distribution = entry.distribution,
        license = entry.license,
        upstream_repo = entry.upstream_repo,
        upstream_ref = entry.upstream_ref,
        artifact_name = entry.artifact_name,
        upstream_files = upstream,
        provenance_urls = [f.url for f in upstream],
        notes = notes,
    )
end

function _matches_format(model_format::Symbol, query_format::Symbol)::Bool
    if model_format == query_format
        return true
    elseif query_format == :bpe_encoder
        return model_format in (:bpe_gpt2, :hf_tokenizer_json)
    elseif query_format == :wordpiece
        return model_format in (:wordpiece, :wordpiece_vocab)
    elseif query_format == :sentencepiece_model
        return model_format in (:sentencepiece, :sentencepiece_model)
    elseif query_format == :sentencepiece
        return model_format in (:sentencepiece, :sentencepiece_model)
    elseif query_format == :bpe_gpt2
        return model_format in (:bpe_gpt2, :hf_tokenizer_json)
    elseif query_format == :bpe
        return model_format in (:bpe, :bpe_gpt2, :bytebpe, :hf_tokenizer_json)
    end

    return false
end

"""
Resolve built-in model name to on-disk path.
"""
function model_path(name::Symbol; auto_prefetch::Bool=true)::String
    _ensure_local_models_loaded()
    info = describe_model(name)

    if !info.exists && auto_prefetch
        prefetch_models([name])
        info = describe_model(name)
    end

    if !info.exists
        if info.distribution == :installable_gated
            throw(ArgumentError(
                "Model asset missing on disk for $name at $(info.path). Install it first with install_model!(:$name; token=ENV[\"HF_TOKEN\"]).",
            ))
        end
        throw(ArgumentError("Model asset missing on disk for $name at $(info.path)"))
    end
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

    if ispath(resolved)
        if startswith(normpath(resolved), normpath(joinpath(_PACKAGE_ROOT, "models")))
            return :in_repo
        end
        return :external
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
            if !isempty(matches)
                return String[matches...]
            end

            fallback = joinpath(resolved, "tokenizer.model")
            isfile(fallback) && return String[fallback]
            return String[matches...]
        end
        return String[resolved]
    elseif entry.format == :hf_tokenizer_json
        if isdir(resolved)
            tokenizer_json = joinpath(resolved, "tokenizer.json")
            if isfile(tokenizer_json)
                return String[tokenizer_json]
            end

            vocab_json = joinpath(resolved, "vocab.json")
            merges_txt = joinpath(resolved, "merges.txt")
            if isfile(vocab_json) && isfile(merges_txt)
                return String[vocab_json, merges_txt]
            end

            encoder_json = joinpath(resolved, "encoder.json")
            vocab_bpe = joinpath(resolved, "vocab.bpe")
            if isfile(encoder_json) && isfile(vocab_bpe)
                return String[encoder_json, vocab_bpe]
            end

            return String[tokenizer_json]
        end
        return String[resolved]
    end

    return String[resolved]
end

function _expected_model_files(entry::BuiltinModelEntry)::Vector{String}
    if entry.format in (:bpe, :bytebpe)
        return String["vocab.txt", "merges.txt"]
    elseif entry.format == :bpe_gpt2
        return String["vocab.json + merges.txt", "encoder.json + vocab.bpe"]
    elseif entry.format in (:wordpiece, :wordpiece_vocab)
        return String["vocab.txt"]
    elseif entry.format in (:sentencepiece, :sentencepiece_model)
        return String["spm.model / tokenizer.model / tokenizer.model.v3 / sentencepiece.bpe.model"]
    elseif entry.format == :tiktoken
        return String["*.tiktoken or tokenizer.model (tiktoken text)"]
    elseif entry.format == :hf_tokenizer_json
        return String["tokenizer.json (preferred)", "vocab.json + merges.txt (fallback)"]
    end
    return String["<unknown>"]
end

function _is_shipped_entry(entry::BuiltinModelEntry)::Bool
    return entry.distribution in (:shipped, :artifact_public)
end

function _ensure_local_models_loaded()::Nothing
    _LOCAL_MODELS_LOADED[] && return nothing
    _LOCAL_MODELS_LOADED[] = true

    isfile(_LOCAL_MODELS_TOML) || return nothing
    parsed = try
        TOML.parsefile(_LOCAL_MODELS_TOML)
    catch
        return nothing
    end

    models = get(parsed, "models", Dict{String,Any}())
    models isa AbstractDict || return nothing

    for (key_raw, entry_any) in models
        key = Symbol(String(key_raw))
        entry_any isa AbstractDict || continue

        path = get(entry_any, "path", nothing)
        format_raw = get(entry_any, "format", nothing)
        path isa AbstractString || continue
        format_raw isa AbstractString || continue

        description = String(get(entry_any, "description", "User-supplied local tokenizer"))
        family = Symbol(String(get(entry_any, "family", "local")))
        license = String(get(entry_any, "license", "external-user-supplied"))
        upstream_repo = String(get(entry_any, "upstream_repo", "user-supplied"))
        upstream_ref = String(get(entry_any, "upstream_ref", "user-supplied"))
        distribution = Symbol(String(get(entry_any, "distribution", "user_local")))
        distribution in _VALID_MODEL_DISTRIBUTIONS || (distribution = :user_local)
        notes = String(get(entry_any, "notes", ""))
        resolved_files_raw = get(entry_any, "resolved_files", String[])
        resolved_files = String[]
        if resolved_files_raw isa AbstractVector
            append!(resolved_files, [normpath(String(p)) for p in resolved_files_raw])
        end

        _register_model_entry!(
            key,
            String(path);
            format=Symbol(format_raw),
            description=description,
            family=family,
            license=license,
            upstream_repo=upstream_repo,
            upstream_ref=upstream_ref,
            distribution=distribution,
            persist=false,
        )

        _LOCAL_MODEL_RESOLVED_FILES[key] = isempty(resolved_files) ?
            _resolved_model_files(_MODEL_REGISTRY[key], normpath(String(path))) :
            resolved_files
        isempty(notes) || (_LOCAL_MODEL_NOTES[key] = notes)
        push!(_PERSISTED_LOCAL_KEYS, key)
    end

    return nothing
end

function _write_local_models_registry()::Nothing
    mkpath(dirname(_LOCAL_MODELS_TOML))
    models = Dict{String,Any}()

    for key in sort!(collect(_PERSISTED_LOCAL_KEYS))
        haskey(_MODEL_REGISTRY, key) || continue
        entry = _MODEL_REGISTRY[key]
        models[String(key)] = Dict(
            "path" => entry.fallback_path,
            "format" => String(entry.format),
            "description" => entry.description,
            "family" => String(entry.family),
            "distribution" => String(entry.distribution),
            "license" => entry.license,
            "upstream_repo" => entry.upstream_repo,
            "upstream_ref" => entry.upstream_ref,
            "resolved_files" => get(_LOCAL_MODEL_RESOLVED_FILES, key, _resolved_model_files(entry, entry.fallback_path)),
            "notes" => get(_LOCAL_MODEL_NOTES, key, ""),
        )
    end

    open(_LOCAL_MODELS_TOML, "w") do io
        TOML.print(io, Dict("models" => models))
    end
    return nothing
end

function _sanitize_cache_segment(value::AbstractString)::String
    raw = String(value)
    out = IOBuffer()
    for c in raw
        if isletter(c) || isnumeric(c) || c in ('-', '_', '.')
            print(out, c)
        else
            print(out, '_')
        end
    end
    result = String(take!(out))
    return isempty(result) ? "_" : result
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
