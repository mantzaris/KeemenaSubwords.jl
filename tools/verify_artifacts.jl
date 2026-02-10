using Pkg.Artifacts
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const ARTIFACTS_TOML = joinpath(REPO_ROOT, "Artifacts.toml")
const DEFAULT_ARTIFACTS = String[
    "keemena_public_tokenizer_assets_v1",
    "mistral_v1_sentencepiece",
    "mistral_v3_sentencepiece",
    "phi2_bpe",
    "qwen2_5_bpe",
    "roberta_base_bpe",
    "xlm_roberta_base_sentencepiece_bpe",
    "bert_base_multilingual_cased_wordpiece",
]

function artifact_urls(artifact_name::String)::Vector{String}
    parsed = try
        TOML.parsefile(ARTIFACTS_TOML)
    catch
        return String[]
    end
    section = get(parsed, artifact_name, nothing)
    section isa AbstractDict || return String[]

    downloads = get(section, "download", nothing)
    urls = String[]
    if downloads isa AbstractVector
        for dl in downloads
            dl isa AbstractDict || continue
            url = get(dl, "url", nothing)
            url isa AbstractString && push!(urls, String(url))
        end
    elseif downloads isa AbstractDict
        url = get(downloads, "url", nothing)
        url isa AbstractString && push!(urls, String(url))
    end

    return urls
end

function verify_artifact(artifact_name::String; force::Bool=false)::Bool
    hash = Artifacts.artifact_hash(artifact_name, ARTIFACTS_TOML)
    urls = artifact_urls(artifact_name)
    println("\n==> ", artifact_name)
    println("    urls: ", isempty(urls) ? "<none>" : join(urls, ", "))

    if hash === nothing
        println("    status: FAIL (artifact not bound in Artifacts.toml)")
        return false
    end

    println("    git-tree-sha1: ", hash)
    already = Artifacts.artifact_exists(hash)
    println("    installed_before: ", already)

    if force && already
        rm(Artifacts.artifact_path(hash); recursive=true, force=true)
        println("    removed existing installation due to force=true")
    end

    try
        Artifacts.ensure_artifact_installed(artifact_name, ARTIFACTS_TOML)
    catch err
        println("    status: FAIL")
        println("    error: ", sprint(showerror, err))
        return false
    end

    installed = Artifacts.artifact_exists(hash)
    path = installed ? Artifacts.artifact_path(hash) : "<missing>"
    println("    installed_after: ", installed)
    println("    artifact_path: ", path)

    if !installed
        println("    status: FAIL (ensure_artifact_installed returned without installation)")
        return false
    end

    println("    status: OK")
    return true
end

function main()::Nothing
    if !isfile(ARTIFACTS_TOML)
        error("Artifacts.toml not found at $(ARTIFACTS_TOML)")
    end

    force = get(ENV, "KEEMENA_ARTIFACT_FORCE", "0") == "1"
    artifacts = copy(DEFAULT_ARTIFACTS)
    if !isempty(ARGS)
        artifacts = String[arg for arg in ARGS]
    end

    println("Artifacts.toml: ", ARTIFACTS_TOML)
    println("force: ", force)
    println("artifacts: ", join(artifacts, ", "))

    failed = String[]
    for artifact_name in artifacts
        ok = verify_artifact(artifact_name; force=force)
        ok || push!(failed, artifact_name)
    end

    if isempty(failed)
        println("\nAll requested artifacts verified successfully.")
        return nothing
    end

    println("\nArtifact verification failed for: ", join(failed, ", "))
    exit(1)
end

main()
