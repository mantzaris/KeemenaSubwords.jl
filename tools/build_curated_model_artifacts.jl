#!/usr/bin/env julia

using Dates
using Downloads
using Pkg.Artifacts
using SHA
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const ARTIFACTS_TOML = joinpath(REPO_ROOT, "Artifacts.toml")
const OUTDIR = get(ENV, "KEEMENA_ARTIFACT_OUTDIR", joinpath(REPO_ROOT, "artifacts-build"))
const RELEASE_BASE_URL = get(ENV, "KEEMENA_RELEASE_BASE_URL", "")

const MODEL_SPECS = [
    (
        key = :mistral_v1_sentencepiece,
        artifact_name = "mistral_v1_sentencepiece",
        license = "Apache-2.0",
        upstream_ref = "huggingface:mistralai/Mixtral-8x7B-Instruct-v0.1@eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
        files = [
            (
                name = "tokenizer.model",
                url = "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/resolve/eba92302a2861cdc0098cc54bc9f17cb2c47eb61/tokenizer.model",
                sha256 = "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055",
            ),
        ],
    ),
    (
        key = :mistral_v3_sentencepiece,
        artifact_name = "mistral_v3_sentencepiece",
        license = "Apache-2.0",
        upstream_ref = "huggingface:mistralai/Mistral-7B-Instruct-v0.3@c170c708c41dac9275d15a8fff4eca08d52bab71",
        files = [
            (
                name = "tokenizer.model.v3",
                url = "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/c170c708c41dac9275d15a8fff4eca08d52bab71/tokenizer.model.v3",
                sha256 = "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
            ),
        ],
    ),
    (
        key = :phi2_bpe,
        artifact_name = "phi2_bpe",
        license = "MIT",
        upstream_ref = "huggingface:microsoft/phi-2@810d367871c1d460086d9f82db8696f2e0a0fcd0",
        files = [
            (
                name = "vocab.json",
                url = "https://huggingface.co/microsoft/phi-2/resolve/810d367871c1d460086d9f82db8696f2e0a0fcd0/vocab.json",
                sha256 = "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7",
            ),
            (
                name = "merges.txt",
                url = "https://huggingface.co/microsoft/phi-2/resolve/810d367871c1d460086d9f82db8696f2e0a0fcd0/merges.txt",
                sha256 = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            ),
        ],
    ),
    (
        key = :roberta_base_bpe,
        artifact_name = "roberta_base_bpe",
        license = "MIT",
        upstream_ref = "huggingface:FacebookAI/roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        files = [
            (
                name = "vocab.json",
                url = "https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/vocab.json",
                sha256 = "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
            ),
            (
                name = "merges.txt",
                url = "https://huggingface.co/FacebookAI/roberta-base/resolve/e2da8e2f811d1448a5b465c236feacd80ffbac7b/merges.txt",
                sha256 = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            ),
        ],
    ),
    (
        key = :xlm_roberta_base_sentencepiece_bpe,
        artifact_name = "xlm_roberta_base_sentencepiece_bpe",
        license = "MIT",
        upstream_ref = "huggingface:FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        files = [
            (
                name = "sentencepiece.bpe.model",
                url = "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/sentencepiece.bpe.model",
                sha256 = "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            ),
        ],
    ),
]

function file_sha256(path::AbstractString)::String
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function verify_sha256(path::AbstractString, expected::String)::String
    actual = lowercase(file_sha256(path))
    lowercase(expected) == actual || throw(ArgumentError("SHA-256 mismatch for $(path): expected=$(expected), actual=$(actual)"))
    return actual
end

function write_metadata(path::AbstractString, spec, records)::Nothing
    payload = Dict(
        "generated_at_utc" => string(Dates.now(Dates.UTC)),
        "key" => String(spec.key),
        "artifact_name" => spec.artifact_name,
        "license" => spec.license,
        "upstream_ref" => spec.upstream_ref,
        "files" => [
            Dict(
                "name" => rec.name,
                "url" => rec.url,
                "sha256" => rec.sha256,
                "bytes" => rec.bytes,
            ) for rec in records
        ],
    )

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return nothing
end

function build_one(spec)::NamedTuple
    mkpath(OUTDIR)
    recorded = NamedTuple[]

    hash = create_artifact() do artifact_dir
        model_dir = joinpath(artifact_dir, spec.artifact_name)
        mkpath(model_dir)

        for file_spec in spec.files
            out_path = joinpath(model_dir, file_spec.name)
            println("Downloading ", file_spec.url)
            Downloads.download(file_spec.url, out_path)
            actual_sha = verify_sha256(out_path, file_spec.sha256)
            bytes = filesize(out_path)

            push!(recorded, (
                name = file_spec.name,
                url = file_spec.url,
                sha256 = actual_sha,
                bytes = bytes,
            ))
        end

        metadata_path = joinpath(model_dir, "upstream_hashes.toml")
        write_metadata(metadata_path, spec, recorded)
    end

    bind_artifact!(ARTIFACTS_TOML, spec.artifact_name, hash; force=true, lazy=true)

    hash_str = hash isa Base.SHA1 ? bytes2hex(hash.bytes) : string(hash)
    tar_name = "$(spec.artifact_name)-$(hash_str).tar.gz"
    tar_path = joinpath(OUTDIR, tar_name)
    archive_artifact(hash, tar_path)
    tar_sha256 = file_sha256(tar_path)

    return (
        key = spec.key,
        artifact_name = spec.artifact_name,
        hash = hash_str,
        tar_name = tar_name,
        tar_path = tar_path,
        tar_sha256 = tar_sha256,
    )
end

function print_release_stanza(result)::Nothing
    println()
    println("[", result.artifact_name, "]")
    println("git-tree-sha1 = \"", result.hash, "\"")
    println("lazy = true")

    if !isempty(RELEASE_BASE_URL)
        println("[[", result.artifact_name, ".download]]")
        println("url = \"", RELEASE_BASE_URL, "/", result.tar_name, "\"")
        println("sha256 = \"", result.tar_sha256, "\"")
    else
        println("# add download after uploading tarball:")
        println("# [[", result.artifact_name, ".download]]")
        println("# url = \"https://github.com/<owner>/<repo>/releases/download/<tag>/", result.tar_name, "\"")
        println("# sha256 = \"", result.tar_sha256, "\"")
    end
end

function main()::Nothing
    println("Building curated tokenizer artifacts")
    println("Artifacts.toml: ", ARTIFACTS_TOML)
    println("Output dir: ", OUTDIR)

    results = NamedTuple[]
    for spec in MODEL_SPECS
        println()
        println("==> ", spec.key, " (", spec.artifact_name, ")")
        result = build_one(spec)
        push!(results, result)
        println("  git-tree-sha1: ", result.hash)
        println("  tarball: ", result.tar_path)
        println("  tarball sha256: ", result.tar_sha256)
    end

    println("\nArtifacts.toml stanza suggestions:")
    for result in results
        print_release_stanza(result)
    end
end

main()
