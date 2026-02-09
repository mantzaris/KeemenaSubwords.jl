#!/usr/bin/env julia

using Dates
using Downloads
using Pkg.Artifacts
using SHA
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const ARTIFACTS_TOML = joinpath(REPO_ROOT, "Artifacts.toml")
const ARTIFACT_NAME = get(ENV, "KEEMENA_ARTIFACT_NAME", "keemena_public_tokenizer_assets_v1")
const OUTDIR = get(ENV, "KEEMENA_ARTIFACT_OUTDIR", joinpath(REPO_ROOT, "artifacts-build"))

const ASSET_SPECS = [
    (
        relative_path = "tiktoken/o200k_base/o200k_base.tiktoken",
        url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        sha256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    ),
    (
        relative_path = "tiktoken/cl100k_base/cl100k_base.tiktoken",
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        sha256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
    (
        relative_path = "tiktoken/r50k_base/r50k_base.tiktoken",
        url = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        sha256 = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    ),
    (
        relative_path = "tiktoken/p50k_base/p50k_base.tiktoken",
        url = "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
        sha256 = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    ),
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
    (
        relative_path = "wordpiece/bert_base_uncased/vocab.txt",
        url = "https://huggingface.co/bert-base-uncased/resolve/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt",
        sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
    (
        relative_path = "sentencepiece/t5_small/spiece.model",
        url = "https://huggingface.co/google-t5/t5-small/resolve/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/spiece.model",
        sha256 = "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    ),
]

function file_sha256(path::AbstractString)::String
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function verify_sha256(path::AbstractString, expected::Union{Nothing,String})::String
    actual = lowercase(file_sha256(path))
    if expected !== nothing && lowercase(expected) != actual
        throw(ArgumentError("SHA-256 mismatch for $(path): expected=$(expected), actual=$(actual)"))
    end
    return actual
end

function write_metadata(path::AbstractString, records)::Nothing
    payload = Dict(
        "generated_at_utc" => string(Dates.now(Dates.UTC)),
        "artifact_name" => ARTIFACT_NAME,
        "files" => [
            Dict(
                "relative_path" => rec.relative_path,
                "url" => rec.url,
                "sha256" => rec.sha256,
            ) for rec in records
        ],
    )

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return nothing
end

function build_artifact()::Tuple{String,String,String}
    mkpath(OUTDIR)
    recorded = NamedTuple[]

    hash = create_artifact() do artifact_dir
        for spec in ASSET_SPECS
            out_path = joinpath(artifact_dir, spec.relative_path)
            mkpath(dirname(out_path))

            println("Downloading ", spec.url)
            Downloads.download(spec.url, out_path)
            actual_sha = verify_sha256(out_path, spec.sha256)

            push!(recorded, (
                relative_path = spec.relative_path,
                url = spec.url,
                sha256 = actual_sha,
            ))
        end

        metadata_path = joinpath(artifact_dir, "metadata", "upstream_hashes.toml")
        mkpath(dirname(metadata_path))
        write_metadata(metadata_path, recorded)
    end

    bind_artifact!(ARTIFACTS_TOML, ARTIFACT_NAME, hash; force=true, lazy=true)

    hash_str = hash isa Base.SHA1 ? bytes2hex(hash.bytes) : string(hash)

    tarball = joinpath(OUTDIR, "$(ARTIFACT_NAME)-$(hash_str).tar.gz")
    archive_artifact(hash, tarball)
    tar_sha256 = file_sha256(tarball)

    return (hash_str, tarball, tar_sha256)
end

function main()::Nothing
    println("Building artifact $(ARTIFACT_NAME)")
    println("Artifacts.toml: ", ARTIFACTS_TOML)
    println("Output dir: ", OUTDIR)

    hash, tarball, tar_sha256 = build_artifact()

    println()
    println("Artifact build complete.")
    println("  git-tree-sha1: ", hash)
    println("  tarball: ", tarball)
    println("  tarball sha256: ", tar_sha256)
    println()
    println("Next steps:")
    println("1) Upload tarball to a GitHub release for this repository.")
    println("2) Add a download stanza under [$(ARTIFACT_NAME)] in Artifacts.toml:")
    println()
    println("[[$(ARTIFACT_NAME).download]]")
    println("url = \"https://github.com/<owner>/<repo>/releases/download/<tag>/$(basename(tarball))\"")
    println("sha256 = \"$(tar_sha256)\"")
    println()
    println("3) Commit Artifacts.toml and the tooling updates.")
end

main()
