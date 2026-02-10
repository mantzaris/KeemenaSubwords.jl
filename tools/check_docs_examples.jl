#!/usr/bin/env julia

const ROOT = normpath(joinpath(@__DIR__, ".."))
const README_PATH = joinpath(ROOT, "README.md")
const DOCS_SRC = joinpath(ROOT, "docs", "src")
const SRC_DIR = joinpath(ROOT, "src")

function _markdown_files()::Vector{String}
    files = String[README_PATH]
    for entry in sort(readdir(DOCS_SRC))
        endswith(entry, ".md") || continue
        push!(files, joinpath(DOCS_SRC, entry))
    end
    return files
end

function _source_files()::Vector{String}
    files = String[]
    for (dir, _, names) in walkdir(SRC_DIR)
        for name in names
            endswith(name, ".jl") || continue
            push!(files, joinpath(dir, name))
        end
    end
    sort!(files)
    return files
end

function _code_blocks(text::String)::Vector{String}
    blocks = String[]
    for m in eachmatch(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)\n```"s, text)
        push!(blocks, String(m.captures[1]))
    end
    return blocks
end

function _push_block_error!(
    errors::Vector{String},
    file::AbstractString,
    block_index::Int,
    message::AbstractString,
)::Nothing
    push!(errors, "$(relpath(file, ROOT)) [code block $(block_index)]: $(message)")
    return nothing
end

function _check_code_block!(errors::Vector{String}, file::String, block::String, idx::Int)::Nothing
    for raw_line in split(block, '\n')
        line = strip(raw_line)
        isempty(line) && continue

        if occursin(r"format\s*=\s*:bpe_gpt2", line) && occursin(r"vocab\.txt", line)
            _push_block_error!(
                errors,
                file,
                idx,
                "found format=:bpe_gpt2 paired with vocab.txt; expected vocab.json + merges.txt",
            )
        end
        if occursin(r"load_bpe_gpt2\s*\(", line) && occursin(r"vocab\.txt", line)
            _push_block_error!(
                errors,
                file,
                idx,
                "found load_bpe_gpt2(...) with vocab.txt; expected vocab.json + merges.txt",
            )
        end
        if occursin(r"format\s*=\s*:bpe_encoder", line) && (occursin(r"vocab\.txt", line) || occursin(r"merges\.txt", line))
            _push_block_error!(
                errors,
                file,
                idx,
                "found format=:bpe_encoder paired with vocab.txt/merges.txt; expected encoder.json + vocab.bpe",
            )
        end
        if occursin(r"load_bpe_encoder\s*\(", line) && (occursin(r"vocab\.txt", line) || occursin(r"merges\.txt", line))
            _push_block_error!(
                errors,
                file,
                idx,
                "found load_bpe_encoder(...) with vocab.txt/merges.txt; expected encoder.json + vocab.bpe",
            )
        end
        if occursin(r"load_wordpiece\s*\(", line) && (occursin(r"vocab\.json", line) || occursin(r"merges\.txt", line))
            _push_block_error!(
                errors,
                file,
                idx,
                "found load_wordpiece(...) with vocab.json/merges.txt; expected vocab.txt",
            )
        end
        if occursin(r"load_wordpiece\s*\(", line) && (occursin(r"encoder\.json", line) || occursin(r"vocab\.bpe", line))
            _push_block_error!(
                errors,
                file,
                idx,
                "found load_wordpiece(...) with encoder.json/vocab.bpe; expected vocab.txt",
            )
        end
        if occursin(r"format\s*=\s*:(wordpiece|wordpiece_vocab)", line) &&
           (occursin(r"vocab\.json", line) || occursin(r"merges\.txt", line) || occursin(r"encoder\.json", line) || occursin(r"vocab\.bpe", line))
            _push_block_error!(
                errors,
                file,
                idx,
                "found WordPiece format paired with BPE files; expected vocab.txt",
            )
        end
    end
    return nothing
end

function main()::Nothing
    errors = String[]

    for file in _markdown_files()
        text = read(file, String)
        rel = relpath(file, ROOT)

        if occursin("register_external_model!", text) && rel != "docs/src/api.md"
            push!(
                errors,
                "$rel: found register_external_model! outside API deprecation note; use register_local_model! in examples",
            )
        end

        for (idx, block) in enumerate(_code_blocks(text))
            _check_code_block!(errors, file, block, idx)
        end
    end

    for file in _source_files()
        rel = relpath(file, ROOT)
        for (lineno, raw_line) in enumerate(eachline(file))
            line = strip(raw_line)
            isempty(line) && continue

            if occursin(r"format\s*=\s*:bpe_gpt2", line) && occursin(r"vocab\.txt", line)
                push!(errors, "$rel:$lineno: found format=:bpe_gpt2 paired with vocab.txt; expected vocab.json + merges.txt")
            end
            if occursin(r"load_bpe_gpt2\s*\(", line) && occursin(r"vocab\.txt", line)
                push!(errors, "$rel:$lineno: found load_bpe_gpt2(...) with vocab.txt; expected vocab.json + merges.txt")
            end
            if occursin(r"format\s*=\s*:bpe_encoder", line) && (occursin(r"vocab\.txt", line) || occursin(r"merges\.txt", line))
                push!(errors, "$rel:$lineno: found format=:bpe_encoder paired with vocab.txt/merges.txt; expected encoder.json + vocab.bpe")
            end
            if occursin(r"load_bpe_encoder\s*\(", line) && (occursin(r"vocab\.txt", line) || occursin(r"merges\.txt", line))
                push!(errors, "$rel:$lineno: found load_bpe_encoder(...) with vocab.txt/merges.txt; expected encoder.json + vocab.bpe")
            end
            if occursin(r"load_wordpiece\s*\(", line) && (occursin(r"vocab\.json", line) || occursin(r"merges\.txt", line))
                push!(errors, "$rel:$lineno: found load_wordpiece(...) with vocab.json/merges.txt; expected vocab.txt")
            end
            if occursin(r"load_wordpiece\s*\(", line) && (occursin(r"encoder\.json", line) || occursin(r"vocab\.bpe", line))
                push!(errors, "$rel:$lineno: found load_wordpiece(...) with encoder.json/vocab.bpe; expected vocab.txt")
            end
            if occursin(r"format\s*=\s*:(wordpiece|wordpiece_vocab)", line) &&
               (occursin(r"vocab\.json", line) || occursin(r"merges\.txt", line) || occursin(r"encoder\.json", line) || occursin(r"vocab\.bpe", line))
                push!(errors, "$rel:$lineno: found WordPiece format paired with BPE files; expected vocab.txt")
            end
        end
    end

    if isempty(errors)
        println("Docs examples passed consistency checks.")
        return nothing
    end

    println("Docs example consistency check failed:")
    for err in errors
        println(" - ", err)
    end
    error("Fix docs example inconsistencies listed above.")
end

main()
