#!/usr/bin/env julia

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE = joinpath(ROOT, "notes", "OffsetContract.md")
const TARGET = joinpath(ROOT, "docs", "src", "normalization_offsets_contract.md")

function _usage()::String
    return "Usage: julia --project=. tools/sync_offset_contract.jl [--check]"
end

function _read_text(path::String)::String
    isfile(path) || error("Missing file: $path")
    return read(path, String)
end

function _normalize_newlines(text::String)::String
    normalized = replace(text, "\r\n" => "\n")
    return endswith(normalized, "\n") ? normalized : normalized * "\n"
end

function main(args::Vector{String})::Int
    check_mode = false
    for arg in args
        if arg == "--check"
            check_mode = true
        elseif arg in ("-h", "--help")
            println(_usage())
            return 0
        else
            println(stderr, "Unknown argument: $arg")
            println(stderr, _usage())
            return 2
        end
    end

    source_text = _normalize_newlines(_read_text(SOURCE))

    if check_mode
        target_text = isfile(TARGET) ? _normalize_newlines(_read_text(TARGET)) : ""
        if source_text == target_text
            println("Offset contract is in sync.")
            return 0
        end
        println(stderr, "Offset contract is out of sync:")
        println(stderr, "  source: $SOURCE")
        println(stderr, "  target: $TARGET")
        println(stderr, "Run: julia --project=. tools/sync_offset_contract.jl")
        return 1
    end

    mkpath(dirname(TARGET))
    write(TARGET, source_text)
    println("Synced offset contract:")
    println("  source: $SOURCE")
    println("  target: $TARGET")
    return 0
end

exit(main(ARGS))
