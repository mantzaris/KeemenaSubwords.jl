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

function _split_lines(text::String)::Vector{String}
    return split(chomp(text), "\n"; keepempty=true)
end

function _truncate_line(line::AbstractString; limit::Int=200)::String
    line_str = String(line)
    if length(line_str) <= limit || limit <= 3
        return line_str
    end
    return string(first(line_str, limit - 3), "...")
end

function _print_mismatch_diagnostics(source_text::String, target_text::String)::Nothing
    source_lines = _split_lines(source_text)
    target_lines = _split_lines(target_text)
    common_length = min(length(source_lines), length(target_lines))

    first_mismatch = nothing
    for i in 1:common_length
        if source_lines[i] != target_lines[i]
            first_mismatch = i
            break
        end
    end

    if first_mismatch !== nothing
        line_index = first_mismatch::Int
        println(stderr, "  first mismatch line: $line_index")
        println(stderr, "  source line: ", repr(_truncate_line(source_lines[line_index])))
        println(stderr, "  target line: ", repr(_truncate_line(target_lines[line_index])))
        return nothing
    end

    line_index = common_length + 1
    println(stderr, "  first mismatch line: $line_index")
    if length(source_lines) > length(target_lines)
        println(stderr, "  source has extra trailing lines starting at line $line_index.")
        println(stderr, "  source line: ", repr(_truncate_line(source_lines[line_index])))
        println(stderr, "  target line: <no line>")
    else
        println(stderr, "  target has extra trailing lines starting at line $line_index.")
        println(stderr, "  source line: <no line>")
        println(stderr, "  target line: ", repr(_truncate_line(target_lines[line_index])))
    end
    return nothing
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
        println(stderr, "Offset contract is out of sync.")
        println(stderr, "  source: $SOURCE")
        println(stderr, "  target: $TARGET")
        _print_mismatch_diagnostics(source_text, target_text)
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
