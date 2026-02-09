#!/usr/bin/env julia

using KeemenaSubwords

const DOCS_MODELS_PATH = normpath(joinpath(@__DIR__, "..", "docs", "src", "models.md"))
const START_MARKER = "<!-- KEEMENA_MODELS_START -->"
const END_MARKER = "<!-- KEEMENA_MODELS_END -->"

function _regex_escape(value::AbstractString)::String
    return replace(String(value), r"([\\.^$|?*+(){}\[\]])" => s"\\\1")
end

function _escape_cell(value)::String
    text = String(value)
    text = replace(text, "\n" => " ")
    text = replace(text, "|" => "\\|")
    return text
end

function _render_inventory_table()::String
    rows = NamedTuple[]
    for key in available_models()
        info = describe_model(key)
        info.distribution == :user_local && continue

        push!(rows, (
            key = key,
            format = info.format,
            family = info.family,
            distribution = info.distribution,
            license = info.license,
            upstream_repo = info.upstream_repo,
            upstream_ref = info.upstream_ref,
            expected_files = join(info.expected_files, ", "),
            description = info.description,
        ))
    end

    sort!(rows, by=row -> (String(row.format), String(row.family), String(row.key)))

    lines = String[
        "_Generated from the registry by `tools/sync_readme_models.jl` (excluding `:user_local` entries)._",
        "",
    ]

    active_group = nothing
    for row in rows
        group = (row.format, row.family)
        if active_group !== group
            if active_group !== nothing
                push!(lines, "")
            end
            push!(lines, "### `$(row.format)` / `$(row.family)`")
            push!(lines, "")
            push!(lines, "| Key | Distribution | License | Upstream Repo | Upstream Ref | Expected Files | Description |")
            push!(lines, "| --- | --- | --- | --- | --- | --- | --- |")
            active_group = group
        end

        push!(lines, string(
            "| `:", row.key,
            "` | `", row.distribution,
            "` | `", row.license,
            "` | `", _escape_cell(row.upstream_repo),
            "` | `", _escape_cell(row.upstream_ref),
            "` | ", _escape_cell(row.expected_files),
            " | ", _escape_cell(row.description),
            " |",
        ))
    end

    return join(lines, "\n")
end

function _replace_marked_section(path::AbstractString, content::AbstractString; check::Bool)::Bool
    text = read(path, String)
    pattern = Regex("(?s)" * _regex_escape(START_MARKER) * ".*?" * _regex_escape(END_MARKER))
    occursin(pattern, text) || throw(ArgumentError("Missing docs markers in $(path): $(START_MARKER) ... $(END_MARKER)"))

    replacement = string(START_MARKER, "\n", content, "\n", END_MARKER)
    updated = replace(text, pattern => replacement)

    if check
        return updated == text
    end

    if updated != text
        write(path, updated)
        return true
    end

    return false
end

function main()::Nothing
    check_mode = any(arg -> arg in ("--check", "-c"), ARGS)
    content = _render_inventory_table()

    if check_mode
        ok_docs = _replace_marked_section(DOCS_MODELS_PATH, content; check=true)
        if ok_docs
            println("Docs model inventory is in sync.")
            return nothing
        end
        error("Model inventory is out of sync. Run: julia --project=. tools/sync_readme_models.jl")
    end

    changed_docs = _replace_marked_section(DOCS_MODELS_PATH, content; check=false)

    if changed_docs
        println("Updated model inventory section in docs.")
    else
        println("Docs model inventory already up to date.")
    end

    return nothing
end

main()
