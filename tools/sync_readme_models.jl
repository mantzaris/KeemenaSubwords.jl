#!/usr/bin/env julia

using KeemenaSubwords

const README_PATH = normpath(joinpath(@__DIR__, "..", "README.md"))
const DOCS_MODELS_PATH = normpath(joinpath(@__DIR__, "..", "docs", "src", "models.md"))

const README_FEATURED_START = "## Featured Models"
const README_FEATURED_END = "## Documentation"
const DOCS_MODELS_START = "The inventory below is generated from the same registry used by `available_models()` and `describe_model(...)`."
const DOCS_MODELS_START_LEGACY = "The table below is generated from the same registry used by `available_models()` and `describe_model(...)`."
const DOCS_MODELS_END = "`describe_model(key)` includes provenance metadata such as `license`, `family`, `distribution`, `upstream_repo`, `upstream_ref`, and `upstream_files`."

const FEATURED_MODEL_PRIORITY = [
    :tiktoken_cl100k_base,
    :tiktoken_o200k_base,
    :openai_gpt2_bpe,
    :mistral_v3_sentencepiece,
    :phi2_bpe,
    :qwen2_5_bpe,
    :roberta_base_bpe,
    :xlm_roberta_base_sentencepiece_bpe,
    :bert_base_multilingual_cased_wordpiece,
    :bert_base_uncased_wordpiece,
    :llama3_8b_tokenizer,
    :core_bpe_en,
]

function _regex_escape(value::AbstractString)::String
    return replace(String(value), r"([\\.^$|?*+(){}\[\]])" => s"\\\1")
end

function _single_line(value)::String
    return replace(String(value), "\n" => " ")
end

function _collect_rows()
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
    return rows
end

function _render_featured_list(rows)::String
    row_by_key = Dict{Symbol,NamedTuple}(row.key => row for row in rows)

    featured_keys = Symbol[]
    for key in FEATURED_MODEL_PRIORITY
        haskey(row_by_key, key) || continue
        push!(featured_keys, key)
        length(featured_keys) >= 10 && break
    end

    if length(featured_keys) < 8
        for row in rows
            row.key in featured_keys && continue
            push!(featured_keys, row.key)
            length(featured_keys) >= 8 && break
        end
    end

    grouped = Dict{Symbol,Vector{Symbol}}()
    format_order = Symbol[]
    for key in featured_keys
        format = row_by_key[key].format
        if !haskey(grouped, format)
            grouped[format] = Symbol[]
            push!(format_order, format)
        end
        push!(grouped[format], key)
    end

    lines = String["_Generated from registry metadata via `tools/sync_readme_models.jl`._", ""]

    for format in format_order
        keys = grouped[format]
        rendered = join(["`:$key`" for key in keys], ", ")
        push!(lines, "- **`$(format)`**: $(rendered)")
    end

    return join(lines, "\n")
end

function _render_inventory_list(rows)::String
    lines = String["_Generated from the registry by `tools/sync_readme_models.jl` (excluding `:user_local` entries)._", ""]

    active_group = nothing
    for row in rows
        group = (row.format, row.family)
        if active_group !== group
            if active_group !== nothing
                push!(lines, "")
            end
            push!(lines, "### `$(row.format)` / `$(row.family)`")
            push!(lines, "")
            active_group = group
        end

        push!(lines, "- `:$(row.key)`")
        push!(lines, "  - Distribution: `$(row.distribution)`")
        push!(lines, "  - License: `$(row.license)`")
        push!(lines, string("  - Upstream: `", _single_line(row.upstream_repo), "` @ `", _single_line(row.upstream_ref), "`"))
        push!(lines, string("  - Expected files: ", _single_line(row.expected_files)))
        push!(lines, string("  - Description: ", _single_line(row.description)))
    end

    return join(lines, "\n")
end

function _replace_marked_section(
    path::AbstractString,
    start_marker::AbstractString,
    end_marker::AbstractString,
    content::AbstractString;
    check::Bool,
    search_start_markers::Vector{String}=String[String(start_marker)],
)::Bool
    text = read(path, String)
    escaped_start_markers = _regex_escape.(search_start_markers)
    marker_pattern = "(?:" * join(escaped_start_markers, "|") * ")"
    pattern = Regex("(?s)" * marker_pattern * ".*?" * _regex_escape(end_marker))
    occursin(pattern, text) || throw(ArgumentError("Missing marker block in $(path): $(start_marker) ... $(end_marker)"))

    replacement = string(start_marker, "\n\n", content, "\n\n", end_marker)
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
    rows = _collect_rows()
    featured_content = _render_featured_list(rows)
    inventory_content = _render_inventory_list(rows)

    if check_mode
        ok_readme = _replace_marked_section(
            README_PATH,
            README_FEATURED_START,
            README_FEATURED_END,
            featured_content;
            check=true,
        )
        ok_docs = _replace_marked_section(
            DOCS_MODELS_PATH,
            DOCS_MODELS_START,
            DOCS_MODELS_END,
            inventory_content;
            check=true,
            search_start_markers=String[DOCS_MODELS_START, DOCS_MODELS_START_LEGACY],
        )

        if ok_readme && ok_docs
            println("README featured models and docs model inventory are in sync.")
            return nothing
        end

        error("Generated docs/README blocks are out of sync. Run: julia --project=. tools/sync_readme_models.jl")
    end

    changed_readme = _replace_marked_section(
        README_PATH,
        README_FEATURED_START,
        README_FEATURED_END,
        featured_content;
        check=false,
    )
    changed_docs = _replace_marked_section(
        DOCS_MODELS_PATH,
        DOCS_MODELS_START,
        DOCS_MODELS_END,
        inventory_content;
        check=false,
        search_start_markers=String[DOCS_MODELS_START, DOCS_MODELS_START_LEGACY],
    )

    if changed_readme || changed_docs
        println("Updated generated model blocks in README/docs.")
    else
        println("Generated model blocks already up to date.")
    end

    return nothing
end

main()
