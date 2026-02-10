using JSON3

const GOLDEN_DIR = joinpath(@__DIR__, "golden")

function _golden_spec_files()::Vector{String}
    files = String[]
    for name in sort(readdir(GOLDEN_DIR))
        endswith(name, ".json") || continue
        push!(files, joinpath(GOLDEN_DIR, name))
    end
    return files
end

function _load_golden_tokenizer(spec)::AbstractSubwordTokenizer
    source = spec["source"]
    kind = String(source["kind"])
    if kind == "key"
        return load_tokenizer(Symbol(String(source["value"])))
    elseif kind == "path"
        rel = String(source["value"])
        local_path = joinpath(@__DIR__, rel)
        fmt = haskey(source, "format") ? Symbol(String(source["format"])) : Symbol(String(spec["format"]))
        return load_tokenizer(local_path; format=fmt)
    end

    throw(ArgumentError("Unsupported golden source kind: $kind"))
end

function _bool_setting(spec, case_obj, key::String, default::Bool)::Bool
    if haskey(case_obj, key)
        return Bool(case_obj[key])
    elseif haskey(spec, "settings")
        settings = spec["settings"]
        if haskey(settings, key)
            return Bool(settings[key])
        end
    end
    return default
end

function _assert_ids_equal(expected::Vector{Int}, actual::Vector{Int}, context::String)::Nothing
    if expected == actual
        return nothing
    end

    limit = min(length(expected), length(actual))
    mismatch = nothing
    for i in 1:limit
        if expected[i] != actual[i]
            mismatch = i
            break
        end
    end

    if mismatch === nothing
        error(
            "[$context] id length mismatch: expected $(length(expected)), got $(length(actual)); " *
            "expected=$(expected), actual=$(actual)",
        )
    end

    i = mismatch::Int
    lo = max(1, i - 2)
    hi = min(limit, i + 2)
    error(
        "[$context] id mismatch at index $i: expected=$(expected[i]) got=$(actual[i]); " *
        "window expected=$(expected[lo:hi]) actual=$(actual[lo:hi])",
    )
end

@testset "Section 17 golden conformance" begin
    files = _golden_spec_files()
    @test !isempty(files)

    for spec_path in files
        spec = JSON3.read(read(spec_path, String))
        name = String(spec["name"])
        tok = _load_golden_tokenizer(spec)

        @testset "$name" begin
            for case_obj in spec["cases"]
                text = String(case_obj["text"])
                expected_ids = Int[Int(x) for x in case_obj["expected_ids"]]
                add_special = _bool_setting(spec, case_obj, "add_special_tokens", false)
                actual_ids = encode(tok, text; add_special_tokens=add_special)
                _assert_ids_equal(expected_ids, actual_ids, "$name / $(repr(text))")

                if haskey(case_obj, "expected_tokens")
                    expected_tokens = String[String(x) for x in case_obj["expected_tokens"]]
                    actual_tokens = tokenize(tok, text)
                    @test actual_tokens == expected_tokens
                end

                if haskey(case_obj, "expected_decoded")
                    expected_decoded = String(case_obj["expected_decoded"])
                    @test decode(tok, actual_ids) == expected_decoded
                end
            end
        end
    end
end
