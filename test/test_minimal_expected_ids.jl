using TOML

const MINIMAL_EXPECTED_IDS_PATH = joinpath(@__DIR__, "golden", "minimal_expected_ids.toml")

function _load_minimal_ids_tokenizer(spec::Dict{String,Any})::AbstractSubwordTokenizer
    kind = String(spec["source_kind"])
    source = String(spec["source"])
    format = Symbol(String(spec["format"]))

    if kind == "key"
        return load_tokenizer(Symbol(source))
    elseif kind == "path"
        return load_tokenizer(joinpath(@__DIR__, source); format=format)
    end

    throw(ArgumentError("Unsupported source_kind in minimal_expected_ids.toml: $kind"))
end

@testset "Section 21 minimal expected ids" begin
    spec = TOML.parsefile(MINIMAL_EXPECTED_IDS_PATH)
    models = get(spec, "models", Any[])
    @test !isempty(models)

    for model_any in models
        model = model_any::Dict{String,Any}
        name = String(model["name"])
        tokenizer = _load_minimal_ids_tokenizer(model)
        add_special_tokens = Bool(get(model, "add_special_tokens", false))
        cases = get(model, "cases", Any[])
        @test !isempty(cases)

        @testset "$name" begin
            for case_any in cases
                case_obj = case_any::Dict{String,Any}
                text = String(case_obj["text"])
                expected_ids = Int[Int(x) for x in case_obj["expected_ids"]]
                actual_ids = encode(tokenizer, text; add_special_tokens=add_special_tokens)
                @test actual_ids == expected_ids
            end
        end
    end
end
