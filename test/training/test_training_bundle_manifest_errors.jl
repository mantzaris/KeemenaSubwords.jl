using JSON3

function _write_manifest_payload(
    bundle_dir::String,
    payload::Dict{String,Any},
)::Nothing
    open(joinpath(bundle_dir, "keemena_training_manifest.json"), "w") do io
        JSON3.write(io, payload)
    end
    return nothing
end

function _base_manifest_payload()::Dict{String,Any}
    return Dict(
        "schema_version" => 1,
        "trainer" => "wordpiece",
        "load_format" => "wordpiece",
        "export_format" => "wordpiece_vocab",
        "files" => Dict("vocab_txt" => "vocab.txt"),
        "load_kwargs" => Dict(
            "continuation_prefix" => "##",
            "unk_token" => "[UNK]",
            "max_input_chars_per_word" => 100,
            "special_tokens" => Dict("unk" => "[UNK]"),
        ),
        "training_config" => Dict{String,Any}(),
        "metadata" => Dict{String,Any}(),
        "warnings" => String[],
    )
end

@testset "Training bundle manifest errors" begin
    @testset "Schema version mismatch" begin
        bundle_dir = mktempdir()
        payload = _base_manifest_payload()
        payload["schema_version"] = 999
        _write_manifest_payload(bundle_dir, payload)

        err = try
            read_training_manifest(bundle_dir)
            nothing
        catch inner
            inner
        end
        @test err isa ArgumentError
        @test occursin(
            "Unsupported training manifest schema_version",
            sprint(showerror, err),
        )
    end

    @testset "Missing required field" begin
        bundle_dir = mktempdir()
        payload = _base_manifest_payload()
        delete!(payload, "files")
        _write_manifest_payload(bundle_dir, payload)

        err = try
            read_training_manifest(bundle_dir)
            nothing
        catch inner
            inner
        end
        @test err isa ArgumentError
        @test occursin("Missing required manifest field", sprint(showerror, err))
    end

    @testset "Missing file referenced by manifest" begin
        bundle_dir = mktempdir()
        payload = _base_manifest_payload()
        _write_manifest_payload(bundle_dir, payload)

        err = try
            load_training_bundle(bundle_dir)
            nothing
        catch inner
            inner
        end
        @test err isa ArgumentError
        @test occursin("points to missing file", sprint(showerror, err))
        @test occursin("vocab_txt", sprint(showerror, err))
    end
end
