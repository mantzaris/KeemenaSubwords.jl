using JSON3

function _write_manifest_with_vocab_path(
    bundle_dir::String,
    vocab_path::String,
)::Nothing
    payload = Dict(
        "schema_version" => 1,
        "trainer" => "wordpiece",
        "load_format" => "wordpiece",
        "export_format" => "wordpiece_vocab",
        "files" => Dict("vocab_txt" => vocab_path),
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

    open(joinpath(bundle_dir, "keemena_training_manifest.json"), "w") do io
        JSON3.write(io, payload)
    end
    return nothing
end

@testset "Extended training manifest path hardening" begin
    @testset "Reject traversal path" begin
        bundle_dir = mktempdir()
        _write_manifest_with_vocab_path(bundle_dir, "../outside.txt")

        err = try
            load_training_bundle(bundle_dir)
            nothing
        catch inner
            inner
        end
        @test err isa ArgumentError
        @test occursin("vocab_txt", sprint(showerror, err))
        @test occursin("escapes bundle directory", sprint(showerror, err))
    end

    @testset "Reject absolute path" begin
        bundle_dir = mktempdir()
        absolute_target = abspath(joinpath(mktempdir(), "outside_vocab.txt"))
        write(absolute_target, "[UNK]\n")
        _write_manifest_with_vocab_path(bundle_dir, absolute_target)

        err = try
            load_training_bundle(bundle_dir)
            nothing
        catch inner
            inner
        end
        @test err isa ArgumentError
        @test occursin("vocab_txt", sprint(showerror, err))
        @test occursin("must be relative", sprint(showerror, err))
    end
end
