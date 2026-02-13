@testset "Byte-level BPE training edge cases" begin
    too_small_err = try
        train_bytebpe(
            ["hello world"];
            vocab_size=258,
            min_frequency=1,
            special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
            include_full_byte_alphabet=true,
        )
        nothing
    catch ex
        ex
    end
    @test too_small_err isa ArgumentError
    @test occursin("vocab_size", lowercase(sprint(showerror, too_small_err)))
    @test occursin("too small", lowercase(sprint(showerror, too_small_err)))

    ascii_only = train_bytebpe(
        ["hello world", "ascii only corpus"];
        vocab_size=128,
        min_frequency=1,
        special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        include_full_byte_alphabet=false,
        model_name="bytebpe_ascii_only",
    )

    non_ascii_ids = encode(ascii_only, "€"; add_special_tokens=false)
    @test !isempty(non_ascii_ids)
    @test unk_id(ascii_only) in non_ascii_ids
end
