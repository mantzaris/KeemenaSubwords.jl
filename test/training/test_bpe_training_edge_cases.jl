@testset "BPE training edge cases" begin
    empty_err = try
        train_bpe(String[]; vocab_size=32, min_frequency=1)
        nothing
    catch ex
        ex
    end
    @test empty_err isa ArgumentError
    @test occursin("empty corpus", lowercase(sprint(showerror, empty_err)))

    missing_unk_err = try
        train_bpe(
            ["hello world"];
            vocab_size=32,
            min_frequency=1,
            special_tokens=Dict(:pad => "[PAD]"),
        )
        nothing
    catch ex
        ex
    end
    @test missing_unk_err isa ArgumentError
    @test occursin(":unk", lowercase(sprint(showerror, missing_unk_err)))

    too_small_err = try
        train_bpe(
            ["ab"];
            vocab_size=4,
            min_frequency=1,
            special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        )
        nothing
    catch ex
        ex
    end
    @test too_small_err isa ArgumentError
    @test occursin("vocab_size", lowercase(sprint(showerror, too_small_err)))
    @test occursin("too small", lowercase(sprint(showerror, too_small_err)))

    high_minfreq = train_bpe_result(
        ["hello world", "world hello"];
        vocab_size=64,
        min_frequency=10_000,
        special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        end_of_word_marker="</w>",
        model_name="high_minfreq_bpe",
    )
    @test isempty(high_minfreq.artifacts.merge_pairs)
    @test isempty(high_minfreq.artifacts.pair_ranks)

    sample = "hello world"
    encoded = encode(high_minfreq.tokenizer, sample)
    @test !isempty(encoded)
    @test decode(high_minfreq.tokenizer, encoded) == sample
end
