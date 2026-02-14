@testset "WordPiece training edge cases" begin
    empty_err = try
        train_wordpiece(String[]; vocab_size=64, min_frequency=1)
        nothing
    catch ex
        ex
    end
    @test empty_err isa ArgumentError
    @test occursin("empty corpus", lowercase(sprint(showerror, empty_err)))

    too_small_err = try
        train_wordpiece(
            ["ab"];
            vocab_size=5,
            min_frequency=1,
            special_tokens=Dict(:unk => "[UNK]", :pad => "[PAD]"),
        )
        nothing
    catch ex
        ex
    end
    @test too_small_err isa ArgumentError
    @test occursin("vocab_size", lowercase(sprint(showerror, too_small_err)))
    @test occursin("too small", lowercase(sprint(showerror, too_small_err)))

    empty_prefix_err = try
        train_wordpiece(
            ["hello world"];
            vocab_size=64,
            min_frequency=1,
            continuation_prefix="",
        )
        nothing
    catch ex
        ex
    end
    @test empty_prefix_err isa ArgumentError
    @test occursin("continuation_prefix", lowercase(sprint(showerror, empty_prefix_err)))

    missing_unk_err = try
        train_wordpiece(
            ["hello world"];
            vocab_size=64,
            min_frequency=1,
            special_tokens=Dict(:pad => "[PAD]", :cls => "[CLS]"),
        )
        nothing
    catch ex
        ex
    end
    @test missing_unk_err isa ArgumentError
    @test occursin(":unk", lowercase(sprint(showerror, missing_unk_err)))
end

@testset "WordPiece max_input_chars_per_word behavior" begin
    training = train_wordpiece_result(
        ["mini alphabet", "tiny mini"];
        vocab_size=96,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "[UNK]",
            :pad => "[PAD]",
            :cls => "[CLS]",
            :sep => "[SEP]",
        ),
        continuation_prefix="##",
        max_input_chars_per_word=4,
        model_name="wordpiece_max_chars",
    )
    tokenizer = training.tokenizer

    @test !haskey(training.artifacts.word_counts, "alphabet")
    @test encode(tokenizer, "alphabet"; add_special_tokens=false) == [unk_id(tokenizer)]
end
