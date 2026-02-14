@testset "SentencePiece training edge cases" begin
    empty_err = try
        train_sentencepiece(String[]; vocab_size=64, model_type=:unigram)
        nothing
    catch ex
        ex
    end
    @test empty_err isa ArgumentError
    @test occursin("empty corpus", lowercase(sprint(showerror, empty_err)))

    missing_unk_err = try
        train_sentencepiece(
            ["hello world"];
            vocab_size=64,
            model_type=:bpe,
            min_frequency=1,
            special_tokens=Dict(:pad => "<pad>"),
        )
        nothing
    catch ex
        ex
    end
    @test missing_unk_err isa ArgumentError
    @test occursin(":unk", lowercase(sprint(showerror, missing_unk_err)))

    bad_model_type_err = try
        train_sentencepiece(
            ["hello world"];
            vocab_size=64,
            model_type=:wordpiece,
        )
        nothing
    catch ex
        ex
    end
    @test bad_model_type_err isa ArgumentError
    @test occursin("model_type", lowercase(sprint(showerror, bad_model_type_err)))

    too_small_vocab_err = try
        train_sentencepiece(
            ["ab"];
            vocab_size=2,
            model_type=:bpe,
            min_frequency=1,
            special_tokens=Dict(:unk => "<unk>", :pad => "<pad>"),
            whitespace_marker="▁",
        )
        nothing
    catch ex
        ex
    end
    @test too_small_vocab_err isa ArgumentError
    @test occursin("vocab_size", lowercase(sprint(showerror, too_small_vocab_err)))
    @test occursin("too small", lowercase(sprint(showerror, too_small_vocab_err)))

    empty_marker_err = try
        train_sentencepiece(
            ["hello world"];
            vocab_size=64,
            model_type=:unigram,
            whitespace_marker="",
        )
        nothing
    catch ex
        ex
    end
    @test empty_marker_err isa ArgumentError
    @test occursin("whitespace_marker", lowercase(sprint(showerror, empty_marker_err)))

    bad_unigram_iters_err = try
        train_sentencepiece(
            ["hello world"];
            vocab_size=64,
            model_type=:unigram,
            num_iters=0,
        )
        nothing
    catch ex
        ex
    end
    @test bad_unigram_iters_err isa ArgumentError
    @test occursin("num_iters", lowercase(sprint(showerror, bad_unigram_iters_err)))
end
