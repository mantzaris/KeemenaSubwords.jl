@testset "Unigram training edge cases" begin
    empty_err = try
        train_unigram(String[]; vocab_size=64, seed_size=200, num_iters=2)
        nothing
    catch ex
        ex
    end
    @test empty_err isa ArgumentError
    @test occursin("empty corpus", lowercase(sprint(showerror, empty_err)))

    small_vocab_err = try
        train_unigram(
            ["ab"];
            vocab_size=3,
            seed_size=200,
            num_iters=2,
            special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>"),
        )
        nothing
    catch ex
        ex
    end
    @test small_vocab_err isa ArgumentError
    @test occursin("too small", lowercase(sprint(showerror, small_vocab_err)))

    bad_seed_err = try
        train_unigram(
            ["hello world"];
            vocab_size=64,
            seed_size=0,
            num_iters=2,
        )
        nothing
    catch ex
        ex
    end
    @test bad_seed_err isa ArgumentError
    @test occursin("seed_size", lowercase(sprint(showerror, bad_seed_err)))

    bad_iters_err = try
        train_unigram(
            ["hello world"];
            vocab_size=64,
            seed_size=200,
            num_iters=0,
        )
        nothing
    catch ex
        ex
    end
    @test bad_iters_err isa ArgumentError
    @test occursin("num_iters", lowercase(sprint(showerror, bad_iters_err)))

    bad_maxlen_err = try
        train_unigram(
            ["hello world"];
            vocab_size=64,
            seed_size=200,
            num_iters=2,
            max_subword_length=0,
        )
        nothing
    catch ex
        ex
    end
    @test bad_maxlen_err isa ArgumentError
    @test occursin("max_subword_length", lowercase(sprint(showerror, bad_maxlen_err)))
end

@testset "Unigram default whitespace marker roundtrip" begin
    corpus = [
        "hello world",
        "hello tokenizers",
        "world training",
    ]

    training = train_unigram_result(
        corpus;
        vocab_size=64,
        seed_size=200,
        num_iters=2,
        max_subword_length=6,
        prune_fraction=0.2,
        special_tokens=Dict(
            :unk => "<UNK>",
            :pad => "<PAD>",
        ),
    )
    tokenizer = training.tokenizer

    @test training.config.whitespace_marker == "▁"
    @test tokenizer.whitespace_marker == "▁"

    text = "hello world"
    ids = encode(tokenizer, text; add_special_tokens=false)
    @test decode(tokenizer, ids) == text
end
