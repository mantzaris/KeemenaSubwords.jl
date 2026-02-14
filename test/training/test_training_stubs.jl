@testset "Training stubs" begin
    corpus = ["hello world", "keemena subwords"]

    wordpiece = train_wordpiece(
        corpus;
        vocab_size=64,
        min_frequency=1,
    )
    @test wordpiece isa WordPieceTokenizer

    sentencepiece_error = try
        train_sentencepiece(corpus; vocab_size=100)
        nothing
    catch ex
        ex
    end
    @test sentencepiece_error isa ArgumentError
    @test occursin("train_sentencepiece", sprint(showerror, sentencepiece_error))
    @test occursin("not implemented yet", lowercase(sprint(showerror, sentencepiece_error)))
end
