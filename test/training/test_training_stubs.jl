@testset "Training stubs" begin
    corpus = ["hello world", "keemena subwords"]

    wordpiece_error = try
        train_wordpiece(corpus; vocab_size=100)
        nothing
    catch ex
        ex
    end
    @test wordpiece_error isa ArgumentError
    @test occursin("train_wordpiece", sprint(showerror, wordpiece_error))
    @test occursin("not implemented yet", lowercase(sprint(showerror, wordpiece_error)))

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
