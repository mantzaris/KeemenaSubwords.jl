@testset "Extended training order invariance" begin
    corpus = synthetic_corpus(1200)
    corpus_reversed = reverse(corpus)
    samples = [
        synthetic_long_text(corpus[1:8]),
        synthetic_long_text(corpus[120:126]),
        "café naïve 你好 世界 42 !",
    ]

    cases = [
        (
            name="BPE",
            train=(data) -> train_bpe_result(
                data;
                vocab_size=320,
                min_frequency=2,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                model_name="extended_order_bpe",
                version=v"0.3.0",
            ),
            export_format=:bpe,
            files=["vocab.txt", "merges.txt"],
        ),
        (
            name="ByteBPE",
            train=(data) -> train_bytebpe_result(
                data;
                vocab_size=384,
                min_frequency=2,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                include_full_byte_alphabet=true,
                model_name="extended_order_bytebpe",
                version=v"0.3.0",
            ),
            export_format=:bpe,
            files=["vocab.txt", "merges.txt"],
        ),
        (
            name="WordPiece",
            train=(data) -> train_wordpiece_result(
                data;
                vocab_size=320,
                min_frequency=2,
                special_tokens=Dict(
                    :unk => "[UNK]",
                    :pad => "[PAD]",
                    :cls => "[CLS]",
                    :sep => "[SEP]",
                    :mask => "[MASK]",
                ),
                continuation_prefix="##",
                max_input_chars_per_word=100,
                model_name="extended_order_wordpiece",
                version=v"0.3.0",
            ),
            export_format=:wordpiece_vocab,
            files=["vocab.txt"],
        ),
        (
            name="Unigram",
            train=(data) -> train_unigram_result(
                data;
                vocab_size=256,
                seed_size=1_500,
                num_iters=2,
                max_subword_length=10,
                prune_fraction=0.2,
                special_tokens=Dict(:unk => "<unk>", :pad => "<pad>", :bos => "<s>", :eos => "</s>"),
                whitespace_marker="▁",
                model_name="extended_order_unigram",
                version=v"0.3.0",
            ),
            export_format=:unigram_tsv,
            files=["unigram.tsv"],
        ),
    ]

    for case in cases
        @testset "$(case.name)" begin
            result_a = case.train(corpus)
            result_b = case.train(corpus_reversed)

            dir_a = mktempdir()
            dir_b = mktempdir()
            export_tokenizer(result_a.tokenizer, dir_a; format=case.export_format)
            export_tokenizer(result_b.tokenizer, dir_b; format=case.export_format)

            for filename in case.files
                bytes_a = read(joinpath(dir_a, filename))
                bytes_b = read(joinpath(dir_b, filename))
                @test bytes_a == bytes_b
            end

            for text in samples
                @test tokenize(result_b.tokenizer, text) == tokenize(result_a.tokenizer, text)
                for add_special_tokens in (false, true)
                    ids_a = encode(result_a.tokenizer, text; add_special_tokens=add_special_tokens)
                    ids_b = encode(result_b.tokenizer, text; add_special_tokens=add_special_tokens)
                    @test ids_b == ids_a
                end
            end
        end
    end
end
