function _extended_assert_smoke(
    tokenizer,
    long_text::String;
    require_string_boundaries::Bool,
    require_exact_decode::Bool=true,
)::Nothing
    tokenization_text = tokenization_view(tokenizer, long_text)
    ids = encode(tokenizer, tokenization_text; add_special_tokens=false)
    @test !isempty(ids)

    decoded = decode(tokenizer, ids)
    if require_exact_decode
        @test decoded == tokenization_text
    else
        @test !isempty(decoded)
    end

    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        add_special_tokens=true,
        return_offsets=true,
        return_masks=true,
    )

    @test result.offsets !== nothing
    @test_nowarn assert_offsets_contract(
        tokenization_text,
        result.offsets;
        require_string_boundaries=require_string_boundaries,
    )
    @test offsets_are_nonoverlapping(
        result.offsets;
        ignore_sentinel=true,
        ignore_empty=true,
    )
    return nothing
end

@testset "Extended large-corpus training smoke" begin
    corpus = synthetic_corpus(2400)
    long_text = synthetic_long_text(corpus)

    cases = [
        (
            name="BPE",
            train=() -> train_bpe_result(
                corpus;
                vocab_size=384,
                min_frequency=2,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                model_name="extended_bpe_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="ByteBPE",
            train=() -> train_bytebpe_result(
                corpus;
                vocab_size=448,
                min_frequency=2,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                include_full_byte_alphabet=true,
                model_name="extended_bytebpe_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=false,
        ),
        (
            name="WordPiece",
            train=() -> train_wordpiece_result(
                corpus;
                vocab_size=384,
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
                model_name="extended_wordpiece_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="Unigram",
            train=() -> train_unigram_result(
                corpus;
                vocab_size=320,
                seed_size=2_000,
                num_iters=2,
                max_subword_length=12,
                prune_fraction=0.2,
                special_tokens=Dict(:unk => "<unk>", :pad => "<pad>", :bos => "<s>", :eos => "</s>"),
                whitespace_marker="▁",
                model_name="extended_unigram_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="SentencePiece",
            train=() -> train_sentencepiece_result(
                corpus;
                vocab_size=320,
                model_type=:unigram,
                special_tokens=Dict(:unk => "<unk>", :pad => "<pad>", :bos => "<s>", :eos => "</s>"),
                whitespace_marker="▁",
                seed_size=2_000,
                num_iters=2,
                max_subword_length=12,
                prune_fraction=0.2,
                model_name="extended_sentencepiece_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="HF GPT-2 ByteBPE",
            train=() -> train_hf_gpt2_bytebpe_result(
                corpus;
                vocab_size=448,
                min_frequency=2,
                special_tokens=Dict(:unk => "<|endoftext|>"),
                add_prefix_space=false,
                trim_offsets=true,
                use_regex=true,
                export_unk_token_null=true,
                model_name="extended_hf_gpt2_smoke",
                version=v"0.3.0",
            ),
            require_string_boundaries=false,
            require_exact_decode=false,
        ),
    ]

    for case in cases
        @testset "$(case.name)" begin
            result = case.train()
            _extended_assert_smoke(
                result.tokenizer,
                long_text;
                require_string_boundaries=case.require_string_boundaries,
                require_exact_decode=get(case, :require_exact_decode, true),
            )
        end
    end
end
