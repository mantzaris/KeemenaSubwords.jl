function _extended_assert_bundle_roundtrip(
    original,
    reloaded,
    samples::Vector{String};
    require_string_boundaries::Bool,
)::Nothing
    @test typeof(reloaded) == typeof(original)

    for text in samples
        @test tokenize(reloaded, text) == tokenize(original, text)

        for add_special_tokens in (false, true)
            ids_original = encode(original, text; add_special_tokens=add_special_tokens)
            ids_reloaded = encode(reloaded, text; add_special_tokens=add_special_tokens)
            @test ids_reloaded == ids_original
            @test decode(reloaded, ids_reloaded) == decode(original, ids_original)
        end
    end

    long_text = synthetic_long_text(samples)
    tokenization_text = tokenization_view(reloaded, long_text)
    result = encode_result(
        reloaded,
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

@testset "Extended large-corpus training bundle roundtrip" begin
    corpus = synthetic_corpus(1800)
    sample_texts = [
        synthetic_long_text(corpus[1:12]),
        synthetic_long_text(corpus[200:212]),
        synthetic_long_text(corpus[900:912]),
    ]

    cases = [
        (
            name="BPE",
            train=() -> train_bpe_result(
                corpus;
                vocab_size=320,
                min_frequency=2,
                special_tokens=Dict(:unk => "<UNK>", :pad => "<PAD>", :bos => "<BOS>", :eos => "<EOS>"),
                end_of_word_marker="</w>",
                model_name="extended_bundle_bpe",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="WordPiece",
            train=() -> train_wordpiece_result(
                corpus;
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
                model_name="extended_bundle_wordpiece",
                version=v"0.3.0",
            ),
            require_string_boundaries=true,
        ),
        (
            name="HF GPT-2 ByteBPE",
            train=() -> train_hf_gpt2_bytebpe_result(
                corpus;
                vocab_size=384,
                min_frequency=2,
                special_tokens=Dict(:unk => "<|endoftext|>"),
                add_prefix_space=false,
                trim_offsets=true,
                use_regex=true,
                export_unk_token_null=true,
                model_name="extended_bundle_hf_gpt2",
                version=v"0.3.0",
            ),
            require_string_boundaries=false,
        ),
    ]

    for case in cases
        @testset "$(case.name)" begin
            result = case.train()
            bundle_root = mktempdir()
            bundle_dir = joinpath(bundle_root, "bundle")
            save_training_bundle(result, bundle_dir)

            manifest = read_training_manifest(bundle_dir)
            @test manifest.schema_version == 1
            @test isfile(joinpath(bundle_dir, "keemena_training_manifest.json"))

            reloaded = load_training_bundle(bundle_dir)
            _extended_assert_bundle_roundtrip(
                result.tokenizer,
                reloaded,
                sample_texts;
                require_string_boundaries=case.require_string_boundaries,
            )
        end
    end
end
