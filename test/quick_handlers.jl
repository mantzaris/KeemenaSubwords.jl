@testset "Section 26 quick orchestration handlers" begin
    @testset "quick_tokenize returns stable keys and roundtrip decode" begin
        tokenizer = load_tokenizer(:core_bpe_en)
        input_text = "hello world"

        output = quick_tokenize(tokenizer, input_text)

        expected_keys = (
            :token_pieces,
            :token_ids,
            :decoded_text,
            :tokenization_text,
            :offsets,
            :attention_mask,
            :token_type_ids,
            :special_tokens_mask,
            :metadata,
        )
        @test keys(output) == expected_keys

        @test output.tokenization_text == tokenization_view(tokenizer, input_text)
        @test output.token_ids == encode(
            tokenizer,
            output.tokenization_text;
            add_special_tokens=true,
        )
        @test output.decoded_text == decode(tokenizer, output.token_ids)
        @test output.offsets !== nothing
        @test output.attention_mask !== nothing
        @test output.token_type_ids !== nothing
        @test output.special_tokens_mask !== nothing
        @test length(output.token_pieces) == length(output.token_ids)
        @test output.metadata.offsets_reference == :input_text

        source_output = quick_tokenize(:core_bpe_en, input_text)
        @test source_output.token_ids == output.token_ids
    end

    @testset "quick_causal_lm_batch collates and shifts labels" begin
        tokenizer = load_tokenizer(:core_wordpiece_en)
        input_texts = ["hello world", "hello", "world hello world"]

        output = quick_causal_lm_batch(
            tokenizer,
            input_texts;
            return_offsets=false,
            pad_to_multiple_of=4,
            ignore_index=-100,
        )

        @test size(output.ids) == size(output.attention_mask)
        @test size(output.ids) == size(output.labels)
        @test size(output.ids, 2) == length(input_texts)
        @test output.pad_token_id == pad_id(tokenizer)
        @test output.sequence_lengths == [sum(output.attention_mask[:, i]) for i in axes(output.attention_mask, 2)]
        @test size(output.ids, 1) % 4 == 0

        for column_index in axes(output.ids, 2)
            valid_positions = findall(output.attention_mask[:, column_index] .== 1)
            for i in 1:(length(valid_positions) - 1)
                position = valid_positions[i]
                next_position = valid_positions[i + 1]
                @test output.labels[position, column_index] == output.ids[next_position, column_index]
            end
            if !isempty(valid_positions)
                @test output.labels[last(valid_positions), column_index] == -100
            end
            padded_positions = findall(output.attention_mask[:, column_index] .== 0)
            for position in padded_positions
                @test output.labels[position, column_index] == -100
            end
        end
    end

    @testset "causal_lm_labels zero_based conversion keeps ignore_index" begin
        ids = [
            10 20
            11 21
            12 99
            13 99
        ]
        attention_mask = [
            1 1
            1 1
            1 0
            0 0
        ]

        labels_one_based = causal_lm_labels(ids, attention_mask; ignore_index=-100, zero_based=false)
        labels_zero_based = causal_lm_labels(ids, attention_mask; ignore_index=-100, zero_based=true)

        @test labels_one_based[1, 1] == 11
        @test labels_one_based[2, 1] == 12
        @test labels_one_based[3, 1] == -100
        @test labels_one_based[4, 1] == -100
        @test labels_one_based[1, 2] == 21
        @test labels_one_based[2, 2] == -100

        @test labels_zero_based[1, 1] == 10
        @test labels_zero_based[2, 1] == 11
        @test labels_zero_based[1, 2] == 20
        @test labels_zero_based[2, 2] == -100
        @test count(==(-100), labels_zero_based) == count(==(-100), labels_one_based)
    end

    @testset "quick_train_bundle wordpiece roundtrip" begin
        corpus = [
            "hello world",
            "hello tokenizer",
            "world tokenizer",
            "tokenizer world",
        ]

        output = quick_train_bundle(
            :wordpiece,
            corpus;
            vocab_size=48,
            min_frequency=1,
            model_name="quick_wordpiece_test",
            sanity_text="hello world",
            overwrite=true,
        )

        @test isdir(output.bundle_directory)
        @test "keemena_training_manifest.json" in output.bundle_files
        @test output.tokenizer isa AbstractSubwordTokenizer
        @test !isempty(output.sanity_encoded_ids)
        @test output.sanity_decoded_text isa String

        reload_ids = encode(output.tokenizer, "hello world"; add_special_tokens=false)
        reload_decoded = decode(output.tokenizer, reload_ids)
        @test reload_ids == output.sanity_encoded_ids
        @test reload_decoded == output.sanity_decoded_text

        default_output = quick_train_bundle(
            corpus;
            vocab_size=48,
            min_frequency=1,
            model_name="quick_wordpiece_default_overload_test",
            sanity_text="hello world",
            overwrite=true,
        )
        @test default_output.training_summary.trainer == :wordpiece
        @test !isempty(default_output.bundle_files)
        @test !isempty(default_output.sanity_encoded_ids)
    end
end
