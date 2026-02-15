using JSON3

@testset "Section 26a HF StripAccents normalizer regression" begin
    tokenizer_payload = Dict(
        "version" => "1.0",
        "truncation" => nothing,
        "padding" => nothing,
        "normalizer" => Dict("type" => "StripAccents"),
        "pre_tokenizer" => Dict("type" => "WhitespaceSplit"),
        "post_processor" => nothing,
        "decoder" => Dict("type" => "WordPiece", "prefix" => "##"),
        "model" => Dict(
            "type" => "WordPiece",
            "unk_token" => "[UNK]",
            "continuing_subword_prefix" => "##",
            "max_input_chars_per_word" => 32,
            "vocab" => Dict(
                "[UNK]" => 0,
                "Cafe" => 1,
            ),
        ),
        "added_tokens" => Any[
            Dict(
                "id" => 0,
                "content" => "[UNK]",
                "single_word" => false,
                "lstrip" => false,
                "rstrip" => false,
                "normalized" => false,
                "special" => true,
            ),
        ],
    )

    outdir = mktempdir()
    tokenizer_path = joinpath(outdir, "tokenizer.json")
    open(tokenizer_path, "w") do io
        JSON3.write(io, tokenizer_payload)
    end

    tokenizer = load_hf_tokenizer_json(tokenizer_path)
    @test tokenizer isa HuggingFaceJSONTokenizer
    @test normalize(tokenizer, "Café") == "Cafe"

    ids = encode(tokenizer, "Café"; add_special_tokens=false)
    @test ids == [token_to_id(tokenizer, "Cafe")]
    @test decode(tokenizer, ids) == "Cafe"

    tokenization_text = tokenization_view(tokenizer, "Café")
    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=false,
    )

    @test result.offsets !== nothing
    @test_nowarn assert_offsets_contract(
        tokenization_text,
        result.offsets;
        require_string_boundaries=true,
    )
    @test offsets_are_nonoverlapping(
        result.offsets;
        ignore_sentinel=true,
        ignore_empty=true,
    )
    @test length(result.tokens) == 1
    @test try_span_substring(tokenization_text, result.offsets[1]) == "Cafe"
end
