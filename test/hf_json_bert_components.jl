using JSON3

function _bert_json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

@testset "Section 27 HF BertNormalizer + BertPreTokenizer" begin
    tokenizer = load_tokenizer(
        fixture("hf_json_bert_wordpiece", "tokenizer.json");
        format=:hf_tokenizer_json,
    )

    @test tokenizer isa HuggingFaceJSONTokenizer
    @test tokenizer.normalizer isa KeemenaSubwords.HFBertNormalizer
    @test tokenizer.pretokenizer isa KeemenaSubwords.HFBertPreTokenizer

    @test encode(tokenizer, "Hello"; add_special_tokens=false) == [token_to_id(tokenizer, "hello")]
    @test encode(tokenizer, "Café"; add_special_tokens=false) == [token_to_id(tokenizer, "cafe")]
    @test encode(tokenizer, "你好"; add_special_tokens=false) == [
        token_to_id(tokenizer, "你"),
        token_to_id(tokenizer, "好"),
    ]
    @test tokenize(tokenizer, "Hello,world!") == ["hello", ",", "world", "!"]

    tokenization_text = tokenization_view(tokenizer, "Hello,world!")
    @test tokenization_text == "hello,world!"

    result = encode_result(
        tokenizer,
        tokenization_text;
        assume_normalized=true,
        return_offsets=true,
        return_masks=true,
        add_special_tokens=true,
    )

    @test result.offsets !== nothing
    @test result.special_tokens_mask !== nothing
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

    cls_idx = findfirst(==("[CLS]"), result.tokens)
    sep_idx = findlast(==("[SEP]"), result.tokens)
    comma_idx = findfirst(==(","), result.tokens)
    bang_idx = findfirst(==("!"), result.tokens)

    @test cls_idx !== nothing
    @test sep_idx !== nothing
    @test comma_idx !== nothing
    @test bang_idx !== nothing

    @test result.offsets[cls_idx] == offsets_sentinel()
    @test result.offsets[sep_idx] == offsets_sentinel()
    @test result.special_tokens_mask[cls_idx] == 1
    @test result.special_tokens_mask[sep_idx] == 1

    @test try_span_substring(tokenization_text, result.offsets[comma_idx]) == ","
    @test try_span_substring(tokenization_text, result.offsets[bang_idx]) == "!"

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)
    exported_path = joinpath(outdir, "tokenizer.json")
    @test isfile(exported_path)

    exported_root = JSON3.read(read(exported_path, String))
    exported_normalizer = _bert_json_get(exported_root, "normalizer")
    exported_pretokenizer = _bert_json_get(exported_root, "pre_tokenizer")
    @test String(_bert_json_get(exported_normalizer, "type")) == "BertNormalizer"
    @test Bool(_bert_json_get(exported_normalizer, "clean_text"))
    @test Bool(_bert_json_get(exported_normalizer, "handle_chinese_chars"))
    @test Bool(_bert_json_get(exported_normalizer, "strip_accents"))
    @test Bool(_bert_json_get(exported_normalizer, "lowercase"))
    @test String(_bert_json_get(exported_pretokenizer, "type")) == "BertPreTokenizer"

    reloaded = load_tokenizer(outdir; format=:hf_tokenizer_json)
    @test reloaded isa HuggingFaceJSONTokenizer

    for text in ("Hello,world!", "Café", "你好")
        @test tokenize(reloaded, text) == tokenize(tokenizer, text)
        for add_special_tokens in (false, true)
            @test encode(reloaded, text; add_special_tokens=add_special_tokens) ==
                  encode(tokenizer, text; add_special_tokens=add_special_tokens)
        end
        ids_reloaded = encode(reloaded, text; add_special_tokens=true)
        ids_original = encode(tokenizer, text; add_special_tokens=true)
        @test decode(reloaded, ids_reloaded) == decode(tokenizer, ids_original)
    end
end
