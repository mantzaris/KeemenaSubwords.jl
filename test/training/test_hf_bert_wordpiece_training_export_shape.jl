using JSON3

function _shape_json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

function _shape_json_has(obj, key::String)::Bool
    return haskey(obj, key) || haskey(obj, Symbol(key))
end

@testset "HF BERT WordPiece training export shape" begin
    corpus = [
        "Hello, world!",
        "Café naïve façade",
        "你好 世界",
        "Hello world",
    ]

    training = train_hf_bert_wordpiece_result(
        corpus;
        vocab_size=128,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "[UNK]",
            :pad => "[PAD]",
            :cls => "[CLS]",
            :sep => "[SEP]",
            :mask => "[MASK]",
        ),
        continuation_prefix="##",
        max_input_chars_per_word=100,
        clean_text=true,
        handle_chinese_chars=true,
        lowercase=true,
        strip_accents=nothing,
        model_name="training_hf_bert_wordpiece_export_shape",
        version=v"0.3.0",
    )
    tokenizer = training.tokenizer

    outdir = mktempdir()
    export_tokenizer(tokenizer, outdir; format=:hf_tokenizer_json)
    tokenizer_path = joinpath(outdir, "tokenizer.json")
    @test isfile(tokenizer_path)

    root = JSON3.read(read(tokenizer_path, String))

    normalizer = _shape_json_get(root, "normalizer")
    @test String(_shape_json_get(normalizer, "type")) == "BertNormalizer"
    @test Bool(_shape_json_get(normalizer, "clean_text")) == training.config.clean_text
    @test Bool(_shape_json_get(normalizer, "handle_chinese_chars")) == training.config.handle_chinese_chars
    @test Bool(_shape_json_get(normalizer, "lowercase")) == training.config.lowercase
    strip_accents = _shape_json_get(normalizer, "strip_accents")
    @test strip_accents === nothing || strip_accents isa Bool
    @test strip_accents == training.config.strip_accents

    pretokenizer = _shape_json_get(root, "pre_tokenizer")
    @test String(_shape_json_get(pretokenizer, "type")) == "BertPreTokenizer"

    postprocessor = _shape_json_get(root, "post_processor")
    @test String(_shape_json_get(postprocessor, "type")) == "BertProcessing"
    cls = _shape_json_get(postprocessor, "cls")
    sep = _shape_json_get(postprocessor, "sep")
    @test length(cls) == 2
    @test length(sep) == 2
    @test cls[1] == "[CLS]"
    @test sep[1] == "[SEP]"
    @test cls[2] isa Integer && cls[2] >= 0
    @test sep[2] isa Integer && sep[2] >= 0

    decoder = _shape_json_get(root, "decoder")
    @test String(_shape_json_get(decoder, "type")) == "WordPiece"
    @test String(_shape_json_get(decoder, "prefix")) == "##"

    model = _shape_json_get(root, "model")
    @test String(_shape_json_get(model, "type")) == "WordPiece"
    @test String(_shape_json_get(model, "unk_token")) == "[UNK]"
    @test String(_shape_json_get(model, "continuing_subword_prefix")) == "##"
    @test Int(_shape_json_get(model, "max_input_chars_per_word")) == 100
    vocab = _shape_json_get(model, "vocab")
    @test _shape_json_has(vocab, "[UNK]")
    @test _shape_json_has(vocab, "[CLS]")
    @test _shape_json_has(vocab, "[SEP]")

    added_tokens = _shape_json_get(root, "added_tokens")
    @test !isempty(added_tokens)
    required_specials = Set(["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
    seen_specials = Set{String}()

    for entry in added_tokens
        _shape_json_get(entry, "id") isa Integer || continue
        token = String(_shape_json_get(entry, "content"))
        is_special = Bool(_shape_json_get(entry, "special"))
        if token in required_specials
            push!(seen_specials, token)
            @test is_special
            @test Int(_shape_json_get(entry, "id")) >= 0
        end
    end

    @test seen_specials == required_specials
end
