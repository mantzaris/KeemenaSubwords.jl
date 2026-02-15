using JSON3

function _gpt2_json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

function _gpt2_json_has(obj, key::String)::Bool
    return haskey(obj, key) || haskey(obj, Symbol(key))
end

@testset "HF GPT-2 ByteBPE training export shape" begin
    corpus = [
        "Hello my friend, how is your day going?",
        "Hello there\nHello there",
        "Hello there      dear",
        "café costs 5 euros",
        "emoji 🙂 token",
        "<|endoftext|> appears in text",
    ]

    training = train_hf_gpt2_bytebpe_result(
        corpus;
        vocab_size=384,
        min_frequency=1,
        special_tokens=Dict(:unk => "<|endoftext|>"),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=true,
        use_regex=true,
        export_unk_token_null=true,
        model_name="training_hf_gpt2_bytebpe_export_shape",
        version=v"0.3.0",
    )

    outdir = mktempdir()
    export_tokenizer(training.tokenizer, outdir; format=:hf_tokenizer_json)
    tokenizer_path = joinpath(outdir, "tokenizer.json")
    @test isfile(tokenizer_path)

    root = JSON3.read(read(tokenizer_path, String))

    normalizer = _gpt2_json_get(root, "normalizer")
    @test normalizer === nothing

    pretokenizer = _gpt2_json_get(root, "pre_tokenizer")
    @test String(_gpt2_json_get(pretokenizer, "type")) == "ByteLevel"
    @test Bool(_gpt2_json_get(pretokenizer, "add_prefix_space")) == training.config.add_prefix_space
    @test Bool(_gpt2_json_get(pretokenizer, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_gpt2_json_get(pretokenizer, "use_regex")) == training.config.use_regex

    postprocessor = _gpt2_json_get(root, "post_processor")
    @test String(_gpt2_json_get(postprocessor, "type")) == "ByteLevel"
    @test Bool(_gpt2_json_get(postprocessor, "add_prefix_space")) == training.config.add_prefix_space
    @test Bool(_gpt2_json_get(postprocessor, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_gpt2_json_get(postprocessor, "use_regex")) == training.config.use_regex

    decoder = _gpt2_json_get(root, "decoder")
    @test String(_gpt2_json_get(decoder, "type")) == "ByteLevel"
    @test Bool(_gpt2_json_get(decoder, "add_prefix_space")) == training.config.add_prefix_space
    @test Bool(_gpt2_json_get(decoder, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_gpt2_json_get(decoder, "use_regex")) == training.config.use_regex

    model = _gpt2_json_get(root, "model")
    @test String(_gpt2_json_get(model, "type")) == "BPE"
    @test _gpt2_json_has(model, "unk_token")
    @test _gpt2_json_get(model, "unk_token") === nothing
    @test _gpt2_json_has(model, "vocab")

    vocab = _gpt2_json_get(model, "vocab")
    @test _gpt2_json_has(vocab, "<|endoftext|>")

    vocab_ids = sort(collect(Int(id_zero) for (_, id_zero) in pairs(vocab)))
    @test !isempty(vocab_ids)
    @test first(vocab_ids) == 0
    @test vocab_ids == collect(0:(length(vocab_ids) - 1))

    added_tokens = _gpt2_json_get(root, "added_tokens")
    @test !isempty(added_tokens)
    saw_eot_special = false
    for entry in added_tokens
        content = String(_gpt2_json_get(entry, "content"))
        content == "<|endoftext|>" || continue
        saw_eot_special = true
        @test Bool(_gpt2_json_get(entry, "special"))
        @test Int(_gpt2_json_get(entry, "id")) >= 0
    end

    @test saw_eot_special
end
