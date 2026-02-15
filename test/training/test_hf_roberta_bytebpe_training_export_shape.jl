using JSON3

function _roberta_json_get(obj, key::String)
    if haskey(obj, key)
        return obj[key]
    elseif haskey(obj, Symbol(key))
        return obj[Symbol(key)]
    end
    throw(ArgumentError("Missing key '$key' in JSON object"))
end

function _roberta_json_has(obj, key::String)::Bool
    return haskey(obj, key) || haskey(obj, Symbol(key))
end

@testset "HF RoBERTa ByteBPE training export shape" begin
    corpus = [
        "hello world",
        "hello, world!",
        "byte level bpe",
        "café costs 5 euros",
        "emoji 🙂 token",
    ]

    training = train_hf_roberta_bytebpe_result(
        corpus;
        vocab_size=384,
        min_frequency=1,
        special_tokens=Dict(
            :unk => "<unk>",
            :pad => "<pad>",
            :bos => "<s>",
            :eos => "</s>",
            :mask => "<mask>",
        ),
        end_of_word_marker="</w>",
        add_prefix_space=false,
        trim_offsets=false,
        use_regex=false,
        nfkc=true,
        lowercase=true,
        model_name="training_hf_roberta_bytebpe_export_shape",
        version=v"0.3.0",
    )

    outdir = mktempdir()
    export_tokenizer(training.tokenizer, outdir; format=:hf_tokenizer_json)
    tokenizer_path = joinpath(outdir, "tokenizer.json")
    @test isfile(tokenizer_path)

    root = JSON3.read(read(tokenizer_path, String))

    normalizer = _roberta_json_get(root, "normalizer")
    @test String(_roberta_json_get(normalizer, "type")) == "Sequence"
    normalizers = _roberta_json_get(normalizer, "normalizers")
    @test length(normalizers) == 2
    @test String(_roberta_json_get(normalizers[1], "type")) == "NFKC"
    @test String(_roberta_json_get(normalizers[2], "type")) == "Lowercase"

    pretokenizer = _roberta_json_get(root, "pre_tokenizer")
    @test String(_roberta_json_get(pretokenizer, "type")) == "ByteLevel"
    @test Bool(_roberta_json_get(pretokenizer, "add_prefix_space")) == training.config.add_prefix_space
    @test Bool(_roberta_json_get(pretokenizer, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_roberta_json_get(pretokenizer, "use_regex")) == training.config.use_regex

    decoder = _roberta_json_get(root, "decoder")
    @test String(_roberta_json_get(decoder, "type")) == "ByteLevel"
    @test Bool(_roberta_json_get(decoder, "add_prefix_space")) == training.config.add_prefix_space
    @test Bool(_roberta_json_get(decoder, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_roberta_json_get(decoder, "use_regex")) == training.config.use_regex

    postprocessor = _roberta_json_get(root, "post_processor")
    @test String(_roberta_json_get(postprocessor, "type")) == "RobertaProcessing"
    cls = _roberta_json_get(postprocessor, "cls")
    sep = _roberta_json_get(postprocessor, "sep")
    @test length(cls) == 2
    @test length(sep) == 2
    @test cls[1] == "<s>"
    @test sep[1] == "</s>"
    @test cls[2] isa Integer && cls[2] >= 0
    @test sep[2] isa Integer && sep[2] >= 0
    @test Bool(_roberta_json_get(postprocessor, "trim_offsets")) == training.config.trim_offsets
    @test Bool(_roberta_json_get(postprocessor, "add_prefix_space")) == training.config.add_prefix_space

    model = _roberta_json_get(root, "model")
    @test String(_roberta_json_get(model, "type")) == "BPE"
    @test String(_roberta_json_get(model, "unk_token")) == "<unk>"
    @test _roberta_json_has(model, "vocab")
    vocab = _roberta_json_get(model, "vocab")
    @test _roberta_json_has(vocab, "<unk>")
    @test _roberta_json_has(vocab, "<s>")
    @test _roberta_json_has(vocab, "</s>")

    vocab_ids = sort(collect(Int(id_zero) for (_, id_zero) in pairs(vocab)))
    @test !isempty(vocab_ids)
    @test first(vocab_ids) == 0
    @test vocab_ids == collect(0:(length(vocab_ids) - 1))

    added_tokens = _roberta_json_get(root, "added_tokens")
    @test !isempty(added_tokens)
    required_specials = Set(["<unk>", "<pad>", "<s>", "</s>", "<mask>"])
    seen_specials = Set{String}()

    for entry in added_tokens
        token = String(_roberta_json_get(entry, "content"))
        if token in required_specials
            push!(seen_specials, token)
            @test Bool(_roberta_json_get(entry, "special"))
            @test Int(_roberta_json_get(entry, "id")) >= 0
        end
    end

    @test seen_specials == required_specials
end
