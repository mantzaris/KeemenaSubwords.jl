"""
    train_hf_roberta_bytebpe(corpus; kwargs...) -> HuggingFaceJSONTokenizer

Train a RoBERTa-style ByteLevel BPE tokenizer and return a
`HuggingFaceJSONTokenizer` pipeline composed of:
- ByteLevel pre-tokenization
- RobertaProcessing (`<s> ... </s>` insertion)
- ByteLevel decoding

Special token behavior:
- `add_special_tokens=true` inserts BOS/EOS via RobertaProcessing.
- Special tokens present verbatim in input text can be matched via HF
  `added_tokens` patterns.
- By default the preset enables HF-style ByteLevel settings:
  `use_regex=true`, `add_prefix_space=true`, and `trim_offsets=true`.

KeemenaPreprocessing integration:
- `tokenization_text = tokenization_view(tokenizer, clean_text)`
- `encode_result(tokenizer, tokenization_text; assume_normalized=true,
  return_offsets=true, return_masks=true)`

Export/reload flow:
- `export_tokenizer(tokenizer, out_dir; format=:hf_tokenizer_json)`
- `load_hf_tokenizer_json(joinpath(out_dir, "tokenizer.json"))`
"""
function train_hf_roberta_bytebpe(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "<unk>",
        :pad => "<pad>",
        :bos => "<s>",
        :eos => "</s>",
        :mask => "<mask>",
    ),
    end_of_word_marker::String="</w>",
    add_prefix_space::Bool=true,
    trim_offsets::Bool=true,
    use_regex::Bool=true,
    nfkc::Bool=false,
    lowercase::Bool=false,
    model_name::String="trained_hf_roberta_bytebpe",
    version::VersionNumber=v"0.3.0",
)::HuggingFaceJSONTokenizer
    return train_hf_roberta_bytebpe_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        end_of_word_marker=end_of_word_marker,
        add_prefix_space=add_prefix_space,
        trim_offsets=trim_offsets,
        use_regex=use_regex,
        nfkc=nfkc,
        lowercase=lowercase,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
    train_hf_roberta_bytebpe_result(corpus; kwargs...) ->
        TrainingResult{HuggingFaceJSONTokenizer,RobertaByteBPETrainingConfig,RobertaByteBPETrainingArtifacts}

Train a RoBERTa-style ByteLevel BPE tokenizer and return:
- `tokenizer::HuggingFaceJSONTokenizer`
- `config::RobertaByteBPETrainingConfig`
- `artifacts::RobertaByteBPETrainingArtifacts`

The returned tokenizer wraps an inner trained `ByteBPETokenizer` and preserves a
HF-native pipeline (`ByteLevel` pre-tokenizer/decoder + `RobertaProcessing`).
"""
function train_hf_roberta_bytebpe_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "<unk>",
        :pad => "<pad>",
        :bos => "<s>",
        :eos => "</s>",
        :mask => "<mask>",
    ),
    end_of_word_marker::String="</w>",
    add_prefix_space::Bool=true,
    trim_offsets::Bool=true,
    use_regex::Bool=true,
    nfkc::Bool=false,
    lowercase::Bool=false,
    model_name::String="trained_hf_roberta_bytebpe",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    RobertaByteBPETrainingConfig,
    RobertaByteBPETrainingArtifacts,
}
    config = RobertaByteBPETrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        String(end_of_word_marker),
        add_prefix_space,
        trim_offsets,
        use_regex,
        nfkc,
        lowercase,
        String(model_name),
        version,
    )
    return _train_hf_roberta_bytebpe_result_impl(corpus, config)
end

function _train_hf_roberta_bytebpe_result_impl(
    corpus,
    config::RobertaByteBPETrainingConfig,
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    RobertaByteBPETrainingConfig,
    RobertaByteBPETrainingArtifacts,
}
    _validate_hf_roberta_bytebpe_config(config)

    normalizer = _hf_roberta_normalizer(config)
    training_pretokenizer = _hf_roberta_training_pretokenizer(config)
    inner_config = ByteBPETrainingConfig(
        config.vocab_size,
        config.min_frequency,
        config.special_tokens,
        config.end_of_word_marker,
        training_pretokenizer,
        true,
        config.model_name * "_inner",
        config.version,
    )
    inner_result = _train_bytebpe_result_impl(corpus, inner_config)
    base_tokenizer = inner_result.tokenizer

    cls_token = config.special_tokens[:bos]
    sep_token = config.special_tokens[:eos]
    cls_id = token_to_id(base_tokenizer, cls_token)
    sep_id = token_to_id(base_tokenizer, sep_token)

    model = HFBPEModelSpec(
        copy(base_tokenizer.base.vocab.id_to_token),
        copy(inner_result.artifacts.merge_pairs),
        base_tokenizer.base.unk_token,
        true,
        nothing,
        base_tokenizer.base.end_of_word_marker,
        false,
        false,
        nothing,
    )
    pretokenizer = HFByteLevelPreTokenizer(
        config.add_prefix_space,
        config.trim_offsets,
        config.use_regex,
    )
    postprocessor = HFRobertaProcessingPostProcessor(
        cls_token,
        cls_id,
        sep_token,
        sep_id,
        config.trim_offsets,
        config.add_prefix_space,
    )
    decoder = HFByteLevelDecoder(
        config.add_prefix_space,
        config.trim_offsets,
        config.use_regex,
    )
    hf_added_tokens = _hf_roberta_special_added_tokens(base_tokenizer, config.special_tokens)

    tokenizer = _build_hf_tokenizer_from_parts(
        model,
        base_tokenizer,
        normalizer,
        pretokenizer,
        postprocessor,
        decoder;
        added_tokens=hf_added_tokens,
        model_name=config.model_name,
        version=config.version,
        source_path="<trained>",
    )
    artifacts = RobertaByteBPETrainingArtifacts(
        inner_result.artifacts,
        copy(hf_added_tokens),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _hf_roberta_normalizer(
    config::RobertaByteBPETrainingConfig,
)
    items = HFJSONNormalizer[]
    config.nfkc && push!(items, HFNFKCNormalizer())
    config.lowercase && push!(items, HFLowercaseNormalizer())

    if isempty(items)
        return HFNoopNormalizer()
    elseif length(items) == 1
        return items[1]
    end

    return HFSequenceNormalizer(items)
end

function _hf_roberta_training_pretokenizer(
    config::RobertaByteBPETrainingConfig,
)::Function
    normalizer = _hf_roberta_normalizer(config)
    return text -> begin
        normalized = _apply_hf_normalizer(normalizer, String(text))
        raw_splits, _ = _hf_bytelevel_raw_splits_with_work_spans(
            normalized;
            add_prefix_space=config.add_prefix_space,
            use_regex=config.use_regex,
        )

        out = String[]
        for split in raw_splits
            isempty(split.piece) && continue
            push!(out, split.piece)
        end
        return out
    end
end

function _hf_roberta_special_added_tokens(
    tokenizer::ByteBPETokenizer,
    special_tokens_map::Dict{Symbol,String},
)::Vector{HFAddedToken}
    added_tokens = HFAddedToken[]
    seen_ids = Set{Int}()

    for pair in _ordered_special_token_pairs(special_tokens_map)
        token = pair.second
        token_id = token_to_id(tokenizer, token)
        token_id in seen_ids && continue
        push!(added_tokens, HFAddedToken(
            token,
            token_id,
            true,
            false,
            false,
            false,
            false,
        ))
        push!(seen_ids, token_id)
    end

    sort!(added_tokens; by=t -> (t.id, t.content))
    return added_tokens
end
