"""
    train_hf_bert_wordpiece(corpus; kwargs...) -> HuggingFaceJSONTokenizer

Train a BERT-style WordPiece tokenizer and return a `HuggingFaceJSONTokenizer`
pipeline composed of:
- `BertNormalizer`
- `BertPreTokenizer`
- `BertProcessing` (CLS/SEP insertion)
- `WordPiece` decoder

Special token behavior:
- `add_special_tokens=true` inserts `[CLS]` and `[SEP]` via post-processing.
- Special tokens present verbatim in input text can also be matched via HF
  `added_tokens` patterns.

KeemenaPreprocessing integration:
- `tokenization_text = tokenization_view(tokenizer, clean_text)`
- `encode_result(tokenizer, tokenization_text; assume_normalized=true,
  return_offsets=true, return_masks=true)`

Export/reload flow:
- `export_tokenizer(tokenizer, out_dir; format=:hf_tokenizer_json)`
- `load_hf_tokenizer_json(joinpath(out_dir, "tokenizer.json"))`
"""
function train_hf_bert_wordpiece(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "[UNK]",
        :pad => "[PAD]",
        :cls => "[CLS]",
        :sep => "[SEP]",
        :mask => "[MASK]",
    ),
    continuation_prefix::String="##",
    max_input_chars_per_word::Int=100,
    clean_text::Bool=true,
    handle_chinese_chars::Bool=true,
    lowercase::Bool=true,
    strip_accents::Union{Nothing,Bool}=nothing,
    model_name::String="trained_hf_bert_wordpiece",
    version::VersionNumber=v"0.3.0",
)::HuggingFaceJSONTokenizer
    return train_hf_bert_wordpiece_result(
        corpus;
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        continuation_prefix=continuation_prefix,
        max_input_chars_per_word=max_input_chars_per_word,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        lowercase=lowercase,
        strip_accents=strip_accents,
        model_name=model_name,
        version=version,
    ).tokenizer
end

"""
    train_hf_bert_wordpiece_result(corpus; kwargs...) ->
        TrainingResult{HuggingFaceJSONTokenizer,BertWordPieceTrainingConfig,BertWordPieceTrainingArtifacts}

Train a BERT-style WordPiece tokenizer and return:
- `tokenizer::HuggingFaceJSONTokenizer`
- `config::BertWordPieceTrainingConfig`
- `artifacts::BertWordPieceTrainingArtifacts`

The returned tokenizer includes `BertNormalizer`, `BertPreTokenizer`,
`BertProcessing`, and `WordPiece` decoding, with special tokens exported as HF
`added_tokens` for deterministic save/reload parity.
"""
function train_hf_bert_wordpiece_result(
    corpus;
    vocab_size::Int,
    min_frequency::Int=2,
    special_tokens::Dict{Symbol,String}=Dict(
        :unk => "[UNK]",
        :pad => "[PAD]",
        :cls => "[CLS]",
        :sep => "[SEP]",
        :mask => "[MASK]",
    ),
    continuation_prefix::String="##",
    max_input_chars_per_word::Int=100,
    clean_text::Bool=true,
    handle_chinese_chars::Bool=true,
    lowercase::Bool=true,
    strip_accents::Union{Nothing,Bool}=nothing,
    model_name::String="trained_hf_bert_wordpiece",
    version::VersionNumber=v"0.3.0",
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    BertWordPieceTrainingConfig,
    BertWordPieceTrainingArtifacts,
}
    config = BertWordPieceTrainingConfig(
        vocab_size,
        min_frequency,
        _normalize_special_tokens(special_tokens),
        String(continuation_prefix),
        max_input_chars_per_word,
        clean_text,
        handle_chinese_chars,
        lowercase,
        strip_accents,
        String(model_name),
        version,
    )
    return _train_hf_bert_wordpiece_result_impl(corpus, config)
end

function _train_hf_bert_wordpiece_result_impl(
    corpus,
    config::BertWordPieceTrainingConfig,
)::TrainingResult{
    HuggingFaceJSONTokenizer,
    BertWordPieceTrainingConfig,
    BertWordPieceTrainingArtifacts,
}
    _validate_hf_bert_wordpiece_config(config)

    normalizer = HFBertNormalizer(
        config.clean_text,
        config.handle_chinese_chars,
        config.strip_accents,
        config.lowercase,
    )
    training_pretokenizer = _hf_bert_training_pretokenizer(normalizer)

    inner_config = WordPieceTrainingConfig(
        config.vocab_size,
        config.min_frequency,
        config.special_tokens,
        training_pretokenizer,
        config.continuation_prefix,
        config.max_input_chars_per_word,
        config.model_name * "_inner",
        config.version,
    )
    inner_result = _train_wordpiece_result_impl(corpus, inner_config)
    base_tokenizer = inner_result.tokenizer

    cls_token = config.special_tokens[:cls]
    sep_token = config.special_tokens[:sep]
    cls_id = token_to_id(base_tokenizer, cls_token)
    sep_id = token_to_id(base_tokenizer, sep_token)

    model = HFWordPieceModelSpec(
        copy(base_tokenizer.vocab.id_to_token),
        base_tokenizer.unk_token,
        base_tokenizer.continuation_prefix,
        base_tokenizer.max_input_chars_per_word,
    )
    pretokenizer = HFBertPreTokenizer()
    postprocessor = HFBertProcessingPostProcessor(cls_token, cls_id, sep_token, sep_id)
    decoder = HFWordPieceDecoder(base_tokenizer.continuation_prefix)
    hf_added_tokens = _hf_bert_special_added_tokens(base_tokenizer, config.special_tokens)

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
    artifacts = BertWordPieceTrainingArtifacts(
        inner_result.artifacts,
        copy(hf_added_tokens),
    )
    return TrainingResult(tokenizer, config, artifacts)
end

function _hf_bert_training_pretokenizer(
    normalizer::HFBertNormalizer,
)::Function
    return text -> begin
        normalized = _apply_hf_normalizer(normalizer, String(text))
        pieces = _hf_bert_pretokenize_with_spans(normalized)
        out = String[]
        for piece in pieces
            isempty(piece.piece) && continue
            push!(out, piece.piece)
        end
        return out
    end
end

function _hf_bert_special_added_tokens(
    tokenizer::WordPieceTokenizer,
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
