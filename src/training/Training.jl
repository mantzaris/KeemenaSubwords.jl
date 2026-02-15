module Training

import ..KeemenaSubwords: BPETokenizer,
    ByteBPETokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
    HuggingFaceJSONTokenizer,
    TokenizerMetadata,
    HFBPEModelSpec,
    HFJSONNormalizer,
    HFWordPieceModelSpec,
    HFNoopNormalizer,
    HFLowercaseNormalizer,
    HFNFKCNormalizer,
    HFSequenceNormalizer,
    HFBertNormalizer,
    HFBertPreTokenizer,
    HFByteLevelPreTokenizer,
    HFBertProcessingPostProcessor,
    HFRobertaProcessingPostProcessor,
    HFByteLevelDecoder,
    HFWordPieceDecoder,
    HFAddedToken,
    _apply_hf_normalizer,
    _hf_bert_pretokenize_with_spans,
    _hf_bytelevel_raw_splits_with_work_spans,
    _build_hf_tokenizer_from_parts,
    token_to_id,
    _byte_unicode_tables,
    build_vocab

include("training_types.jl")
include("training_common.jl")
include("training_api.jl")
include("bpe_trainer.jl")
include("bytebpe_trainer.jl")
include("unigram_train.jl")
include("wordpiece_train.jl")
include("sentencepiece_train.jl")
include("presets/bert_wordpiece_hf.jl")
include("presets/roberta_bytebpe_hf.jl")

export AbstractTrainingConfig,
    AbstractTrainingArtifacts,
    TrainingResult,
    BPETrainingConfig,
    BPETrainingArtifacts,
    ByteBPETrainingConfig,
    ByteBPETrainingArtifacts,
    UnigramTrainingConfig,
    UnigramTrainingArtifacts,
    WordPieceTrainingConfig,
    WordPieceTrainingArtifacts,
    SentencePieceTrainingConfig,
    SentencePieceTrainingArtifacts,
    BertWordPieceTrainingConfig,
    BertWordPieceTrainingArtifacts,
    RobertaByteBPETrainingConfig,
    RobertaByteBPETrainingArtifacts,
    train_bpe,
    train_bpe_result,
    train_bytebpe,
    train_bytebpe_result,
    train_unigram,
    train_unigram_result,
    train_wordpiece,
    train_wordpiece_result,
    train_sentencepiece,
    train_sentencepiece_result,
    train_hf_bert_wordpiece,
    train_hf_bert_wordpiece_result,
    train_hf_roberta_bytebpe,
    train_hf_roberta_bytebpe_result

end # module Training
