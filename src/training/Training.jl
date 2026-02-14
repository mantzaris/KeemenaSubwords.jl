module Training

import ..KeemenaSubwords: BPETokenizer,
    ByteBPETokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
    TokenizerMetadata,
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
    train_bpe,
    train_bpe_result,
    train_bytebpe,
    train_bytebpe_result,
    train_unigram,
    train_unigram_result,
    train_wordpiece,
    train_wordpiece_result,
    train_sentencepiece

end # module Training
