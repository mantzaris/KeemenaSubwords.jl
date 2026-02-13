@testset "Section 7 training" begin
    include("test_bpe_training_e2e.jl")
    include("test_bpe_training_offsets.jl")
    include("test_training_stubs.jl")
end
