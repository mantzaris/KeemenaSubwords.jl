@testset "Extended validity suite" begin
    include("test_large_corpus_training_smoke.jl")
    include("test_training_order_invariance.jl")
    include("test_large_corpus_bundle_roundtrip.jl")
end
