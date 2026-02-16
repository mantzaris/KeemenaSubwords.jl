using Pkg
Pkg.activate(@__DIR__)                              # use docs/Project.toml
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))  # point to the local package
Pkg.instantiate()


using KeemenaSubwords
using Documenter

DocMeta.setdocmeta!(KeemenaSubwords, :DocTestSetup, :(using KeemenaSubwords); recursive=true)

makedocs(
    modules   = [KeemenaSubwords
            ],
    sitename  = "KeemenaSubwords.jl",
    authors   = "Alexander V. Mantzaris",
    format    = Documenter.HTML(;
                  canonical = "https://mantzaris.github.io/KeemenaSubwords.jl",
                  edit_link = "main"),
    doctest   = true,
    checkdocs = :exports,              # complain only for exported names
    pages = [
        "Home"          => "index.md",
        "Concepts"      => "concepts.md",
        "Structured Outputs and Batching" => "structured_outputs_and_batching.md",
        "Integration"   => "integration.md",
        "Normalization & Offsets" => "normalization_offsets_contract.md",
        "Offsets Alignment Examples" => "offset_alignment_examples.md",
        "Loading"       => "loading.md",
        "Training"      => "training.md",
        "Formats"       => "formats.md",
        "Loading Local" => "loading_local.md",
        "LLM Cookbook"  => "llm_cookbook.md",
        "Gated Models"  => "gated_models.md",
        "Troubleshooting" => "troubleshooting.md",
        "Built-In Models" => "models.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(repo      = "github.com/mantzaris/KeemenaSubwords.jl",
           devbranch = "main")
