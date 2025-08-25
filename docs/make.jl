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
    checkdocs = :exports,              # complain only for *exported* names :contentReference[oaicite:0]{index=0}
    pages = [
        "Home"          => "index.md"
    ],
)

deploydocs(repo      = "github.com/mantzaris/KeemenaSubwords.jl",
           devbranch = "main")