using Test, LinearMaps, ChainRulesTestUtils
using ChainRulesCore: NoTangent

@testset "AD rules" begin
    x = randn(10)
    for A in (
        LinearMap(rand(10, 10)),
        LinearMap(cumsum, reverse∘cumsum∘reverse, 10),
        LinearMap((y, x) -> cumsum!(y, x), (y, x) -> reverse!(cumsum!(y, reverse!(copyto!(y, x)))), 10)
    )
        test_rrule(*, A ⊢ NoTangent(), x)
        test_rrule(A ⊢ NoTangent(), x)
        test_rrule(*, A' ⊢ NoTangent(), x)
        test_rrule(A' ⊢ NoTangent(), x)
    end
end
