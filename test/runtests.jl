using Test
using iTEBD

const TEST_ROOT = @__DIR__

include(joinpath(TEST_ROOT, "test_utils.jl"))

const TEST_GROUPS = Dict(
    "unit" => [
        "test_tensor_algebra.jl",
        "test_observables.jl",
        "test_adaptive_bonddim.jl",
        "test_imps_canonical_api.jl",
        "test_gate_api.jl",
        "test_krylov.jl",
        "test_scarfinder_api.jl",
        "test_scarfinder_regression.jl",
        "test_scarfinder_convergence.jl",
        "test_performance_improvements.jl",
        "test_schmidt_fixes.jl",
        "test_imps_fixes.jl",
        "test_symmetric_struct.jl",
    ],
    "api" => [
        "test_truncation_control.jl",
        "test_evolve_api.jl",
        "test_scarfinder_nstep.jl",
    ],
    "smoke" => [
        "test_docs_smoke.jl",
    ],
    "bench" => [
        "test_bench_smoke.jl",
        "test_pxp_legacy_smoke.jl",
    ],
    "integration" => [
        "test_aklt_integration.jl",
    ],
)

const TEST_ALIASES = Dict(
    "default" => ["unit", "api", "smoke", "integration"],
    "fast" => ["unit", "api", "smoke"],
    "all" => ["unit", "api", "smoke", "bench", "integration"],
    "long" => ["integration"],
)

function _requested_test_groups()
    raw = lowercase(strip(get(ENV, "ITEBD_TEST_GROUP", "default")))
    isempty(raw) && return TEST_ALIASES["default"]

    groups = String[]
    for token in split(raw, [',', ';', ' ', ':']; keepempty=false)
        if haskey(TEST_ALIASES, token)
            append!(groups, TEST_ALIASES[token])
        elseif haskey(TEST_GROUPS, token)
            push!(groups, token)
        else
            throw(ArgumentError(
                "unknown ITEBD_TEST_GROUP=$(repr(token)); choose one of " *
                join(sort!(collect(keys(TEST_GROUPS))), ", ") *
                " or aliases " *
                join(sort!(collect(keys(TEST_ALIASES))), ", ")
            ))
        end
    end
    return unique(groups)
end

function _selected_test_files(groups)
    files = String[]
    for group in groups
        append!(files, TEST_GROUPS[group])
    end
    return unique(files)
end

function _selected_test_files_from_args(args)
    files = String[]
    for arg in args
        file = basename(arg)
        path = joinpath(TEST_ROOT, file)
        isfile(path) || throw(ArgumentError(
            "unknown test file $(repr(arg)); pass a filename under test/ or use ITEBD_TEST_GROUP"
        ))
        push!(files, file)
    end
    return unique(files)
end

const REQUESTED_GROUPS = isempty(ARGS) ? _requested_test_groups() : String["ARGS"]
const REQUESTED_FILES = isempty(ARGS) ? _selected_test_files(REQUESTED_GROUPS) : _selected_test_files_from_args(ARGS)

@info "Running iTEBD test groups" groups=REQUESTED_GROUPS files=REQUESTED_FILES

@testset "iTEBD.jl" begin
    for file in REQUESTED_FILES
        @testset "$file" begin
            include(joinpath(TEST_ROOT, file))
        end
    end
end
