#=
using Revise; include("test/runtests.jl")
=#
using Test
import CountReadsWrites as CRW
import Base.Broadcast: Broadcasted, broadcasted, instantiate, materialize

function get_args(N)
    x = zeros(N)
    y = zeros(N)
    A = zeros(N, N)
    vecmat = collect(map(x->zeros(3,3), 1:N))
    vecvec = collect(map(x->zeros(3), 1:N))
    return (; x, y, A, vecmat, vecvec)
end

function test_kernel(kernel!, N = 10, args = get_args(N))
    c = kernel!(args); # compile first
    c = CRW.@count_reads_writes kernel!(args)
    ans = answer(kernel!)
    n_successes = 0
    if CRW.n_reads(c) == ans.nreads*N
        n_successes+=1
    else
        @error "Broken reads for kernel $(kernel!). Expected $(ans.nreads*N) got $(CRW.n_reads(c))."
    end
    if CRW.n_writes(c) == ans.nwrites*N
        n_successes+=1
    else
        @error "Broken writes for kernel $(kernel!). Expected $(ans.nwrites*N) got $(CRW.n_writes(c))."
    end
    return (c, n_successes)
end

function kernel_assignment!(args)
    (; x, y) = args
    for i in eachindex(x)
        x[i] = y[i]
    end
    return nothing
end
answer(::typeof(kernel_assignment!)) = (; nreads=1,nwrites=1)

function kernel_bc_simple!(args)
    (; x, y) = args
    z = x .+ 1 # setindex! is used when `x .+ 1` is populated into a new array.
    return nothing
end
answer(::typeof(kernel_bc_simple!)) = (; nreads=1,nwrites=1)

function kernel_bc_assignment!(args)
    (; x, y) = args
    @. x = y
    return nothing
end
answer(::typeof(kernel_bc_assignment!)) = (; nreads=1,nwrites=1)

function kernel_bc_add!(args)
    (; x, y) = args
    @. x = y + 1
    return nothing
end
answer(::typeof(kernel_bc_add!)) = (; nreads=1,nwrites=1)

function kernel_bc_assignment_incremental_1!(args)
    (; x, y) = args
    # broadcasted(+, y)
    copyto!(y, instantiate(broadcasted(identity, x)))
    return nothing
end
answer(::typeof(kernel_bc_assignment_incremental_1!)) = (; nreads=1,nwrites=1)

function kernel_plus_equals!(args)
    (; x,y) = args
    for i in eachindex(x)
        x[i] += y[i]
    end
    return nothing
end
answer(::typeof(kernel_plus_equals!)) = (; nreads=2,nwrites=1)

function kernel_matvecmul!(args)
    (; vecmat,vecvec) = args
    for i in eachindex(vecvec)
        vecmat[i]*vecvec[i]
    end
    return nothing
end
answer(::typeof(kernel_matvecmul!)) = (; nreads=2,nwrites=0)

function kernel_matvecmul_bc_assign!(args)
    (; vecmat,vecvec) = args
    for i in eachindex(vecvec)
        vecmat[i] .= vecmat[i]*vecvec[i]
    end
    return nothing
end
answer(::typeof(kernel_matvecmul_bc_assign!)) = (; nreads=2,nwrites=3)

@testset "CountReadsWrites" begin
    (c,n) = test_kernel(kernel_assignment!); @test n == 2
    (c,n) = test_kernel(kernel_plus_equals!); @test n == 2
    (c,n) = test_kernel(kernel_matvecmul!); @test n == 2
    (c,n) = test_kernel(kernel_bc_simple!); @test n == 2
    (c,n) = test_kernel(kernel_bc_add!); @test n == 2
    (c,n) = test_kernel(kernel_bc_assignment!); @test_broken n == 2
    (c,n) = test_kernel(kernel_bc_assignment_incremental_1!); @test_broken n == 2
end

#=
CRW.@count_reads_writes does not work for `@. x = y` because
Julia special cases this assignment and calls
`copyto!(dest, do, src, so, N)`, which eventually calls
```
function memmove(dst::Ptr, src::Ptr, n::Integer)
    ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, n)
end
```
which performs the memory movement.


This special case was added in Julia 1.10, and is not really useful
for performance analysis. So, we may leave it broken for the moment.


Here is a script for digging into the src:
```julia
using Revise; include("test/runtests.jl")
import Base.Broadcast: Broadcast, Broadcasted, broadcasted, instantiate, materialize
N = 10
args = get_args(N);
(; x, y) = args;
bc = broadcasted(identity, x);
bci = instantiate(bc);
bct = Core.apply_type(Broadcasted, Broadcast.Nothing)
bcc = Broadcast.convert(bct, bci)
@code_lowered copyto!(y, bcc)
@code_lowered copyto!(y, bcc.args[1])
@code_lowered copyto!(y, 1, bcc.args[1], 1, length(bcc.args[1]))
```
=#
