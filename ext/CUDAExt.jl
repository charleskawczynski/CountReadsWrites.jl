module CUDAExt

using CUDA
CUDA.allowscalar(false)
import CountReadsWrites: cuda_types

cuda_types() = ((CUDA.CuArray, :CuArray),)

end