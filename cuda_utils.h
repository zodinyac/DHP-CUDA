#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y)
#define BLOCK_DIM dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)

#define GRID_DIM_X(x) ((x - 1) / BLOCK_SIZE_X + 1)
#define GRID_DIM_Y(y) ((y - 1) / BLOCK_SIZE_Y + 1)
#define GRID_DIM(x, y) dim3(GRID_DIM_X(x), GRID_DIM_Y(y))

#define INDEX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define INDEX_Y (blockIdx.y * blockDim.y + threadIdx.y)
#define INDEX(c) (INDEX_Y * c + INDEX_X)

#define THREAD_IDX (threadIdx.y * BLOCK_SIZE_X + threadIdx.x)

// left part of equation -Laplace u
#define left_part(P, i, j)                                                 \
    ((-(P[ic*(j)+i+1]-P[ic*(j)+i])/hx+(P[ic*(j)+i]-P[ic*(j)+i-1])/hx)/hx+  \
    (-(P[ic*(j+1)+i]-P[ic*(j)+i])/hy+(P[ic*(j)+i]-P[ic*(j-1)+i])/hy)/hy)
    
// function phi(x, y). it's also solution function
__device__ double boundary_value(double x, double y)
{
	return log(1.0 + x * y);
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMax(double *address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ void cuda_reduce_sum(volatile double *shared, int thread_idx, double val)
{
    shared[thread_idx] = val;
    __syncthreads();
    if (BLOCK_SIZE >= 512 && thread_idx < 256) {
        shared[thread_idx] += shared[thread_idx + 256];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 256 && thread_idx < 128) {
        shared[thread_idx] += shared[thread_idx + 128];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 128 && thread_idx < 64) {
        shared[thread_idx] += shared[thread_idx + 64];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 64 && thread_idx < 32) {
        shared[thread_idx] += shared[thread_idx + 32];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 32 && thread_idx < 16) {
        shared[thread_idx] += shared[thread_idx + 16];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 16 && thread_idx < 8) {
        shared[thread_idx] += shared[thread_idx + 8];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 8 && thread_idx < 4) {
        shared[thread_idx] += shared[thread_idx + 4];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 4 && thread_idx < 2) {
        shared[thread_idx] += shared[thread_idx + 2];
    }
    __syncthreads();
    if (BLOCK_SIZE >= 2 && thread_idx < 1) {
        shared[thread_idx] += shared[thread_idx + 1];
    }
}

__device__ void cuda_reduce_max(volatile double *shared, int thread_idx, double val)
{
    shared[thread_idx] = val;
    __syncthreads();
    if (BLOCK_SIZE >= 512 && thread_idx < 256) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 256]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 256 && thread_idx < 128) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 128]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 128 && thread_idx < 64) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 64]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 64 && thread_idx < 32) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 32]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 32 && thread_idx < 16) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 16]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 16 && thread_idx < 8) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 8]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 8 && thread_idx < 4) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 4]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 4 && thread_idx < 2) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 2]);
    }
    __syncthreads();
    if (BLOCK_SIZE >= 2 && thread_idx < 1) {
        shared[thread_idx] = max(shared[thread_idx], shared[thread_idx + 1]);
    }
}

#endif
