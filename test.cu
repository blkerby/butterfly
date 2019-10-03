#include <ATen/ATen.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdio.h>
#include <time.h>
#include <math.h>


#define gpuCheck(f) { gpuCheckFunc((f), __FILE__, __LINE__); }

inline void gpuCheckFunc(cudaError_t err, const char *file, int line){
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(err));
      exit(1);
   }
}

template <typename scalar_t>
__device__ scalar_t reduce_add_block(scalar_t x, scalar_t *s_tmp) {
    // First reduce across the warp:
    for (int m = 1; m < 32; m <<= 1){
        x += __shfl_xor_sync(0xffffffff, x, m);
    }

    // Now reduce across the block
    if (threadIdx.x % 32 == 0) {
        s_tmp[threadIdx.x / 32] = x;
    }
    __syncthreads();
    if (threadIdx.x < blockDim.x / 32) {
        x = s_tmp[threadIdx.x];
    } else {
        x = 0.0;
    }
    for (int m = 1; m < blockDim.x / 32; m <<= 1){
        x += __shfl_xor_sync(0xffffffff, x, m);
    }
    return x;
}

template <typename scalar_t>
__device__ void reduce_add_global(scalar_t x, scalar_t *s_tmp, scalar_t *g_out) {
    x = reduce_add_block(x, s_tmp);
    atomicAdd(g_out, x);
}


// Saturates the GPU global memory bandwidth with very low utilization of the ALUs. 
// To get better efficiency we would need to fuse multiple layers together to 
// reduce the amount of GPU memory loads and stores.
template <typename scalar_t>
__global__ void cuda_butterfly_forward_slow_kernel(
    const scalar_t *data_in,
    const scalar_t *angles,
    scalar_t *data_out,
    int data_stride,
    int half_width
) {
    // Load the angle for this thread's switch, and compute the corresponding weights.
    scalar_t angle = angles[blockIdx.y];
    scalar_t a = cos(angle);
    scalar_t b = sin(angle);
    
    // Load the input data from GPU global memory
    int data_idx_in = 2 * blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    scalar_t x0 = data_in[data_idx_in];
    scalar_t y0 = data_in[data_idx_in + data_stride];

    // Compute the output data
    scalar_t x1 = a * x0 + b * y0;
    scalar_t y1 = -b * x0 + a * y0;

    // Write the output data to GPU global memory
    int data_idx_out = blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    data_out[data_idx_out] = x1;
    data_out[data_idx_out + data_stride * half_width] = y1;
}

template <typename scalar_t>
__global__ void cuda_butterfly_backward_slow_kernel(
    const scalar_t *data_in,
    const scalar_t *angles,
    const scalar_t *grad_in,
    scalar_t *grad_out,
    scalar_t *grad_angles_accum,
    int data_stride,
    int half_width
) {
    // Load the angle for this thread's switch, and compute the corresponding weights.
    scalar_t angle = angles[blockIdx.y];
    scalar_t a = cos(angle);
    scalar_t b = sin(angle);
    
    // Load the input gradient
    int data_idx_in = blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    scalar_t dx1 = grad_in[data_idx_in];
    scalar_t dy1 = grad_in[data_idx_in + data_stride * half_width];

    // Compute the output gradient for continuing backpropagation into earlier layers
    scalar_t dx0 = a * dx1 - b * dy1;
    scalar_t dy0 = b * dx1 + a * dy1;

    // Write the output gradient to GPU global memory
    int data_idx_out = 2 * blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    grad_out[data_idx_out] = dx0;
    grad_out[data_idx_out + data_stride] = dy0;

    // Accumulate the gradient for the angles in the current layer
    __shared__ scalar_t tmp[32];
    scalar_t x1 = data_in[data_idx_in];
    scalar_t y1 = data_in[data_idx_in + data_stride * half_width];
    scalar_t g = y1*dx1 - x1*dy1;
    reduce_add_global(g, tmp, &grad_angles_accum[blockIdx.y]);
}

void cuda_butterfly_forward_slow(at::Tensor data_in, at::Tensor angles, at::Tensor data_out) {
    int dimBlock = 256;
    dim3 dimGrid(data_in.size(1) / dimBlock, angles.size(0));

    AT_DISPATCH_FLOATING_TYPES(data_in.type(), "test_cuda_double", ([&] {
        cuda_butterfly_forward_slow_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            data_in.data<scalar_t>(),
            angles.data<scalar_t>(),
            data_out.data<scalar_t>(),
            data_in.size(1),
            data_in.size(0) / 2
        );
        gpuCheck( cudaGetLastError() )
    }));
}


void cuda_butterfly_backward_slow(
    at::Tensor data_in,
    at::Tensor angles, 
    at::Tensor grad_in,
    at::Tensor grad_out,
    at::Tensor grad_angles_accum
) {
    int dimBlock = 256;
    dim3 dimGrid(data_in.size(1) / dimBlock, angles.size(0));

    AT_DISPATCH_FLOATING_TYPES(data_in.type(), "test_cuda_double", ([&] {
        cuda_butterfly_backward_slow_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            data_in.data<scalar_t>(),
            angles.data<scalar_t>(),
            grad_in.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            grad_angles_accum.data<scalar_t>(),
            data_in.size(1),
            data_in.size(0) / 2
        );
        gpuCheck( cudaGetLastError() )
    }));
}


// int main() {
//     float *h_angles;
//     float *d_angles;
//     float *h_data;
//     float *d_data_in, *d_data_out;
//     long rounds = 1000;
//     int width_pow = 12;
//     int num_rows = 2048;
//     int num_cols = 1 << width_pow;
//     int len_angles = num_cols / 2;
//     int len_data = num_rows * num_cols;
//     int size_angles = len_angles * sizeof(float);
//     long size_data = len_data * sizeof(float);

//     h_angles = (float *)malloc(size_angles);
//     h_data = (float *)malloc(size_data);
//     if (h_data == NULL) {
//         printf("Unable to allocate host memory");
//         exit(1);
//     }
//     gpuCheck( cudaMalloc(&d_angles, size_angles) )
//     gpuCheck( cudaMalloc(&d_data_in, size_data) )
//     gpuCheck( cudaMalloc(&d_data_out, size_data) )

//     for (int i = 0; i < len_angles; i++) {
//         h_angles[i] = M_PI / 2;
//     }
//     for (long i = 0; i < len_data; i++) {
//         h_data[i] = i + 1; 
//     }

//     gpuCheck( cudaMemcpy(d_angles, h_angles, size_angles, cudaMemcpyHostToDevice) )
//     gpuCheck( cudaMemcpy(d_data_in, h_data, size_data, cudaMemcpyHostToDevice) )
//     gpuCheck( cudaMemset(d_data_out, 0, size_data) )
//     gpuCheck( cudaDeviceSynchronize() )

//     int dimBlock = 256;
//     dim3 dimGrid(num_rows / dimBlock, len_angles);
 
//     clock_t begin = clock();
//     for (long i = 0; i < rounds; i++){
//         butterfly_forward_slow<<<dimGrid, dimBlock>>>(
//             d_data_in, 
//             d_angles,
//             d_data_out,
//             num_rows
//         );
//         gpuCheck( cudaGetLastError() )
//         gpuCheck( cudaDeviceSynchronize() )
//     }
//     clock_t end = clock();
//     double duration = (double)(end - begin) / CLOCKS_PER_SEC;
//     double bytes = (double)size_data * rounds;
//     printf("Took %f seconds (%f GB/s)\n", duration, bytes / duration / 1000000000.0);
//     gpuCheck( cudaMemcpy(h_data, d_data_out, size_data, cudaMemcpyDeviceToHost) )
//     gpuCheck( cudaDeviceSynchronize() )

//     for (int i = 0; i < len_data; i++) {
//         if (h_data[i] == 0.0) {
//             printf("zero at %d\n", i);
//         }
//     }
    
//     free(h_angles);
//     free(h_data);
//     gpuCheck( cudaFree(d_angles) )
//     gpuCheck( cudaFree(d_data_in) )
//     gpuCheck( cudaFree(d_data_out) )
// }