#include <stdio.h>
#include <time.h>
#include <math.h>

#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 5000
#define WORDS_PER_THREAD 32


#define gpuCheck(f) { gpuCheckFunc((f), __FILE__, __LINE__); }

inline void gpuCheckFunc(cudaError_t err, const char *file, int line){
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(err));
      exit(1);
   }
}

__device__ float reduce_add_block(float x, float *s_tmp) {
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

__device__ void reduce_add_global(float x, float *s_tmp, float *g_out) {
    x = reduce_add_block(x, s_tmp);
    atomicAdd(g_out, x);
}


// Saturates the GPU memory bandwidth (so throughput of ~280-290 GB/s on Titan V),
// with very low utilization of the ALUs. To get better efficiency we would need to fuse
// multiple layers together to reduce the amount of GPU memory loads and stores.
__global__ void butterfly_forward_slow(
    const float *data_in,
    const float *angles,
    float *data_out,
    int data_stride
) {
    // Load the angle for this thread's switch, and compute the corresponding weights.
    float angle = angles[blockIdx.y];
    float a = cos(angle);
    float b = sin(angle);
    
    // Load the input data
    int data_idx = 2 * blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    float x = data_in[data_idx];
    float y = data_in[data_idx + data_stride];

    // Write the output data
    data_out[data_idx] = a * x + b * y;
    data_out[data_idx + data_stride] = -b * x + a * y;
}

__global__ void butterfly_backward_slow(
    const float *data_in,
    const float *angles,
    const float *grad_in,
    float *grad_out,
    float *angles_grad_accum,
    int data_stride
) {
    // Load the angle for this thread's switch, and compute the corresponding weights.
    float angle = angles[blockIdx.y];
    float a = cos(angle);
    float b = sin(angle);
    
    // Load the input gradient
    int data_idx = 2 * blockIdx.y * data_stride + threadIdx.x + blockDim.x * blockIdx.x;
    float dx = grad_in[data_idx];
    float dy = grad_in[data_idx + data_stride];

    // Write the output gradient for continuing backpropagation into earlier layers
    grad_out[data_idx] = a * dx - b * dy;
    grad_out[data_idx + data_stride] = b * dx + a * dy;

    // Accumulate the gradient for the angles in the current layer
    __shared__ float tmp[32];
    float x = data_in[data_idx];
    float y = data_in[data_idx + data_stride];
    float g = y*dx - x*dy;
    reduce_add_global(g, tmp, &angles_grad_accum[blockIdx.y]);
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