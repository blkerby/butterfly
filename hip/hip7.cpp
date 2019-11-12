#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include <chrono> 

using namespace std::chrono;
using namespace std;


#define HIP_CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define MODULE_DEPTH 4
#define MODULE_WIDTH (1 << MODULE_DEPTH)

#define BUTTERFLY_NUM_MODULES 4
#define BUTTERFLY_THREADS_PER_BLOCK (MODULE_WIDTH * MODULE_WIDTH)
#define BUTTERFLY_LOOP_SIZE 16
#define BUTTERFLY_WIDTH_POW (2 * MODULE_DEPTH)
#define BUTTERFLY_WIDTH (1 << BUTTERFLY_WIDTH_POW)
#define BUTTERFLY_DEPTH (BUTTERFLY_NUM_MODULES * MODULE_DEPTH)
#define BUTTERFLY_BATCH_SIZE (BUTTERFLY_THREADS_PER_BLOCK / MODULE_WIDTH)
#define BUTTERFLY_NUM_ANGLES (BUTTERFLY_WIDTH / 2 * BUTTERFLY_DEPTH)


template <typename T>
__global__ __launch_bounds__(BUTTERFLY_THREADS_PER_BLOCK, 2) void butterfly_forward(
    T** g_data_in,
    T* g_data_out,
    const T* g_angles, 
    int col_stride,
    int num_modules
) {
    __shared__ T *s_data_in[MODULE_WIDTH];
    __shared__ T s_cosines[BUTTERFLY_NUM_ANGLES];
    __shared__ T s_sines[BUTTERFLY_NUM_ANGLES];
    __shared__ T tmp[MODULE_WIDTH * BUTTERFLY_BATCH_SIZE];
    T data[MODULE_WIDTH];

    // Load the data indices (specifying which data columns participate in the butterfly)
    if (hipThreadIdx_y == 0) {
        s_data_in[hipThreadIdx_x] = g_data_in[hipThreadIdx_x];
    }

    // Load the angles from global memory, putting their cosines & sines into shared memory.
    for (int i = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x; i < BUTTERFLY_NUM_ANGLES; 
            i+= BUTTERFLY_THREADS_PER_BLOCK) {
        T angle = g_angles[i];
        s_cosines[i] = cos(angle);
        s_sines[i] = sin(angle);
    }
    __syncthreads();

    int row_idx = hipBlockIdx_x * MODULE_WIDTH;
    #pragma unroll 1   // Do not unroll this loop
    for (int l = 0; l < BUTTERFLY_LOOP_SIZE; l++) {
        // Load a row of data from global memory into registers
        #pragma unroll  // Unroll this loop completely
        for (int i = 0; i < MODULE_WIDTH; i++) {
            T *data_in = s_data_in[i];
            __syncthreads();
            tmp[hipThreadIdx_x + hipThreadIdx_y * MODULE_WIDTH] = 
                // data_in[0];
                data_in[row_idx + hipThreadIdx_x + hipThreadIdx_y * col_stride];
            __syncthreads();
            data[i] = tmp[hipThreadIdx_y + hipThreadIdx_x * MODULE_WIDTH];
        }

        int angle_idx = hipThreadIdx_x;
        #pragma unroll 1  // Do not unroll this loop
        for (int m = 0; m < BUTTERFLY_NUM_MODULES; m++) {
            // Perform butterfly within the current thread's data
            // int stride = 1 << MODULE_DEPTH;
            // int stride_pow = MODULE_DEPTH;
            int stride = 1;
            int stride_pow = 0;
            #pragma unroll  // Unroll this loop completely
            for (int d = 0; d < MODULE_DEPTH; d++) {
                #pragma unroll  // Unroll this loop completely
                for (int i = 0; i < MODULE_WIDTH / 2; i++) {
                    int idx_x = ((i << (stride_pow + 1)) & (MODULE_WIDTH - 1)) | 
                        ((i >> (MODULE_DEPTH - 1 - stride_pow)) & (stride - 1));
                    int idx_y = idx_x ^ stride;
                    T cosine = s_cosines[angle_idx];
                    T sine = s_sines[angle_idx];
                    T x0 = data[idx_x];
                    T y0 = data[idx_y];
                    T x1 = cosine * x0 + sine * y0;
                    T y1 = -sine * x0 + cosine * y0;
                    data[idx_x] = x1;
                    data[idx_y] = y1;
                    angle_idx += MODULE_WIDTH;
                }
                stride *= 2;
                stride_pow++;
            }

            if (m != BUTTERFLY_NUM_MODULES - 1) {
                // Exchange with other threads within the warp/wave
                #pragma unroll  // Unroll this loop completely
                for (int i = 1; i < MODULE_WIDTH; i++) {
                    data[i] = __shfl_xor(data[i], i);
                }
            }
        }

        // Store a row of data into global memory from registers
        #pragma unroll  // Unroll this loop completely
        for (int i = 0; i < MODULE_WIDTH; i++) {
            T *out_ptr = g_data_out + row_idx + MODULE_WIDTH * i * col_stride;
            __syncthreads();
            tmp[hipThreadIdx_y + hipThreadIdx_x * MODULE_WIDTH] = data[i];
            __syncthreads();
            // out_ptr[row_idx + hipThreadIdx_x + hipThreadIdx_y * col_stride]
            g_data_out[0]
                = tmp[hipThreadIdx_x + hipThreadIdx_y * MODULE_WIDTH];
        }
        // #pragma unroll  // Unroll this loop completely
        // for (int i = 0; i < MODULE_WIDTH; i++) {
        //     out_ptr[hipThreadIdx_x] = data[i];
        //     out_ptr += MODULE_WIDTH;
        // }

        // Update the pointer g_data_row to move to the next row
        row_idx += MODULE_WIDTH * hipGridDim_x;
    }

}

int main(int argc, char* argv[]) {
    float *h_data, *h_angles;
    // float *h_grad_data, *h_grad_angles;
    // int *h_idx_in;
    float **h_data_in;
    // float **h_data_out;
    float *d_data, *d_angles;
    float **d_data_in;
    // int *h_row_strides_in;
    // int *d_row_strides_in;
    // float **d_data_out;
    // float *d_grad_data, *d_grad_angles;
    // int *d_idx_in;
    int rounds = 100;
    long num_rows = (1 << 28) / BUTTERFLY_WIDTH;
    int num_input_cols = BUTTERFLY_WIDTH;
    int num_cols = BUTTERFLY_WIDTH * 2;
    int butterfly_depth = BUTTERFLY_DEPTH;
    long input_size = (long)num_rows * num_input_cols * sizeof(float);
    long data_size = (long)num_rows * num_cols * sizeof(float);
    long data_in_size = MODULE_WIDTH * sizeof(float *);
    // long row_strides_in_size = MODULE_WIDTH * sizeof(int);
    // long data_out_size = MODULE_WIDTH * sizeof(float *);
    int angles_size = BUTTERFLY_WIDTH / 2 * butterfly_depth * sizeof(float);
    static int device = 0;

    hipDeviceProp_t props;
    
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
    
    printf("info: allocate host mem (%6.3f MB)\n", (data_size * 2 + angles_size * 2) / 1000000.0);
    h_data = (float *)malloc(data_size);
    // h_grad_data = (float *)malloc(data_size);
    h_angles = (float *)malloc(angles_size);
    // h_grad_angles = (float *)malloc(angles_size);
    h_data_in = (float **)malloc(data_in_size);
    // h_row_strides_in = (int *)malloc(row_strides_in_size);
    // h_data_out = (float **)malloc(data_out_size);
    if (!h_data || !h_angles || !h_data_in) { //} || !h_data_out) {
        printf("Unable to allocate host memory");
        exit(1);
    }

    printf("info: allocate device mem (%6.3f MB)\n", (data_size * 2 + angles_size * 2) / 1000000.0);
    HIP_CHECK(hipMalloc(&d_data, data_size));
    // HIP_CHECK(hipMalloc(&d_grad_data, data_size));
    HIP_CHECK(hipMalloc(&d_angles, angles_size));
    // HIP_CHECK(hipMalloc(&d_grad_angles, angles_size));
    HIP_CHECK(hipMalloc(&d_data_in, data_in_size));
    // HIP_CHECK(hipMalloc(&d_row_strides_in, row_strides_in_size));
    // HIP_CHECK(hipMalloc(&d_data_out, data_out_size));

    printf("info: initialize data\n");
    double ss = 0.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_input_cols; j++) {
            // h_data[j * num_rows + i] = j * (i % 2 * 2 - 1);
            // h_data[i * num_cols + j] = 0.0;
            double x = (double)rand() / RAND_MAX;
            h_data[i + j * num_rows] = x;
            // h_grad_data[i * num_cols + j] = 0.0;
            ss += x * x;
        }
    }

    // h_data[0] = 1.0;

    // for (int j = 0; j < num_input_cols; j++) {
        // h_data[j] = (double)rand() / RAND_MAX;
    // }

    // for (int i = 0; i < num_rows; i++) {
        // h_grad_data[0 * num_rows + i] = 1.0;
    // }
    
    printf("info: initialize angles\n");
    float *angles_ptr = h_angles;
    // float *grad_angles_ptr = h_grad_angles;
    for (int i = 0; i < butterfly_depth; i++) {
        for (int j = 0; j < BUTTERFLY_WIDTH / 2; j++) {
            *angles_ptr = (float)rand() / RAND_MAX;
            angles_ptr++;

            // *grad_angles_ptr = 0.0;
            // grad_angles_ptr++;
        }
    }
    // h_angles[0] += 0.001;

    printf("info: initialize indices\n");
    for (int i = 0; i < MODULE_WIDTH; i++) {
        h_data_in[i] = d_data + i * MODULE_WIDTH * num_rows;
        // h_row_strides_in[i] = num_cols;
        // h_data_out[i] = d_data + i * MODULE_WIDTH + BUTTERFLY_WIDTH;
    }
    
    printf("info: copy Host2Device\n");
    HIP_CHECK(hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_grad_data, h_grad_data, data_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_angles, h_angles, angles_size, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_grad_angles, h_grad_angles, angles_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_data_in, h_data_in, data_in_size, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_row_strides_in, h_row_strides_in, row_strides_in_size, hipMemcpyHostToDevice));
    // HIP_CHECK(hipMemcpy(d_data_out, h_data_out, data_out_size, hipMemcpyHostToDevice));

    const dim3 blocks(num_rows / BUTTERFLY_BATCH_SIZE / BUTTERFLY_LOOP_SIZE);
    const dim3 threadsPerBlock(MODULE_WIDTH, BUTTERFLY_BATCH_SIZE);

    printf("info: launch kernel %d * (%d, %d)\n", blocks.x, threadsPerBlock.x, threadsPerBlock.y);

    auto start = high_resolution_clock::now();
	for (int i = 0; i < rounds + 1; i++) {
	    hipLaunchKernelGGL(butterfly_forward, blocks, threadsPerBlock, 0, 0, 
            d_data_in,
            d_data + BUTTERFLY_WIDTH * num_rows,
            d_angles,
            num_rows,
            BUTTERFLY_NUM_MODULES);
		if (i == 0) {
			HIP_CHECK(hipDeviceSynchronize());
			start = high_resolution_clock::now(); 
		} 
    }

    HIP_CHECK(hipDeviceSynchronize());
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	auto seconds = (float)duration.count() / 1000000.0;
	cout << "Duration: " << seconds << " (" << (input_size * rounds / seconds / 1000000000) << " GB/s, " << 
		(num_rows * butterfly_depth * BUTTERFLY_WIDTH / 2 * rounds * 8 / seconds / 1000000000000.0) << " TFlops" << endl; 


    HIP_CHECK(hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost));
    printf("Copied data back\n");

    double ss2 = 0.0;
    double ss3 = 0.0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_input_cols; j++) {
            double x = h_data[i + j * num_rows];
            ss2 += x* x;
            x = h_data[i + (j + num_input_cols) * num_rows];
            ss3 += x * x;
        }
    }
    printf("ss=%f, ss2=%f, ss3=%f\n", ss, ss2, ss3);

    // for (int i = 0; i < num_input_cols; i++) {
    //     printf("%d: old=%f, new=%f\n", i, h_data[i], h_data[i + num_input_cols]);
    // }

    // printf("output data:\n");
    // for (int i = 0; i < num_rows; i++) {
    //     printf("Row %d\n", i);
    //     for (int j = 0; j < num_cols; j++) {
    //         printf("Col %d: %f\n", j, h_data[j * num_rows + i]);
    //     }
    //     printf("\n");
    // }

    // hipLaunchKernelGGL(butterfly_brick_backward_slow, blocks, threadsPerBlock, 0, 0, 
    //     d_data,
    //     d_idx_in,
    //     BRICK_INPUT_WIDTH,
    //     d_angles, 
    //     num_rows,
    //     butterfly_depth_in,
    //     butterfly_depth_out,
    //     d_grad_data,
    //     d_grad_angles);

    // HIP_CHECK(hipDeviceSynchronize());
    // printf("info: copy Device2Host\n");
    // HIP_CHECK(hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost));
    // HIP_CHECK(hipMemcpy(h_grad_data, d_grad_data, data_size, hipMemcpyDeviceToHost));
    // HIP_CHECK(hipMemcpy(h_grad_angles, d_grad_angles, angles_size, hipMemcpyDeviceToHost));

    // printf("input data:\n");
    // for (int i = 0; i < num_rows; i++) {
    //     printf("Row %d\n", i);
    //     for (int j = 0; j < num_cols; j++) {
    //         printf("Col %d: %f\n", j, h_data[j * num_rows + i]);
    //     }
    //     printf("\n");
    // }

    // printf("input data gradient:\n");
    // for (int i = 0; i < num_rows; i++) {
    //     printf("Row %d\n", i);
    //     for (int j = 0; j < num_cols; j++) {
    //         printf("Col %d: %f\n", j, h_grad_data[j * num_rows + i]);
    //     }
    //     printf("\n");
    // }

    // printf("angles gradient:\n");
    // grad_angles_ptr = h_grad_angles;
    // for (int i = 0; i < butterfly_depth_in; i++) {
    //     for (int j = 0; j < BRICK_INPUT_WIDTH / 2; j++) {
    //         printf("Input, Depth %d, Angle %d: %f\n", i, j, *grad_angles_ptr);
    //         grad_angles_ptr++;
    //     }
    // }
    // for (int i = 0; i < butterfly_depth_out; i++) {
    //     for (int j = 0; j < BRICK_INPUT_WIDTH; j++) {
    //         printf("Output, Depth %d, Angle %d: %f\n", i, j, *grad_angles_ptr);
    //         grad_angles_ptr++;
    //     }
    // }
    printf("Done\n");
}
