// #include <torch/extension.h>
#include <stdio.h>
#include "hip/hip_runtime.h"

#define HIP_CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

#define BRICK_INPUT_WIDTH_POW 2
#define BRICK_INPUT_WIDTH (1 << BRICK_INPUT_WIDTH_POW)
#define BRICK_BATCH_SIZE 16
#define BRICK_THREADS_PER_BLOCK 2
#define BRICK_MAX_INPUT_DEPTH 18
#define BRICK_MAX_OUTPUT_DEPTH 21


template <typename T>
__global__ void butterfly_brick_forward_slow(
    T *g_data,
    const int *g_idx_in,
    const int idx_out,
    T *angles, 
    int num_rows,
    int butterfly_depth_in,
    int butterfly_depth_out
) {
    __shared__ T s_data[2 * BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE];
    __shared__ T *s_data_ptr[BRICK_INPUT_WIDTH];
    int stride;
    int stride_pow;

    // Load the input data pointers from global memory into shared memory.
    // (After this loop, even though the pointers themselves will be in shared memory they will
    // still point to global memory.)
    for (int i = 0; i < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; i++) {
        s_data_ptr[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x] = g_data + num_rows * g_idx_in[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x];
    }

    // Load the batch of data from global memory into shared memory
    for (int i = 0; i < BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE / BRICK_THREADS_PER_BLOCK; i++) {
        int idx = i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        int col = idx / BRICK_BATCH_SIZE;
        int row = idx % BRICK_BATCH_SIZE;
        s_data[row * BRICK_INPUT_WIDTH * 2 + col] = s_data_ptr[col][row];
    }

    // Perform the butterfly on the input:
    //
    // We set the initial stride_pow in such a way that, by incrementing the stride_pow at each layer, when
    // we reach the activation layer the stride_pow will be at a maximum (ensuring that the two outputs of the
    // activation immediately have the opportunity to interact).
    //
    // Here we keep the data in shared memory at each step. (It would be more efficient to just keep the
    // data in registers and just use warp shuffles to exchange across threads where necessary, or maybe even
    // just use a small enough brick that the full width can fit entirely in one thread's registers, so
    // no cross-thread communication would be necessary at all.)
    for (int i = 0; i < butterfly_depth_in; i++) {
        stride_pow = (i + (BRICK_INPUT_WIDTH_POW - 1) * butterfly_depth_in) % BRICK_INPUT_WIDTH_POW;
        stride = 1 << stride_pow;
        for (int j = 0; j < BRICK_INPUT_WIDTH / 2 / BRICK_THREADS_PER_BLOCK; j++) {
            int base_angle_idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
            int angle_idx = i * BRICK_INPUT_WIDTH / 2 + base_angle_idx;
            T angle = angles[angle_idx];
            T cosine = cos(angle);
            T sine = sin(angle);
            int data_idx_x = ((base_angle_idx << (stride_pow + 1)) & (BRICK_INPUT_WIDTH - 1)) | 
                ((base_angle_idx >> (BRICK_INPUT_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
            int data_idx_y = data_idx_x ^ stride;
            int offset_x = data_idx_x;
            int offset_y = data_idx_y;
            for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
                T x0 = s_data[offset_x];
                T y0 = s_data[offset_y];
                T x1 = cosine * x0 + sine * y0;
                T y1 = -sine * x0 + cosine * y0;
                s_data[offset_x] = x1;
                s_data[offset_y] = y1;
                offset_x += BRICK_INPUT_WIDTH * 2;
                offset_y += BRICK_INPUT_WIDTH * 2;
            }
        }
    }
    
    // Perform the double-RELU activations, doubling the width.
    for (int j = 0; j < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; j++) {
        int idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
            float x = s_data[idx];
            if (x > 0.0) {
                s_data[idx + BRICK_INPUT_WIDTH] = 0.0;
            } else {
                s_data[idx + BRICK_INPUT_WIDTH] = x;
                s_data[idx] = 0.0;
            }
            idx += BRICK_INPUT_WIDTH * 2;
        }
    }

    // Perform the butterfly on the output
    for (int i = 0; i < butterfly_depth_out; i++) {
        stride_pow = (i + butterfly_depth_in + (BRICK_INPUT_WIDTH_POW - 1) * butterfly_depth_in) % BRICK_INPUT_WIDTH_POW;
        stride = 1 << stride_pow;
        for (int j = 0; j < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; j++) {
            int base_angle_idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
            int angle_idx = butterfly_depth_in * BRICK_INPUT_WIDTH / 2 + i * BRICK_INPUT_WIDTH + base_angle_idx;
            T angle = angles[angle_idx];
            T cosine = cos(angle);
            T sine = sin(angle);
            int data_idx_x = ((base_angle_idx << (stride_pow + 1)) & (BRICK_INPUT_WIDTH * 2 - 1)) | 
                ((base_angle_idx >> (BRICK_INPUT_WIDTH_POW - stride_pow)) & (stride - 1));
            int data_idx_y = data_idx_x ^ stride;
            int offset_x = data_idx_x;
            int offset_y = data_idx_y;
            for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
                T x0 = s_data[offset_x];
                T y0 = s_data[offset_y];
                T x1 = cosine * x0 + sine * y0;
                T y1 = -sine * x0 + cosine * y0;
                s_data[offset_x] = x1;
                s_data[offset_y] = y1;
                offset_x += BRICK_INPUT_WIDTH * 2;
                offset_y += BRICK_INPUT_WIDTH * 2;
            }
        }
    }

    // Store the batch of data into global memory from shared memory
    T *data_out_ptr = g_data + num_rows * idx_out;
    for (int i = 0; i < BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE / BRICK_THREADS_PER_BLOCK; i++) {
        int idx = i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        int col = idx / BRICK_BATCH_SIZE;
        int row = idx % BRICK_BATCH_SIZE;
        s_data_ptr[col][row] = s_data[row * BRICK_INPUT_WIDTH * 2 + col];
        data_out_ptr[col * num_rows + row] = s_data[row * BRICK_INPUT_WIDTH * 2 + col + BRICK_INPUT_WIDTH];
    }
}


template <typename T>
__global__ void butterfly_brick_backward_slow(
    T *g_data,
    const int *g_idx_in,
    const int idx_out,
    T *angles, 
    int num_rows,
    int butterfly_depth_in,
    int butterfly_depth_out,
    T *g_grad_data,
    T *g_grad_angles
) {
    __shared__ T s_data[2 * BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE];
    __shared__ T *s_data_ptr[BRICK_INPUT_WIDTH];
    __shared__ T s_grad_data[2 * BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE];
    __shared__ T *s_grad_data_ptr[BRICK_INPUT_WIDTH];
    __shared__ T s_grad_angles[BRICK_INPUT_WIDTH / 2 * BRICK_MAX_INPUT_DEPTH + BRICK_INPUT_WIDTH * BRICK_MAX_OUTPUT_DEPTH];
    int stride;
    int stride_pow;

    // Load the input data pointers from global memory into shared memory.
    // (After this loop, even though the pointers themselves will be in shared memory they will
    // still point to global memory.)
    for (int i = 0; i < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; i++) {
        s_data_ptr[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x] = g_data + num_rows * g_idx_in[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x];
        s_grad_data_ptr[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x] = g_grad_data + num_rows * g_idx_in[i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x];
    }

    // Load the batch of data and gradients from global memory into shared memory
    T *data_out_ptr = g_data + num_rows * idx_out;
    T *grad_data_out_ptr = g_grad_data + num_rows * idx_out;
    for (int i = 0; i < BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE / BRICK_THREADS_PER_BLOCK; i++) {
        int idx = i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        int col = idx / BRICK_BATCH_SIZE;
        int row = idx % BRICK_BATCH_SIZE;
        s_data[row * BRICK_INPUT_WIDTH * 2 + col] = s_data_ptr[col][row];
        s_data[row * BRICK_INPUT_WIDTH * 2 + col + BRICK_INPUT_WIDTH] = data_out_ptr[col * num_rows + row];
        s_grad_data[row * BRICK_INPUT_WIDTH * 2 + col] = s_grad_data_ptr[col][row];
        s_grad_data[row * BRICK_INPUT_WIDTH * 2 + col + BRICK_INPUT_WIDTH] = grad_data_out_ptr[col * num_rows + row];
    }

    // Perform the reverse butterfly on the output
    for (int i = butterfly_depth_out - 1; i >= 0; i--) {
        stride_pow = (i + butterfly_depth_in + (BRICK_INPUT_WIDTH_POW - 1) * butterfly_depth_in) % BRICK_INPUT_WIDTH_POW;
        stride = 1 << stride_pow;
        for (int j = 0; j < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; j++) {
            int base_angle_idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
            int angle_idx = butterfly_depth_in * BRICK_INPUT_WIDTH / 2 + i * BRICK_INPUT_WIDTH + base_angle_idx;
            T angle = angles[angle_idx];
            T cosine = cos(angle);
            T sine = sin(angle);
            int data_idx_x = ((base_angle_idx << (stride_pow + 1)) & (BRICK_INPUT_WIDTH * 2 - 1)) | 
                ((base_angle_idx >> (BRICK_INPUT_WIDTH_POW - stride_pow)) & (stride - 1));
            int data_idx_y = data_idx_x ^ stride;
            int offset_x = data_idx_x;
            int offset_y = data_idx_y;
            T grad_angle = 0.0;
            for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
                // Reverse the data computation
                T x1 = s_data[offset_x];
                T y1 = s_data[offset_y];
                T x0 = cosine * x1 + -sine * y1;
                T y0 = sine * x1 + cosine * y1;
                s_data[offset_x] = x0;
                s_data[offset_y] = y0;

                // Back-propagate the data gradient
                T gx1 = s_grad_data[offset_x];
                T gy1 = s_grad_data[offset_y];
                T gx0 = cosine * gx1 + -sine * gy1;
                T gy0 = sine * gx1 + cosine * gy1;
                s_grad_data[offset_x] = gx0;
                s_grad_data[offset_y] = gy0;

                // Accumulate the angle gradient
                grad_angle += -x0 * gy0 + y0 * gx0;

                offset_x += BRICK_INPUT_WIDTH * 2;
                offset_y += BRICK_INPUT_WIDTH * 2;
            }
            s_grad_angles[angle_idx] = grad_angle;
        }
    }

    // Reverse the double-RELU activations.
    for (int j = 0; j < BRICK_INPUT_WIDTH / BRICK_THREADS_PER_BLOCK; j++) {
        int idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
            s_data[idx] += s_data[idx + BRICK_INPUT_WIDTH];
            if (s_data[idx] < 0.0) {
                s_grad_data[idx] = s_grad_data[idx + BRICK_INPUT_WIDTH]; 
            }
            idx += BRICK_INPUT_WIDTH * 2;
        }
    }

    // Perform the reverse butterfly on the input:
    for (int i = butterfly_depth_in - 1; i >= 0; i--) {
        stride_pow = (i + (BRICK_INPUT_WIDTH_POW - 1) * butterfly_depth_in) % BRICK_INPUT_WIDTH_POW;
        stride = 1 << stride_pow;
        for (int j = 0; j < BRICK_INPUT_WIDTH / 2 / BRICK_THREADS_PER_BLOCK; j++) {
            int base_angle_idx = j * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
            int angle_idx = i * BRICK_INPUT_WIDTH / 2 + base_angle_idx;
            T angle = angles[angle_idx];
            T cosine = cos(angle);
            T sine = sin(angle);
            int data_idx_x = ((base_angle_idx << (stride_pow + 1)) & (BRICK_INPUT_WIDTH - 1)) | 
                ((base_angle_idx >> (BRICK_INPUT_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
            int data_idx_y = data_idx_x ^ stride;
            int offset_x = data_idx_x;
            int offset_y = data_idx_y;
            T grad_angle = 0.0;
            for (int k = 0; k < BRICK_BATCH_SIZE; k++) {
                // Reverse the data computation
                T x1 = s_data[offset_x];
                T y1 = s_data[offset_y];
                T x0 = cosine * x1 + -sine * y1;
                T y0 = sine * x1 + cosine * y1;
                s_data[offset_x] = x0;
                s_data[offset_y] = y0;

                // Back-propagate the data gradient
                T gx1 = s_grad_data[offset_x];
                T gy1 = s_grad_data[offset_y];
                T gx0 = cosine * gx1 + -sine * gy1;
                T gy0 = sine * gx1 + cosine * gy1;
                s_grad_data[offset_x] = gx0;
                s_grad_data[offset_y] = gy0;

                // Accumulate the angle gradient
                grad_angle += -x0 * gy0 + y0 * gx0;

                offset_x += BRICK_INPUT_WIDTH * 2;
                offset_y += BRICK_INPUT_WIDTH * 2;
            }
            s_grad_angles[angle_idx] = grad_angle;
        }
    }
    
    // Store the batch of data and gradients into global memory from shared memory
    for (int i = 0; i < BRICK_INPUT_WIDTH * BRICK_BATCH_SIZE / BRICK_THREADS_PER_BLOCK; i++) {
        int idx = i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        int col = idx / BRICK_BATCH_SIZE;
        int row = idx % BRICK_BATCH_SIZE;
        s_data_ptr[col][row] = s_data[row * BRICK_INPUT_WIDTH * 2 + col];
        s_grad_data_ptr[col][row] = s_grad_data[row * BRICK_INPUT_WIDTH * 2 + col];
    }

    // Accumulate the angle gradients into global memory
    int num_angles = BRICK_INPUT_WIDTH / 2 * BRICK_MAX_INPUT_DEPTH + BRICK_INPUT_WIDTH * BRICK_MAX_OUTPUT_DEPTH;
    for (int i = 0; i < (num_angles + BRICK_THREADS_PER_BLOCK - 1) / BRICK_THREADS_PER_BLOCK; i++) {
        int idx = i * BRICK_THREADS_PER_BLOCK + hipThreadIdx_x;
        if (idx < num_angles) {
            // atomicAdd(&g_grad_angles[idx], s_grad_angles[idx]);
            g_grad_angles[idx] = s_grad_angles[idx];
        }
    }
}


// void hip_square(at::Tensor data_in, at::Tensor data_out) {
//     int dimBlock = 16;
//     dim3 dimGrid(data_in.size(0) / dimBlock);

//     AT_DISPATCH_FLOATING_TYPES(data_in.type(), "hip_square", ([&] {
//         vector_square_kernel<scalar_t><<<dimGrid, dimBlock>>>(
//             data_out.data<scalar_t>(),
//             data_in.data<scalar_t>(),
//             data_in.size(0)
//         );
//         HIP_CHECK(hipGetLastError());
//     }));
// }


int main(int argc, char* argv[]) {
    float *h_data, *h_grad_data, *h_angles, *h_grad_angles;
    int *h_idx_in;
    float *d_data, *d_grad_data, *d_angles, *d_grad_angles;
    int *d_idx_in;
    int num_rows = 16;
    int num_cols = BRICK_INPUT_WIDTH * 2;
    int butterfly_depth_in = BRICK_INPUT_WIDTH_POW;
    int butterfly_depth_out = BRICK_INPUT_WIDTH_POW + 1;
    int data_size = num_rows * num_cols * sizeof(float);
    int angles_size = (BRICK_INPUT_WIDTH / 2 * butterfly_depth_in + BRICK_INPUT_WIDTH * butterfly_depth_out) * sizeof(float);
    int idx_in_size = BRICK_INPUT_WIDTH * sizeof(int);
    static int device = 0;
    hipDeviceProp_t props;
    
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
    
    printf("info: allocate host mem (%6.3f MB)\n", (data_size + angles_size + idx_in_size) / 1000000.0);
    h_data = (float *)malloc(data_size);
    h_grad_data = (float *)malloc(data_size);
    h_angles = (float *)malloc(angles_size);
    h_grad_angles = (float *)malloc(angles_size);
    h_idx_in = (int *)malloc(idx_in_size);
    if (!h_data || !h_angles || !h_idx_in) {
        printf("Unable to allocate host memory");
        exit(1);
    }

    printf("info: allocate device mem (%6.3f MB)\n", (data_size + angles_size + idx_in_size) / 1000000.0);
    HIP_CHECK(hipMalloc(&d_data, data_size));
    HIP_CHECK(hipMalloc(&d_grad_data, data_size));
    HIP_CHECK(hipMalloc(&d_angles, angles_size));
    HIP_CHECK(hipMalloc(&d_grad_angles, angles_size));
    HIP_CHECK(hipMalloc(&d_idx_in, idx_in_size));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            h_data[j * num_rows + i] = j * (i % 2 * 2 - 1);
            h_grad_data[j * num_rows + i] = 0.0;
        }
    }

    for (int i = 0; i < num_rows; i++) {
        h_grad_data[0 * num_rows + i] = 1.0;
    }
    
    float *angles_ptr = h_angles;
    float *grad_angles_ptr = h_grad_angles;
    for (int i = 0; i < butterfly_depth_in; i++) {
        for (int j = 0; j < BRICK_INPUT_WIDTH / 2; j++) {
            *angles_ptr = i * 0.01 + j*0.1;
            angles_ptr++;

            *grad_angles_ptr = 0.0;
            grad_angles_ptr++;
        }
    }
    for (int i = 0; i < butterfly_depth_out; i++) {
        for (int j = 0; j < BRICK_INPUT_WIDTH; j++) {
            *angles_ptr = i * 0.01 + j*0.1 + 0.12345;
            angles_ptr++;

            *grad_angles_ptr = 0.0;
            grad_angles_ptr++;
        }
    }

    // h_angles[0] += 0.001;

    for (int i = 0; i < BRICK_INPUT_WIDTH; i++) {
        h_idx_in[i] = i;
    }
    
    printf("info: copy Host2Device\n");
    HIP_CHECK(hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_grad_data, h_grad_data, data_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_angles, h_angles, angles_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_grad_angles, h_grad_angles, angles_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_idx_in, h_idx_in, idx_in_size, hipMemcpyHostToDevice));

    const dim3 blocks(num_rows / BRICK_BATCH_SIZE);
    const dim3 threadsPerBlock(BRICK_THREADS_PER_BLOCK);

    printf("info: launch kernel\n");
    hipLaunchKernelGGL(butterfly_brick_forward_slow, blocks, threadsPerBlock, 0, 0, 
        d_data,
        d_idx_in,
        BRICK_INPUT_WIDTH,
        d_angles, 
        num_rows,
        butterfly_depth_in,
        butterfly_depth_out);


    HIP_CHECK(hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost));
    printf("output data:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("Row %d\n", i);
        for (int j = 0; j < num_cols; j++) {
            printf("Col %d: %f\n", j, h_data[j * num_rows + i]);
        }
        printf("\n");
    }

    hipLaunchKernelGGL(butterfly_brick_backward_slow, blocks, threadsPerBlock, 0, 0, 
        d_data,
        d_idx_in,
        BRICK_INPUT_WIDTH,
        d_angles, 
        num_rows,
        butterfly_depth_in,
        butterfly_depth_out,
        d_grad_data,
        d_grad_angles);

    HIP_CHECK(hipDeviceSynchronize());
    printf("info: copy Device2Host\n");
    HIP_CHECK(hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_data, d_grad_data, data_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_grad_angles, d_grad_angles, angles_size, hipMemcpyDeviceToHost));

    printf("input data:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("Row %d\n", i);
        for (int j = 0; j < num_cols; j++) {
            printf("Col %d: %f\n", j, h_data[j * num_rows + i]);
        }
        printf("\n");
    }

    printf("input data gradient:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("Row %d\n", i);
        for (int j = 0; j < num_cols; j++) {
            printf("Col %d: %f\n", j, h_grad_data[j * num_rows + i]);
        }
        printf("\n");
    }

    printf("angles gradient:\n");
    grad_angles_ptr = h_grad_angles;
    for (int i = 0; i < butterfly_depth_in; i++) {
        for (int j = 0; j < BRICK_INPUT_WIDTH / 2; j++) {
            printf("Input, Depth %d, Angle %d: %f\n", i, j, *grad_angles_ptr);
            grad_angles_ptr++;
        }
    }
    for (int i = 0; i < butterfly_depth_out; i++) {
        for (int j = 0; j < BRICK_INPUT_WIDTH; j++) {
            printf("Output, Depth %d, Angle %d: %f\n", i, j, *grad_angles_ptr);
            grad_angles_ptr++;
        }
    }
}
