#define MAKE_TORCH_EXTENSION

#ifdef MAKE_TORCH_EXTENSION
#include <torch/extension.h>
#include <ATen/ATen.h>
#endif
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <chrono> 
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>
#include "mipp.h"

using namespace std::chrono; 

using namespace std;

#define ALIGNMENT_BYTES 64

#define BATCH_SIZE (1 << 9)
#define COL_BLOCK_WIDTH_POW 4
#define COL_BLOCK_WIDTH (1 << COL_BLOCK_WIDTH_POW)
#define NUM_WORKERS 4

template <typename T>
inline void cpu_butterfly_forward_layer(
    T *data,
    int *idx_in,
    T *angles,
    long col_stride,
    long col_block,
    long row_idx,
    long stride,
    long stride_pow
) {
    mipp::Reg<T> x0, x1, y0, y1;

    long a = 0; 
    // Perform one layer of butterfly within the given column block and row batch:
    for (int j = 0; j < COL_BLOCK_WIDTH / 2; j++) {
        int idx_x = ((j << (stride_pow + 1)) & (COL_BLOCK_WIDTH - 1)) | 
            ((j >> (COL_BLOCK_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
        int idx_y = idx_x ^ stride;
        int offset_x = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_x] + row_idx;
        int offset_y = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_y] + row_idx;
        T *ptr_x = data + offset_x;
        T *ptr_y = data + offset_y;
        T angle = angles[a];
        // cout << "col_block=" << col_block << ", row_idx=" << row_idx << ", layer=" << layer << ", j=" << j << ", angle=" << angle << ", idx_x=" << idx_x << ", idx_y=" << idx_y << endl;
        T cosine = cos(angle);
        T sine = sin(angle);
        
        // Perform the rotations for a given column pair, across all rows in the batch:
        for (int k = 0; k < BATCH_SIZE; k += mipp::N<T>()) {
            x0.load(ptr_x);
            y0.load(ptr_y);
            x1 = x0 * cosine + y0 * sine;
            y1 = y0 * cosine - x0 * sine;
            x1.store(ptr_x);
            y1.store(ptr_y);
            ptr_x += mipp::N<T>();
            ptr_y += mipp::N<T>();
        }
        a++;
    }
}

template <typename T>
inline void cpu_butterfly_forward_activation(
    T *data,
    int *idx_in,
    int idx_out,
    T *biases,
    T curvature,
    long num_activations,
    long col_stride,
    long col_block,
    long row_idx,
    long stride,
    long stride_pow
) {
    mipp::Reg<T> x0, x1, y0, y1, tmp;
    long a = 0; 
    T c = 0.5 / curvature;
    T c2 = c * c;

    for (int j = 0; j < num_activations; j++) {
        int idx_x = ((j << (stride_pow + 1)) & (COL_BLOCK_WIDTH - 1)) | 
            ((j >> (COL_BLOCK_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
        int idx_y = idx_x ^ stride;
        int offset_x = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_x] + row_idx;
        int offset_y = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_y] + row_idx;
        int offset_out = col_stride * (idx_out + j + col_block * num_activations) + row_idx;
        T *ptr_x = data + offset_x;
        T *ptr_y = data + offset_y;
        T *ptr_out = data + offset_out;
        T bias = biases[a];
        // cout << "col_block=" << col_block << ", row_idx=" << row_idx << ", layer=" << layer << ", j=" << j << ", angle=" << angle << ", idx_x=" << idx_x << ", idx_y=" << idx_y << endl;
        
        // Perform the activation for a given column pair, across all rows in the batch:
        for (int k = 0; k < BATCH_SIZE; k += mipp::N<T>()) {
            x0.load(ptr_x);
            y0.load(ptr_y);

            // Store y0 to the output location (this is the signal which essentially is evicted in order to
            // make room for the two activation outputs).
            y0.store(ptr_out);

            // Compute the two activation outputs (based on x0) and store them into the same locations where x0 and y0 came from.
            x0 += bias;
            tmp = mipp::sqrt(x0 * x0 + c2) - c;
            x1 = (x0 + tmp) * 0.5;
            y1 = (x0 - tmp) * 0.5;
            x1.store(ptr_x);
            y1.store(ptr_y);

            ptr_x += mipp::N<T>();
            ptr_y += mipp::N<T>();
            ptr_out += mipp::N<T>();
        }
        a++;
    }
}


template <typename T>
void cpu_butterfly_forward(
    T *data,
    int *idx_in,
    int idx_out,
    T *angles,
    T *biases,
    T curvature,
    long num_input_layers,
    long num_output_layers,
    long num_activations,
    long num_rows,
    long col_stride,
    long num_col_blocks
) {
    vector<thread> workers;
    atomic<long> next_col_block(0);
    long num_layers = num_input_layers + num_output_layers;
    int angle_block_stride = COL_BLOCK_WIDTH / 2 * num_layers;

    // Create a pool of worker threads to iterate over the blocks of columns (each consisting 
    // of COL_BLOCK_WIDTH columns),    
    for (int w = 0; w < NUM_WORKERS; w++) {
        workers.push_back(thread([=, &next_col_block] {
            for(;;) {
                long col_block = next_col_block++;
                if (col_block >= num_col_blocks) break;
                
                // Within this column block, iterate over small batches of data rows (so that 
                // intermediate results fit in cache)
                for (long row_idx = 0; row_idx < num_rows; row_idx += BATCH_SIZE) {
                    long stride = 1;
                    long stride_pow = 0;

                    // Iterate over the input butterfly layers within this given column block and row batch:
                    for (int layer = 0; layer < num_input_layers; layer++) {
                        T *angles_layer = angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * layer;
                        cpu_butterfly_forward_layer(data, idx_in, angles_layer, col_stride, col_block, row_idx, stride, stride_pow);
                        stride *= 2;
                        stride_pow++;
                        if (stride_pow == COL_BLOCK_WIDTH_POW) {
                            stride = 1;
                            stride_pow = 0;
                        }
                    }

                    // Perform activations:
                    cpu_butterfly_forward_activation(data, idx_in, idx_out, biases, curvature, num_activations, col_stride, col_block, row_idx, stride, stride_pow);

                    // Iterate over the output butterfly layers within this given column block and row batch:
                    for (int layer = 0; layer < num_output_layers; layer++) {
                        T *angles_layer = angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * (layer + num_input_layers);
                        cpu_butterfly_forward_layer(data, idx_in, angles_layer, col_stride, col_block, row_idx, stride, stride_pow);
                        stride *= 2;
                        stride_pow++;
                        if (stride_pow == COL_BLOCK_WIDTH_POW) {
                            stride = 1;
                            stride_pow = 0;
                        }
                    }
                }
            }
        }));
    }

    for_each(workers.begin(), workers.end(), [](thread &t){
        t.join();
    });
}

template <typename T>
inline void cpu_butterfly_backward_layer(
    T *data,
    T *grad_data,
    int *idx_in,
    T *angles,
    T *grad_angles,
    long col_stride,
    long col_block,
    long row_idx,
    long stride,
    long stride_pow
) {
    mipp::Reg<T> x0, x1, y0, y1, dx0, dx1, dy0, dy1, grad_angle;
    long a = 0;    

    // Perform one layer of butterfly within the given column block and row batch:
    for (int j = 0; j < COL_BLOCK_WIDTH / 2; j++) {
        int idx_x = ((j << (stride_pow + 1)) & (COL_BLOCK_WIDTH - 1)) | 
            ((j >> (COL_BLOCK_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
        int idx_y = idx_x ^ stride;
        int offset_x = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_x] + row_idx;
        int offset_y = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_y] + row_idx;
        T *ptr_x = data + offset_x;
        T *ptr_y = data + offset_y;
        T *grad_ptr_x = grad_data + offset_x;
        T *grad_ptr_y = grad_data + offset_y;
        T angle = angles[a];
        T cosine = cos(angle);
        T sine = sin(angle);
        grad_angle = 0.0;

        // Perform the rotations for a given column pair, across all rows in the batch:
        for (int k = 0; k < BATCH_SIZE; k += mipp::N<T>()) {
            x0.load(ptr_x);
            y0.load(ptr_y);
            dx0.load(grad_ptr_x);
            dy0.load(grad_ptr_y);

            // Accumulate the gradient with respect to the current angle
            grad_angle += y0*dx0 - x0*dy0;
            
            // Reverse the computation of the data
            x1 = x0 * cosine - y0 * sine;
            y1 = y0 * cosine + x0 * sine;
            x1.store(ptr_x);
            y1.store(ptr_y);

            // Backpropagate the gradient with respect to the data
            dx1 = dx0 * cosine - dy0 * sine;
            dy1 = dy0 * cosine + dx0 * sine;
            dx1.store(grad_ptr_x);
            dy1.store(grad_ptr_y);

            ptr_x += mipp::N<T>();
            ptr_y += mipp::N<T>();
            grad_ptr_x += mipp::N<T>();
            grad_ptr_y += mipp::N<T>();
        }

        grad_angles[a] += grad_angle.sum();
        a++;
    }

}

template <typename T>
inline void cpu_butterfly_backward_activation(
    T *data,
    T *grad_data,
    int *idx_in,
    int idx_out,
    T *biases,
    T *grad_biases,
    T curvature,
    long num_activations,
    long col_stride,
    long col_block,
    long row_idx,
    long stride,
    long stride_pow
) {
    mipp::Reg<T> x0, x1, y0, y1, dx0, dx1, dy0, dy1, tmp, grad_bias, one;
    long a = 0; 
    T c = 0.5 / curvature;
    T c2 = c * c;
    one = 1.0;

    for (int j = 0; j < num_activations; j++) {
        int idx_x = ((j << (stride_pow + 1)) & (COL_BLOCK_WIDTH - 1)) | 
            ((j >> (COL_BLOCK_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
        int idx_y = idx_x ^ stride;
        int offset_x = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_x] + row_idx;
        int offset_y = col_stride * idx_in[col_block * COL_BLOCK_WIDTH + idx_y] + row_idx;
        int offset_out = col_stride * (idx_out + j + col_block * num_activations) + row_idx;
        T *ptr_x = data + offset_x;
        T *ptr_y = data + offset_y;
        T *ptr_out = data + offset_out;
        T *grad_ptr_x = grad_data + offset_x;
        T *grad_ptr_y = grad_data + offset_y;
        T *grad_ptr_out = grad_data + offset_out;
        T bias = biases[a];
        // cout << "col_block=" << col_block << ", row_idx=" << row_idx << ", offset_x=" << offset_x << ", offset_y=" << offset_y << ", offset_out=" << offset_out << endl;

        // Perform the reverse activation for a given column pair, across all rows in the batch:
        grad_bias = 0.0;
        for (int k = 0; k < BATCH_SIZE; k += mipp::N<T>()) {
            x1.load(ptr_x);
            y1.load(ptr_y);
            dx1.load(grad_ptr_x);
            dy1.load(grad_ptr_y);

            // Reconstruct the activation input by summing the two activation outputs:
            x0 = x1 + y1;

            // Backpropagate the gradient with respect to the data
            tmp = x0 / mipp::sqrt(x0 * x0 + c2);
            dx0 = (dx1 * (tmp + 1.0) - dy1 * (tmp - 1.0)) * 0.5;
            x0 -= bias;
            x0.store(ptr_x);
            dx0.store(grad_ptr_x);

            // Accumulate the gradient with respect to the bias
            grad_bias += dx0;
            // cout << "grad_bias=" << grad_bias.sum() << ", dx0=" << dx0 << ", dx1=" << dx1 << ", dy0=" << dy0 << ", dy1=" << dy1 << endl;

            // Restore the evicted signal and gradient
            y0.load(ptr_out);
            y0.store(ptr_y);
            dy0.load(grad_ptr_out);
            dy0.store(grad_ptr_y);
            
            grad_ptr_x += mipp::N<T>();
            grad_ptr_y += mipp::N<T>();
            grad_ptr_out += mipp::N<T>();
            ptr_x += mipp::N<T>();
            ptr_y += mipp::N<T>();
            ptr_out += mipp::N<T>();
        }

        grad_biases[a] += grad_bias.sum();
        a++;
    }
}

template <typename T>
void cpu_butterfly_backward(
    T *data,
    T *grad_data,
    int *idx_in,
    int idx_out,
    T *angles,
    T *grad_angles,
    T *biases,
    T *grad_biases,
    T curvature,
    long num_input_layers,
    long num_output_layers,
    long num_activations,
    long num_rows,
    long col_stride,
    long num_col_blocks
) {
    vector<thread> workers;
    atomic<long> next_col_block(0);
    long num_layers = num_input_layers + num_output_layers;
    int angle_block_stride = COL_BLOCK_WIDTH / 2 * num_layers;

    // Create a pool of worker threads to iterate over the blocks of columns (each consisting 
    // of COL_BLOCK_WIDTH columns),    
    for (int w = 0; w < NUM_WORKERS; w++) {
        workers.push_back(thread([=, &next_col_block] {
            for(;;) {
                long col_block = next_col_block++;
                if (col_block >= num_col_blocks) break;
                
                // Within this column block, iterate over small batches of data rows (so that 
                // intermediate results fit in cache)
                for (long row_idx = 0; row_idx < num_rows; row_idx += BATCH_SIZE) {
                    int stride_pow = num_layers % COL_BLOCK_WIDTH_POW;
                    int stride = 1 << stride_pow;
                    
                    // Iterate over the output butterfly layers within this given column block and row batch:
                    for (int layer = num_output_layers - 1; layer >= 0; layer--) {
                        if (stride_pow == 0) {
                            stride = COL_BLOCK_WIDTH;
                            stride_pow = COL_BLOCK_WIDTH_POW;
                        }
                        stride /= 2;
                        stride_pow--;

                        T *angles_layer = angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * (layer + num_input_layers);
                        T *grad_angles_layer = grad_angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * (layer + num_input_layers);
                        cpu_butterfly_backward_layer(data, grad_data, idx_in, angles_layer, grad_angles_layer, col_stride, col_block, row_idx, stride, stride_pow);
                    }

                    // Reverse the computation of the activations (and backpropagate):
                    cpu_butterfly_backward_activation(data, grad_data, idx_in, idx_out, biases, grad_biases, curvature, num_activations, col_stride, col_block, row_idx, stride, stride_pow);

                    // Iterate over the input butterfly layers within this given column block and row batch:
                    for (int layer = num_input_layers - 1; layer >= 0; layer--) {
                        if (stride_pow == 0) {
                            stride = COL_BLOCK_WIDTH;
                            stride_pow = COL_BLOCK_WIDTH_POW;
                        }
                        stride /= 2;
                        stride_pow--;

                        T *angles_layer = angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * layer;
                        T *grad_angles_layer = grad_angles + angle_block_stride * col_block + COL_BLOCK_WIDTH / 2 * layer;
                        cpu_butterfly_backward_layer(data, grad_data, idx_in, angles_layer, grad_angles_layer, col_stride, col_block, row_idx, stride, stride_pow);
                    }
                }
            }
        }));
    }

    for_each(workers.begin(), workers.end(), [](thread &t){
        t.join();
    });
}

#ifdef MAKE_TORCH_EXTENSION
void butterfly_forward(
    at::Tensor data, 
    at::Tensor angles, 
    at::Tensor biases,
    at::Tensor indices_in, 
    int idx_out, 
    double curvature,
    int num_input_layers, 
    int num_output_layers,
    int num_activations
) {
    AT_DISPATCH_FLOATING_TYPES(data.type(), "butterfly_forward", ([&] {
        assert (data.dim() == 2);
        assert (data.strides()[1] == 1);
        long num_cols = data.size(0);
        long num_rows = data.size(1);
        assert (num_rows % BATCH_SIZE == 0);  // Relax this eventually
        assert (indices_in.dim() == 1);
        long total_butterfly_width = indices_in.size(0);
        assert (total_butterfly_width % COL_BLOCK_WIDTH == 0);
        long num_col_blocks = total_butterfly_width / COL_BLOCK_WIDTH;
        assert (angles.dim() == 2);
        assert (angles.type() == data.type());
        assert (angles.is_contiguous());
        assert (angles.size(0) == num_input_layers + num_output_layers);
        assert (angles.size(1) == num_col_blocks * COL_BLOCK_WIDTH / 2);
        assert (biases.dim() == 1);
        assert (biases.size(0) == num_activations * num_col_blocks);
        long col_stride = data.strides()[0];
        int *indices_in_ptr = indices_in.data<int>();
        scalar_t *data_ptr = data.data<scalar_t>();
        scalar_t *angles_ptr = angles.data<scalar_t>();
        scalar_t *biases_ptr = biases.data<scalar_t>();

        for (long i = 0; i < total_butterfly_width; i++) {
            long col = indices_in_ptr[i];
            assert (col >= 0 && col < num_cols);
        }
        assert (idx_out >= 0 && idx_out + num_activations * num_col_blocks <= num_cols);

        cpu_butterfly_forward(data_ptr, indices_in_ptr, idx_out, angles_ptr, biases_ptr, (scalar_t)curvature, 
            num_input_layers, num_output_layers, num_activations, num_rows, col_stride, num_col_blocks);
    }));
}

void butterfly_backward(
    at::Tensor data, 
    at::Tensor grad_data, 
    at::Tensor angles, 
    at::Tensor grad_angles, 
    at::Tensor biases,
    at::Tensor grad_biases,
    at::Tensor indices_in, 
    int idx_out,
    double curvature,
    int num_input_layers,
    int num_output_layers,
    int num_activations
) {
    AT_DISPATCH_FLOATING_TYPES(data.type(), "butterfly_forward", ([&] {
        assert (data.dim() == 2);
        assert (data.strides()[1] == 1);
        long num_cols = data.size(0);
        long num_rows = data.size(1);
        assert (num_rows % BATCH_SIZE == 0);  // Relax this eventually
        assert (indices_in.dim() == 1);
        long total_butterfly_width = indices_in.size(0);
        assert (total_butterfly_width % COL_BLOCK_WIDTH == 0);
        long num_col_blocks = total_butterfly_width / COL_BLOCK_WIDTH;
        assert (angles.dim() == 2);
        assert (angles.type() == data.type());
        assert (angles.is_contiguous());
        assert (angles.size(0) == num_input_layers + num_output_layers);
        assert (angles.size(1) == num_col_blocks * COL_BLOCK_WIDTH / 2);
        assert (grad_data.sizes() == data.sizes());
        assert (grad_angles.sizes() == angles.sizes());
        assert (biases.dim() == 1);
        assert (biases.size(0) == num_activations * num_col_blocks);
        long col_stride = data.strides()[0];
        int *indices_in_ptr = indices_in.data<int>();
        scalar_t *data_ptr = data.data<scalar_t>();
        scalar_t *grad_data_ptr = grad_data.data<scalar_t>();
        scalar_t *angles_ptr = angles.data<scalar_t>();
        scalar_t *grad_angles_ptr = grad_angles.data<scalar_t>();
        scalar_t *biases_ptr = biases.data<scalar_t>();
        scalar_t *grad_biases_ptr = grad_biases.data<scalar_t>();

        for (long i = 0; i < total_butterfly_width; i++) {
            long col = indices_in_ptr[i];
            assert (col >= 0 && col < num_cols);
        }
        assert (idx_out >= 0 && idx_out + num_activations * num_col_blocks <= num_cols);

        cpu_butterfly_backward(data_ptr, grad_data_ptr, indices_in_ptr, idx_out, angles_ptr, grad_angles_ptr, biases_ptr, grad_biases_ptr, 
            (scalar_t)curvature, num_input_layers, num_output_layers, num_activations, num_rows, col_stride, num_col_blocks);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("butterfly_forward", &butterfly_forward, "butterfly forward pass");    
    m.def("butterfly_backward", &butterfly_backward, "butterfly backward pass");
}
#endif

#ifndef MAKE_TORCH_EXTENSION
#define NUM_BLOCK_LAYERS (COL_BLOCK_WIDTH_POW * 2)
#define NUM_BLOCK_ANGLES (COL_BLOCK_WIDTH / 2 * NUM_BLOCK_LAYERS)

template <typename T>
void run() {
    long num_rows = 1 << (24 - COL_BLOCK_WIDTH_POW);
    long num_col_blocks = 16;
    long num_input_cols = num_col_blocks * COL_BLOCK_WIDTH;
    long num_angles = num_col_blocks * NUM_BLOCK_ANGLES;
    long idx_out = num_input_cols;
    long num_activations = 2;
    long num_cols = num_input_cols + num_activations * num_col_blocks;
    long num_biases = num_col_blocks * num_activations;
    T curvature = 5.0;
    T *data = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_cols * sizeof(T));
    T *grad_data = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_cols * sizeof(T));
    
    // T *data_out = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_output_cols * sizeof(T));
    // T **data_in_ptrs = (T **)malloc(num_input_cols * sizeof(T *));
    int *idx_in = (int *)malloc(num_input_cols * sizeof(int));
    T *angles = (T *)malloc(num_angles * sizeof(T));
    T *grad_angles = (T *)malloc(num_angles * sizeof(T));
    T *biases = (T *)malloc(num_biases * sizeof(T));
    T *grad_biases = (T *)malloc(num_biases * sizeof(T));
    long rounds = 1;

    printf("Data in size: %ld\n", num_rows * num_input_cols * sizeof(T));

    if (!data || !idx_in || !angles) {
        cerr << "Unable to allocate memory" << endl;
        exit(1);
    }

    for (long i = 0; i < num_cols; i++) {
        for (long j = 0; j < num_rows; j++) {
            data[i * num_rows + j] = i;
            grad_data[i * num_rows + j] = 0.0;
        }
    }

    for (long i = 0; i < num_angles; i++) {
        angles[i] = i;
        grad_angles[i] = 0.0;
    }

    for (long i = 0; i < num_biases; i++) {
        biases[i] = 0.0;
        grad_biases[i] = 0.0;
    }

    for (long i = 0; i < num_input_cols; i++) {
        // data_in_ptrs[i] = &data_in[i * num_rows];
        idx_in[i] = i;
    }



    // T *data,
    // int *idx_in,
    // int idx_out,
    // T *angles,
    // T *bias,
    // T curvature,
    // long num_input_layers,
    // long num_output_layers,
    // long num_activations,
    // long num_rows,
    // long col_stride,
    // long num_col_blocks

    // Forward pass
    auto start = high_resolution_clock::now();
    for (long i = 0; i < rounds; i++) {
        cpu_butterfly_forward(data, idx_in, idx_out, angles, biases, curvature, NUM_BLOCK_LAYERS, 0, num_activations, num_rows, num_rows, num_col_blocks);
    }
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	auto seconds = (float)duration.count() / 1000000.0;
    auto rate = (double)rounds * num_rows * num_angles * 8 / seconds;
    cout << "Forward took " << seconds << "s (" << (rate / 1000000000) << " GFLOPS)" << endl;

    start = high_resolution_clock::now();
    for (long i = 0; i < rounds; i++) {
        cpu_butterfly_backward(data, grad_data, idx_in, idx_out, angles, grad_angles, biases, grad_biases, curvature, NUM_BLOCK_LAYERS, 0, num_activations, num_rows, num_rows, num_col_blocks);
    }
	stop = high_resolution_clock::now(); 
	duration = duration_cast<microseconds>(stop - start); 
	seconds = (float)duration.count() / 1000000.0;
    rate = (double)rounds * num_rows * num_angles * 8 / seconds;
    cout << "Backward took " << seconds << "s (" << (rate / 1000000000) << " GFLOPS)" << endl;

    start = high_resolution_clock::now();
    for (long i = 0; i < rounds; i++) {
        cpu_butterfly_forward(data, idx_in, idx_out, angles, biases, curvature, NUM_BLOCK_LAYERS, 0, num_activations, num_rows, num_rows, num_col_blocks);
        cpu_butterfly_backward(data, grad_data, idx_in, idx_out, angles, grad_angles, biases, grad_biases, curvature, NUM_BLOCK_LAYERS, 0, num_activations, num_rows, num_rows, num_col_blocks);
    }
	stop = high_resolution_clock::now(); 
	duration = duration_cast<microseconds>(stop - start); 
	seconds = (float)duration.count() / 1000000.0;
    rate = (double)rounds * num_rows * num_angles * 8 / seconds;
    cout << "Combined took " << seconds << "s (" << (rate / 1000000000) << " GFLOPS)" << endl;


    // for (int j = 0; j < num_rows; j++) {
    //     for (int i = 0; i < num_cols; i++) {
    //         printf("row %d, col %d: %f\n", j, i, data[i * num_rows + j]);
    //     }
    // }
}

int main() {
    run<float>();
    return 0;
}
#endif