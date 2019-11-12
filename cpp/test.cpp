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

#define BATCH_SIZE (1 << 10)
#define COL_BLOCK_WIDTH_POW 4
#define COL_BLOCK_WIDTH (1 << COL_BLOCK_WIDTH_POW)
#define NUM_BLOCK_LAYERS (COL_BLOCK_WIDTH_POW * 2)
#define NUM_BLOCK_ANGLES (COL_BLOCK_WIDTH / 2 * NUM_BLOCK_LAYERS)
#define NUM_WORKERS 4

template <typename T>
void butterfly_layer_forward(
    T **data_in_ptrs,
    T *data_out,
    T *angles,
    long num_rows,
    long col_stride,
    long num_col_blocks
) {
    vector<thread> workers;
    atomic<long> next_col_block(0);

    // Create a pool of worker threads to iterate over the blocks of columns (each consisting 
    // of COL_BLOCK_WIDTH columns),    
    for (int w = 0; w < NUM_WORKERS; w++) {
        workers.push_back(thread([=, &next_col_block] {
            mipp::Reg<T> x0, x1, y0, y1;
            for(;;) {
                long col_block = next_col_block++;
                if (col_block >= num_col_blocks) break;
                
                // Within this column block, iterate over small batches of data rows (so that 
                // intermediate results fit in cache)
                for (long row_idx = 0; row_idx < num_rows; row_idx += BATCH_SIZE) {
                    int stride = 1;
                    int stride_pow = 0;
                    long a = col_block * NUM_BLOCK_ANGLES;
                    
                    // Iterate over the butterfly layers within this given column block and row batch:
                    for (int layer = 0; layer < NUM_BLOCK_LAYERS; layer++) {
                        // Perform one layer of butterfly within the given column block and row batch:
                        for (int j = 0; j < COL_BLOCK_WIDTH / 2; j++) {
                            int idx_x = ((j << (stride_pow + 1)) & (COL_BLOCK_WIDTH - 1)) | 
                                ((j >> (COL_BLOCK_WIDTH_POW - 1 - stride_pow)) & (stride - 1));
                            int idx_y = idx_x ^ stride;
                            T *ptr_x = data_in_ptrs[col_block * COL_BLOCK_WIDTH + idx_x] + row_idx;
                            T *ptr_y = data_in_ptrs[col_block * COL_BLOCK_WIDTH + idx_y] + row_idx;
                            T angle = angles[a];
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
                            j++;
                            if (j & stride) {
                                j += stride;
                            }
                        }
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
void run() {
    long num_rows = 1 << (24 - COL_BLOCK_WIDTH_POW);
    long num_col_blocks = 64;
    long num_input_cols = num_col_blocks * COL_BLOCK_WIDTH;
    long num_output_cols = num_col_blocks * COL_BLOCK_WIDTH;
    long num_angles = num_col_blocks * NUM_BLOCK_ANGLES;
    T *data_in = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_input_cols * sizeof(T));
    T *data_out = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_output_cols * sizeof(T));
    T **data_in_ptrs = (T **)malloc(num_input_cols * sizeof(T *));
    T *angles = (T *)malloc(num_angles * sizeof(T));
    long rounds = 1;

    printf("Data in size: %ld\n", num_rows * num_input_cols * sizeof(T));

    if (!data_in || !data_out || !data_in || !angles) {
        cerr << "Unable to allocate memory" << endl;
        exit(1);
    }

    for (long i = 0; i < num_input_cols; i++) {
        for (long j = 0; j < num_rows; j++) {
            data_in[i * num_rows + j] = i;
        }
    }

    for (long i = 0; i < num_angles; i++) {
        angles[i] = i;
    }

    for (long i = 0; i < num_input_cols; i++) {
        data_in_ptrs[i] = &data_in[i * num_rows];
    }

    auto start = high_resolution_clock::now();
    for (long i = 0; i < rounds; i++) {
        butterfly_layer_forward(data_in_ptrs, data_out, angles, num_rows, num_rows, num_col_blocks);
    }
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	auto seconds = (float)duration.count() / 1000000.0;
    auto rate = (double)rounds * num_rows * num_angles * 8 / seconds;
    cout << "Took " << seconds << "s (" << (rate / 1000000000) << " GFLOPS)" << endl;
    

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