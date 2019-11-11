#include <iostream>
#include <stdlib.h>
#include <chrono> 
#include "mipp.h"

using namespace std::chrono; 

using namespace std;

#define ALIGNMENT_BYTES 64

template <typename T>
void butterfly_layer_forward(
    T *data,
    long *idx,
    T *angles,
    long num_rows,
    long num_cols,
    long stride
) {
    mipp::Reg<T> x0, x1, y0, y1;
    long a = 0;
    for (long i = 0; i < num_cols;) {
        long offset_x = idx[i];
        long offset_y = idx[i ^ stride];
        T angle = angles[a];
        T cosine = cos(angle);
        T sine = sin(angle);
        T *ptr_x = &data[offset_x];
        T *ptr_y = &data[offset_y];
        for (long j = 0; j < num_rows; j += mipp::N<T>()) {
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
        i++;
        if (i & stride) {
            i += stride;
        }
    }
}

template <typename T>
void run() {
    long num_rows = 256;
    long num_cols = 256;
    long num_angles = num_cols / 2;
    T *data = (T *)aligned_alloc(ALIGNMENT_BYTES, num_rows * num_cols * sizeof(T));
    T *angles = (T *)malloc(num_angles * sizeof(T));
    long *idx = (long *)malloc(num_cols * sizeof(int));
    long rounds = 100000;

    if (!data || !angles || !idx) {
        cerr << "Unable to allocate memory" << endl;
        exit(1);
    }

    for (long i = 0; i < num_cols; i++) {
        for (long j = 0; j < num_rows; j++) {
            data[i * num_rows + j] = i;
        }
    }

    for (long i = 0; i < num_angles; i++) {
        angles[i] = i;
    }

    for (long i = 0; i < num_cols; i++) {
        idx[i] = i * num_rows;
    }

    auto start = high_resolution_clock::now();
    for (long i = 0; i < rounds; i++) {
        butterfly_layer_forward(data, idx, angles, num_rows, num_cols, 1);
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