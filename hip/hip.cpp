#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <chrono> 
using namespace std::chrono; 
using namespace std;

#define WARP_SIZE_POW 6
#define ROWS_PER_THREAD 24
// #define ANGLES_PER_THREAD_POW 0
// #define ANGLES_PER_THREAD (1 << ANGLES_PER_THREAD_POW)
// #define COLS_PER_THREAD (2 * ANGLES_PER_THREAD)
#define COLS_PER_THREAD 1

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// #define DATA_SIZE (1 << 12)

// __global__ void vector_square(float *C_d, const float *A_d, size_t N) {
//     size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
//     size_t stride = hipBlockDim_x * hipGridDim_x ;

//     for (size_t i=offset; i<N; i+=stride) {
//         C_d[i] = A_d[i] * A_d[i];
//     }
// }

// template <typename T>
// __forceinline__ __device__ void butterfly_layer_forward(T *sponge, T cosine, T sine) {
// 	size_t i = hipThreadIdx_x;
// 	T x0 = sponge[2 * i];
// 	T y0 = sponge[2 * i + 1];
// 	// __syncthreads();
// 	T x1 = cosine * x0 + sine * y0;
// 	T y1 = -sine * x0 + cosine * y0;
// 	sponge[i] = x1;
// 	sponge[i + hipBlockDim_x] = y1;
// 	// sponge[i] = x0;
// 	// sponge[i + hipBlockDim_x] = y0;
// 	// __syncthreads();
// }

template <typename T>
__global__ void sponge_forward(
	T * __restrict__  g_sponge, 
	// const T *g_angles, 
	const T * __restrict__ g_cosines, 
	const T * __restrict__ g_sines, 
	int num_layers,
	int rows_per_thread,
	int num_cols_pow
) {
	HIP_DYNAMIC_SHARED(T, tmp)
	T sponge[ROWS_PER_THREAD];
	int num_cols = 1 << num_cols_pow;
	int stride = 1;
	int stride_pow = 0;
	int j = 0;

	// Load sponge data from global memory into registers
	int offset = hipThreadIdx_x + hipBlockDim_x * ROWS_PER_THREAD * hipBlockIdx_x;
	int step = hipBlockDim_x;
	T *g_in = &g_sponge[offset];
	#pragma unroll
	for (int i = 0; i < ROWS_PER_THREAD; i++) {
		sponge[i] = g_in[0];
		g_in += step;
	}

	for (int j = 0; j < num_layers; j++) {
		// Compute angle sine and cosine
		int idx = ((hipThreadIdx_x >> (stride_pow + 1)) << stride_pow) | (hipThreadIdx_x & (stride - 1));
	
		// int idx = hipThreadIdx_x + hipBlockDim_x * ANGLES_PER_THREAD * j;
		// T angle = g_angles[idx];
		// T cosine = cos(angle);
		// T sine = sin(angle);
		T cosine = g_cosines[idx];
		T sine = g_sines[idx];
		// T cosine = 0.99995;
		// T sine = 0.01;
		T neg_sine = -sine;
		T x0, y0, x1, y1;
		idx += hipBlockDim_x / 2;

		#pragma unroll
		for (int i = 0; i < ROWS_PER_THREAD; i++) {
			x0 = sponge[i];
			y0 = __shfl_xor(sponge[i], stride);
			if ((hipThreadIdx_x & stride) == 0){
				sponge[i] = cosine * x0 + sine * y0;
			} else {
				sponge[i] = cosine * x0 + neg_sine * y0;
			}
		}		

		stride_pow++;
		stride *= 2;
		if (stride_pow == WARP_SIZE_POW || j == num_layers - 1) {
			// Perform iterated Faro shuffle
			int srcIdx = ((hipThreadIdx_x << stride_pow) & (num_cols - 1))| (hipThreadIdx_x >> (num_cols_pow - stride_pow));
			for (int i = 0; i < ROWS_PER_THREAD; i++) {
				tmp[hipThreadIdx_x] = sponge[i];
				__syncthreads();
				sponge[i] = tmp[srcIdx];
				__syncthreads();
			}

			stride_pow = 0;
			stride = 1;
		}
		// __syncthreads();
	}

	// Store sponge data into global memory from registers
	T *g_out = &g_sponge[offset];
	#pragma unroll
	for (int i = 0; i < ROWS_PER_THREAD; i++) {
		g_out[0] = sponge[i];
		g_out += step;
	}
}


// // template <typename T>
// // __global__ void sponge_forward(T *data, )

int main() {
	// float *h_angles;
	float *h_cosines;
	float *h_sines;
	float *h_data;
	// float *d_angles;
	float *d_cosines;
	float *d_sines;
	float *d_data;
	size_t num_rows = 1 << 23;
	size_t num_cols_pow = 6;
	size_t num_cols = 1 << num_cols_pow;
	size_t angles_per_layer = num_cols / 2;
	size_t num_layers = 512;
	size_t size_angles = num_layers * angles_per_layer * sizeof(float);
	size_t size_row = num_cols * sizeof(float);
	size_t size_data = num_rows * size_row;
	int rounds = 10;
	int rows_per_thread = ROWS_PER_THREAD;
	int angles_per_thread_pow = 0;
	int angles_per_thread = 1 << angles_per_thread_pow;

	cout << "size_data=" << size_data << endl;
	// h_angles = (float *)malloc(size_angles);
	h_cosines = (float *)malloc(size_angles);
	h_sines = (float *)malloc(size_angles);
	h_data = (float *)malloc(size_data);
	if (!h_cosines || !h_sines || !h_data) {
		cout << "Failed to allocate host memory" << endl;
		return 1;
	}
	// HIP_CHECK(hipMalloc(&d_angles, size_angles));
	HIP_CHECK(hipMalloc(&d_cosines, size_angles));
	HIP_CHECK(hipMalloc(&d_sines, size_angles));
	HIP_CHECK(hipMalloc(&d_data, size_data));

	for (int i = 0; i < num_rows * num_cols; i++) {
		h_data[i] = i;
	}
	for (int i = 0; i < angles_per_layer * num_layers; i++) {
		// h_angles[i] = 0.01;
		h_cosines[i] = cos(0.01);
		h_sines[i] = sin(0.01);
	}

	// HIP_CHECK(hipMemcpy(d_angles, h_angles, size_angles, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_cosines, h_cosines, size_angles, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_sines, h_sines, size_angles, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_data, h_data, size_data, hipMemcpyHostToDevice));
	HIP_CHECK(hipDeviceSynchronize());


	cout << "Starting" << endl;
	const unsigned blocks = num_rows / rows_per_thread;
	const unsigned threads_per_block = num_cols;

	cout << "blocks=" << blocks << ", threads_per_block=" << threads_per_block << endl;
	auto start = high_resolution_clock::now(); 
	for (int i = 0; i < rounds + 1; i++) {
		// hipLaunchKernelGGL(sponge_forward, dim3(blocks), dim3(threads_per_block), size_row, 0, d_data, d_angles, num_layers, rows_per_thread, angles_per_thread_pow); 
		hipLaunchKernelGGL(sponge_forward, dim3(blocks), dim3(threads_per_block), size_row, 0, d_data, d_cosines, d_sines, num_layers, rows_per_thread, num_cols_pow); 
		if (i == 0) {
			HIP_CHECK(hipDeviceSynchronize());
			start = high_resolution_clock::now(); 
		}
	}
	HIP_CHECK(hipDeviceSynchronize());

	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	auto seconds = (float)duration.count() / 1000000.0;
	cout << "Duration: " << seconds << " (" << (size_data * rounds / seconds / 1000000000) << " GB/s, " << 
		(num_rows * num_cols * num_layers * rounds * 4 / seconds / 1000000000000.0) << " TFlops" << endl; 
	


	HIP_CHECK(hipMemcpy(h_data, d_data, size_data, hipMemcpyDeviceToHost));
	
	for (int i = 0; i < 8; i++) {
		printf("%d: %f\n", i, h_data[i]);
	}

	return 0;
}
