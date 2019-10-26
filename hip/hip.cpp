#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <chrono> 
using namespace std::chrono; 
using namespace std;

#define ROWS_PER_THREAD 4
#define ANGLES_PER_THREAD_POW 0
#define ANGLES_PER_THREAD (1 << ANGLES_PER_THREAD_POW)
#define COLS_PER_THREAD (2 * ANGLES_PER_THREAD)

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

template <typename T>
__forceinline__ __device__ void butterfly_layer_forward(T *sponge, T cosine, T sine) {
	size_t i = hipThreadIdx_x;
	T x0 = sponge[2 * i];
	T y0 = sponge[2 * i + 1];
	// __syncthreads();
	T x1 = cosine * x0 + sine * y0;
	T y1 = -sine * x0 + cosine * y0;
	sponge[i] = x1;
	sponge[i + hipBlockDim_x] = y1;
	// sponge[i] = x0;
	// sponge[i + hipBlockDim_x] = y0;
	// __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(64, 1) sponge_forward(
	T *g_sponge, 
	const T *g_angles, 
	int num_layers,
	int rows_per_thread,
	int angles_per_thread_pow
) {
	HIP_DYNAMIC_SHARED(T, tmp)
	T sponge0_0;
	T sponge0_1;
	T sponge0_2;
	T sponge0_3;
	T sponge0_4;
	T sponge0_5;
	T sponge0_6;
	T sponge0_7;
	T sponge1_0;
	T sponge1_1;
	T sponge1_2;
	T sponge1_3;
	T sponge1_4;
	T sponge1_5;
	T sponge1_6;
	T sponge1_7;
	int angles_per_thread = 1 << angles_per_thread_pow;
	int cols_per_thread_pow = angles_per_thread_pow + 1;
	int cols_per_thread = angles_per_thread * 2;
	int stride = 1;
	int j = 0;

	// Load sponge data from global memory into registers
	// #pragma unroll
	// for (int i = 0; i < ROWS_PER_THREAD; i++) {
	// 	if (i >= rows_per_thread) break;
	// 	int offset = hipThreadIdx_x + hipBlockDim_x * COLS_PER_THREAD * (hipBlockIdx_x + i * hipGridDim_x);
	// 	sponge0[i] = g_sponge[offset];
	// 	sponge1[i] = g_sponge[offset + hipBlockDim_x];
	// }
	// Tried making `sponge` an array (2D, or group of 1Ds) but loop unrolling didn't work, so we unroll this manually:
	int offset = hipThreadIdx_x + hipBlockDim_x * COLS_PER_THREAD * hipBlockIdx_x;
	int step = hipBlockDim_x * COLS_PER_THREAD * hipGridDim_x;
	T *g_in = &g_sponge[offset];
	sponge0_0 = g_in[0];
	sponge1_0 = g_in[hipBlockDim_x];
	g_in += step;
	sponge0_1 = g_in[0];
	sponge1_1 = g_in[hipBlockDim_x];
	g_in += step;
	sponge0_2 = g_in[0];
	sponge1_2 = g_in[hipBlockDim_x];
	g_in += step;
	sponge0_3 = g_in[0];
	sponge1_3 = g_in[hipBlockDim_x];
	// g_in += step;
	// sponge0_4 = g_in[0];
	// sponge1_4 = g_in[hipBlockDim_x];
	// g_in += step;
	// sponge0_5 = g_in[0];
	// sponge1_5 = g_in[hipBlockDim_x];
	// g_in += step;
	// sponge0_6 = g_in[0];
	// sponge1_6 = g_in[hipBlockDim_x];
	// g_in += step;
	// sponge0_7 = g_in[0];
	// sponge1_7 = g_in[hipBlockDim_x];


	for (int j = 0; j < num_layers; j++) {
		stride = 1;
	// 	// Compute angle sine and cosine
		T angle = g_angles[hipThreadIdx_x + hipBlockDim_x * ANGLES_PER_THREAD * j];
		// 	T angle = 0.0;
		T cosine = cos(angle);
		T sine = sin(angle);
		T x0, y0, x1, y1;
		
		x0 = sponge0_0;
		y0 = sponge1_0;
		sponge0_0 = cosine * x0 + sine * y0;
		sponge1_0 = -sine * x0 + cosine * y0;

		x0 = sponge0_1;
		y0 = sponge1_1;
		sponge0_1 = cosine * x0 + sine * y0;
		sponge1_1 = -sine * x0 + cosine * y0;

		x0 = sponge0_2;
		y0 = sponge1_2;
		sponge0_2 = cosine * x0 + sine * y0;
		sponge1_2 = -sine * x0 + cosine * y0;

		x0 = sponge0_3;
		y0 = sponge1_3;
		sponge0_3 = cosine * x0 + sine * y0;
		sponge1_3 = -sine * x0 + cosine * y0;

		// x0 = sponge0_4;
		// y0 = sponge1_4;
		// sponge0_4 = cosine * x0 + sine * y0;
		// sponge1_4 = -sine * x0 + cosine * y0;

		// x0 = sponge0_5;
		// y0 = sponge1_5;
		// sponge0_5 = cosine * x0 + sine * y0;
		// sponge1_5 = -sine * x0 + cosine * y0;

		// x0 = sponge0_6;
		// y0 = sponge1_6;
		// sponge0_6 = cosine * x0 + sine * y0;
		// sponge1_6 = -sine * x0 + cosine * y0;

		// x0 = sponge0_7;
		// y0 = sponge1_7;
		// sponge0_7 = cosine * x0 + sine * y0;
		// sponge1_7 = -sine * x0 + cosine * y0;

		// Perform iterated Faro shuffle

	}

	// Store sponge data into global memory from registers
	T *g_out = &g_sponge[offset];
	g_out[0] = sponge0_0 + 1;
	g_out[hipBlockDim_x] = sponge1_0 + 1;
	g_out += step;
	g_out[0] = sponge0_1 + 1;
	g_out[hipBlockDim_x] = sponge1_1 + 1;
	g_out += step;
	g_out[0] = sponge0_2 + 1;
	g_out[hipBlockDim_x] = sponge1_2 + 1;
	g_out += step;
	g_out[0] = sponge0_3 + 1;
	g_out[hipBlockDim_x] = sponge1_3 + 1;
	// g_out += step;
	// g_out[0] = sponge0_4 + 1;
	// g_out[hipBlockDim_x] = sponge1_4 + 1;
	// g_out += step;
	// g_out[0] = sponge0_5 + 1;
	// g_out[hipBlockDim_x] = sponge1_5 + 1;
	// g_out += step;
	// g_out[0] = sponge0_6 + 1;
	// g_out[hipBlockDim_x] = sponge1_6 + 1;
	// g_out += step;
	// g_out[0] = sponge0_7 + 1;
	// g_out[hipBlockDim_x] = sponge1_7 + 1;
}


// // template <typename T>
// // __global__ void sponge_forward(T *data, )

int main() {
	float *h_angles, *h_data;
	float *d_angles, *d_data;
	size_t num_rows = 1<<22;
	size_t num_cols = 512;
	size_t angles_per_layer = num_cols / 2;
	size_t num_layers = 64;
	size_t size_angles = num_layers * angles_per_layer * sizeof(float);
	size_t size_row = num_cols * sizeof(float);
	size_t size_data = num_rows * size_row;
	int rounds = 10;
	int rows_per_thread = ROWS_PER_THREAD;
	int angles_per_thread_pow = 0;
	int angles_per_thread = 1 << angles_per_thread_pow;

	cout << "size_data=" << size_data << endl;
	h_angles = (float *)malloc(size_angles);
	h_data = (float *)malloc(size_data);
	HIP_CHECK(hipMalloc(&d_angles, size_angles));
	HIP_CHECK(hipMalloc(&d_data, size_data));

	for (int i = 0; i < num_rows * num_cols; i++) {
		h_data[i] = i;
	}
	for (int i = 0; i < angles_per_layer * num_layers; i++) {
		h_angles[i] = 0.01;
	}

	HIP_CHECK(hipMemcpy(d_angles, h_angles, size_angles, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_data, h_data, size_data, hipMemcpyHostToDevice));
	HIP_CHECK(hipDeviceSynchronize());


	cout << "Starting" << endl;
	const unsigned blocks = num_rows / rows_per_thread;
	const unsigned threads_per_block = angles_per_layer / angles_per_thread;

	cout << "blocks=" << blocks << ", threads_per_block=" << threads_per_block << endl;
	auto start = high_resolution_clock::now(); 
	for (int i = 0; i < rounds + 1; i++) {
		hipLaunchKernelGGL(sponge_forward, dim3(blocks), dim3(threads_per_block), size_row, 0, d_data, d_angles, num_layers, rows_per_thread, angles_per_thread_pow); 
		if (i == 1) {
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
