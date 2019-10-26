#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <chrono> 
using namespace std::chrono; 
using namespace std;

#define MAX_SPONGE_LAYERS 8


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
__global__ void sponge_forward(
	T *sponge_in, 
	T *angles_in, 
	T *sponge_out,
	int num_layers,
	int rows_per_thread
) {
	HIP_DYNAMIC_SHARED(T, sponge)  // Declare shared-memory array "sponge" of dynamic size
	T cosines[MAX_SPONGE_LAYERS];  // Registers to hold angle cosines
	T sines[MAX_SPONGE_LAYERS];   // Registers to hold angle sines
	
	// Load angles into registers
	for (int j = 0; j < num_layers; j++) {
		T angle = angles_in[hipThreadIdx_x + j * hipBlockDim_x];
		cosines[j] = cos(angle);
		sines[j] = sin(angle);
	}

	for (int i = 0; i < rows_per_thread; i++) {
		// Load sponge data from global memory into shared memory
		int offset = hipThreadIdx_x + 2 * hipBlockDim_x * (hipBlockIdx_x + hipGridDim_x * i);
		sponge[hipThreadIdx_x] = sponge_in[offset];
		sponge[hipThreadIdx_x + hipBlockDim_x] = sponge_in[offset + hipBlockDim_x];

		for (int j = 0; j < num_layers; j++) {
			butterfly_layer_forward(sponge, cosines[j], sines[j]);
		}

		// Store sponge data into global memory from shared memory
		sponge_out[offset] = sponge[hipThreadIdx_x];
		sponge_out[offset + hipBlockDim_x] = sponge[hipThreadIdx_x + hipBlockDim_x];
	}
}


// // template <typename T>
// // __global__ void sponge_forward(T *data, )

int main() {
	float *h_angles, *h_data_in, *h_data_out;
	float *d_angles, *d_data_in, *d_data_out;
	size_t num_rows = 1<<22;
	size_t num_cols = 256;
	size_t num_angles_per_layer = num_cols / 2;
	size_t num_layers = 8;
	size_t size_angles = num_layers * num_angles_per_layer * sizeof(float);
	size_t size_row = num_cols * sizeof(float);
	size_t size_data = num_rows * size_row;
	int rounds = 10;
	int rows_per_block = 256;

	cout << "size_data=" << size_data << endl;
	h_angles = (float *)malloc(size_angles);
	h_data_in = (float *)malloc(size_data);
	h_data_out = (float *)malloc(size_data);
	HIP_CHECK(hipMalloc(&d_angles, size_angles));
	HIP_CHECK(hipMalloc(&d_data_in, size_data));
	HIP_CHECK(hipMalloc(&d_data_out, size_data));

	for (int i = 0; i < num_rows * num_cols; i++) {
		h_data_in[i] = i;
		h_data_out[i] = 0.0;
	}
	for (int i = 0; i < num_angles_per_layer * num_layers; i++) {
		h_angles[i] = 0.01;
	}

	HIP_CHECK(hipMemcpy(d_angles, h_angles, size_angles, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_data_in, h_data_in, size_data, hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(d_data_out, h_data_out, size_data, hipMemcpyHostToDevice));
	HIP_CHECK(hipDeviceSynchronize());

	const unsigned blocks = num_rows / rows_per_block;
	const unsigned threads_per_block = num_angles_per_layer;
	hipLaunchKernelGGL(sponge_forward, dim3(blocks), dim3(threads_per_block), size_row, 0, d_data_in, d_angles, d_data_out, num_layers, rows_per_block); 
	HIP_CHECK(hipDeviceSynchronize());

	cout << "Starting" << endl;
	auto start = high_resolution_clock::now(); 
	for (int i = 0; i < rounds; i++) {
		hipLaunchKernelGGL(sponge_forward, dim3(blocks), dim3(threads_per_block), size_row, 0, d_data_in, d_angles, d_data_out, num_layers, rows_per_block); 
	}
	HIP_CHECK(hipDeviceSynchronize());

	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start); 
	auto seconds = (float)duration.count() / 1000000.0;
	cout << "Duration: " << seconds << " (" << (size_data * rounds / seconds / 1000000000) << " GB/s, " << 
		(num_rows * num_cols * num_layers * 2 / seconds / 1000000000000.0) << " TFlops" << endl; 
	


	HIP_CHECK(hipMemcpy(h_data_out, d_data_out, size_data, hipMemcpyDeviceToHost));
	
	for (int i = 0; i < 8; i++) {
		printf("%d: %f\n", i, h_data_out[i]);
	}

	return 0;
}
