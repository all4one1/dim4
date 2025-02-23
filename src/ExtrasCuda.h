#pragma once
#include "cuda_runtime.h"
#include <vector>

#define cudaCheckError() {                                          \
	cudaError_t e = cudaGetLastError();                                \
if (e != cudaSuccess) {\
	printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
	exit(0); \
}                                                                 \
}

double get_gpu_memory_used()
{
	size_t free_byte, total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status)
	{
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		return -1;
	}
	double free_mb = (double)free_byte / 1024.0 / 1024.0;
	double total_mb = (double)total_byte / 1024.0 / 1024.0;
	double used_mb = total_mb - free_mb;
	return used_mb;
}


size_t allocate_device_arrays(std::vector<double**> v, size_t N)
{
	size_t Nbytes = sizeof(double) * N;
	size_t total_bytes = 0;
	for (auto it : v)
	{
		cudaMalloc((void**)&(*it), Nbytes);
		cudaMemset(*it, 0, Nbytes);
		total_bytes += Nbytes;
	}
	return total_bytes;
}

struct GPU_
{
	double gpu_mem_allocated = 0;
	double mem_start = 0;
	int devID = 0;
	GPU_(int i = 0)
	{
		devID = i;
		mem_start = get_gpu_memory_used();
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, devID);
		printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);
	}

	void show_memory_usage_MB()
	{
		std::cout << "Approximately GPU memory allocated: " << gpu_mem_allocated / (1024.0 * 1024) << " MB" << std::endl;
		std::cout << "Approximately GPU memory allocated: " << get_gpu_memory_used() - mem_start << " MB" << std::endl;
	}
};


struct CudaLaunchSetup
{
	dim3 grid3d, block3d, grid2d, block2d, grid1d, block1d;
	unsigned int threads3d = 8, threads2d = 32, threads = 512;

	CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned int nz = 1)
	{
		//grid3d = dim3(
		//	(unsigned int)ceil((nx + 1.0) / threads3d),
		//	(unsigned int)ceil((ny + 1.0) / threads3d),
		//	(unsigned int)ceil((nz + 1.0) / threads3d));
		//block3d = dim3(threads3d, threads3d, threads3d);

		//grid2d = dim3(
		//	(unsigned int)ceil((nx + 1.0) / threads2d),
		//	(unsigned int)ceil((ny + 1.0) / threads2d));
		//block2d = dim3(threads2d, threads2d);

		grid1d = dim3((unsigned int)ceil((N + 0.0) / threads));
		block1d = threads;
	};
};

/**
* dest - vector of destination arrays, one or more
* src - one source array
* N - the size of
*/
void copyArrayFromHostToDevice(std::vector<double*> dest, double* src, unsigned int N)
{
	for (auto& it : dest)	
		cudaMemcpy(it, src, N * sizeof(double), cudaMemcpyHostToDevice);
};

#define	copyParametersToDevice cudaMemcpyToSymbol(dev, &host, sizeof(Configuration), 0, cudaMemcpyHostToDevice);
