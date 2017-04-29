#include "access_d_LO_from_idx_of_d_O.h"
#include "kernelPrintf.h"

inline cudaError_t access_d_LO_from_idx_of_d_O(int *d_LO,int *d_N,int sizeOfArrayN){
	printf("\n access_d_LO_from_idx_of_d_N \n");
	
	printf("\nValue of d_N:");
	kernelPrintf<<<sizeOfArrayN+32-1,32>>>(d_N,sizeOfArrayN);
	
	cudaError_t cudaStatus;
	dim3 block(32);
	dim3 grid((sizeOfArrayN+block.x-1)/block.x);
	kernelaccess_d_LO_from_idx_of_d_O<<<grid,block>>>(d_LO,d_N,sizeOfArrayN);
	cudaStatus= cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernelaccess_d_LO_from_idx_of_d_O launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//Đồng bộ dữ liệu và đảm bảo không có lỗi xảy ra	
	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
Error:
	printf("\n");
	return cudaStatus;

}
