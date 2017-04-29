#include "markInvalidVertex.h"
#include "kernelMarkInvalidVertex.h"

//********Đếm số đỉnh song song và loại nhỏ những đỉnh nhỏ hơn minsup****
	//Nếu số đỉnh nhỏ hơn minSup thì đánh dấu đỉnh đó là -1 trong mảng O và mảng LO và các cạnh liên quan đến đỉnh đó cũng được đánh dấu là -1
	//1. Cấp phát mảng trên bộ nhớ GPU có kích thước =|LV| 
	//cấp phát vùng nhớ trên GPU
inline cudaError_t  markInvalidVertex(int *d_O,int *d_LO,int sizeOfarrayO,unsigned int minsup){
	printf("\ncall markInvalidVertex.cu\n");
	cudaError_t cudaStatus;
	int grid, block_x=32;
	int n=5;
	size_t nBytesd_labelAmount=n*sizeof(int);
	int *d_labelAmount;

	cudaStatus = cudaMalloc((int**) &d_labelAmount,nBytesd_labelAmount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return cudaStatus;
	}else{	cudaMemset(d_labelAmount,0,nBytesd_labelAmount); }

	grid=(nBytesd_labelAmount+block_x-1/block_x);
	
	printf("\nValue of d_labelAmount is set zero all:");
	kernelPrintf<<<grid,block_x>>>(d_labelAmount,n);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	printf("\nCount label in d_LO and store the result in d_labelAmount:");
	kernelCountLabelInGraphDB<<<grid,block_x>>>(d_LO,d_labelAmount,sizeOfarrayO,n);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	printf("\n");
	printf("\nValue of d_labelAmount in result:");
	kernelPrintf<<<grid,block_x>>>(d_labelAmount,n);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	printf("\nValue of d_O:");

	kernelPrintf<<<grid,block_x>>>(d_O,sizeOfarrayO);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//Những đỉnh nhỏ hơn minsup sẽ được đánh dấu là -1 trong mảng d_O
	//kernelMarkInvalidVertex(int *d_O,int *LO,unsigned int sizeLO,int *d_labelAmount,unsigned int sizeLabelAmount,unsigned int minsup=2){
	printf("\nProcess to mark vertices that have frequency less than minsup is:",minsup);
	kernelMarkInvalidVertex<<<grid,block_x>>>(d_O,d_LO,sizeOfarrayO,d_labelAmount,n,minsup);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	printf("\nCheck d_O in result:");
	kernelPrintf<<<grid,block_x>>>(d_O,sizeOfarrayO);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	cudaFree(d_labelAmount);
	return cudaStatus;
}




