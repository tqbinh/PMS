#include "getLastElement.h"
#include "kernelPrintf.h"

__global__ void kernelGetLastElement(int *Arr,unsigned int noArr,int *value){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noArr){
		if(i==noArr-1){
			value[0] = Arr[i];			
		}
	}
}

cudaError_t getLastElement(int *d_index,unsigned int numberElementd_index,int &numberElementd_UniqueExtension){
	cudaError_t cudaStatus;
	dim3 block(512);
	dim3 grid((numberElementd_index+block.x-1)/block.x);

	int *value;
	cudaMalloc((int**)&value,1*sizeof(int));
	
	kernelGetLastElement<<<grid,block>>>(d_index,numberElementd_index,value);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}
	
	cudaMemcpy(&numberElementd_UniqueExtension,value,1*sizeof(int),cudaMemcpyDeviceToHost);
	printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);
	

Error:
	return cudaStatus;
	cudaFree(value);
	cudaFree(d_index);
}
