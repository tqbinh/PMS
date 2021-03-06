#include "getUniqueExtension.h"
//#include "kernelPrintf.h"

/*--Hàm kernel này ánh xạ mỗi cạnh thành một vị trí duy nhất trong mảng d_allPossibleExtension
*	Và từ vị trí trong d_allPossibleExtension chúng ta có thể ánh xạ ngược lại cạnh
*	Ví dụ: từ cạnh với nhãn (Li,Lij,Lj) là (0,0,0) sẽ tương ứng với phần tử đầu tiên trong mảng d_allPossibleExtension
*/
__global__ void kernelGetUniqueExtension(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		int index=	d_ValidExtension[i].li*Lv*Le + d_ValidExtension[i].lij*Lv + d_ValidExtension[i].lj;
		d_allPossibleExtension[index]=1;
	}

}


/*	Hàm này sẽ gọi kernelGetUniqueExtension để ánh xạ cạnh thành vị trí tương ứng trong mảng d_allPossibleExtension */
cudaError_t getUniqueExtension(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,unsigned int Lv,unsigned int Le,int *d_allPossibleExtension){
	cudaError_t cudaStatus;
	
	dim3 block(1024);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelGetUniqueExtension<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	cudaDeviceSynchronize();

	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
