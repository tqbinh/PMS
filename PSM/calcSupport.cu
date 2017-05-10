#include "calcSupport.h"
#include "kernelPrintf.h"

//__device__ int li,lij,lj;


__global__ void kernelCalcSupport(int li,int lij,int lj,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,int *d_F){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			int index=d_scanB_Result[i];
			d_F[index]=1;
		}		
	}
}



cudaError_t calcSupport(Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,int *d_F,unsigned int noElem_F){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	//chép dữ liệu của mảng d_UniqueExtension sang host

	Extension *h_UniqueExtension;
	h_UniqueExtension = new Extension[noElem_d_UniqueExtension];
	if(h_UniqueExtension==NULL){
		printf("\n!!!Memory Problem h_UniqueExtension");
		exit(1);
	}else{
		memset(h_UniqueExtension,0, noElem_d_UniqueExtension*sizeof(Extension));
	}

	cudaMemcpy(h_UniqueExtension,d_UniqueExtension,noElem_d_UniqueExtension*sizeof(Extension),cudaMemcpyDeviceToHost);

	for (int i=0;i<noElem_d_UniqueExtension;i++){	
		int li,lij,lj;
		li=h_UniqueExtension[i].li;
		lij=h_UniqueExtension[i].lij;
		lj=h_UniqueExtension[i].lj;		
			
		kernelCalcSupport<<<grid,block>>>(li,lij,lj,d_ValidExtension,noElem_d_ValidExtension,d_scanB_Result,d_F);
		cudaDeviceSynchronize();
		printf("\n[%d] d_F:",i);
		printInt(d_F,noElem_F);
		cudaMemset(d_F,0,noElem_F*sizeof(int));
	}
		
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize function failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
