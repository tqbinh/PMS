#ifndef __CUDACC__  
    #define __CUDACC__
#endif
#include <device_functions.h>
#include "getAndStoreExtension.h"


__global__ void kernelPrintExtention(Extension *d_Extension,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){
		__syncthreads();
		printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);
	}

}


__global__ void kernelGetAndStoreExtension(int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,Extension *d_Extension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<numberOfElementd_O){
		if (d_O[i]!=-1){
			int j;
			int ek;
			//printf("\nThread:%d",i);	
			for(j=i+1;j<numberOfElementd_O;++j){					
				if(d_O[j]!=-1) {break;}				
			}			
			
			if (j==numberOfElementd_O) {
				ek=numberOfElementd_N;
			}
			else
			{
				ek=d_O[j];
			}
			//printf("\n[%d]:%d",i,ek);
			for(int k=d_O[i];k<ek;k++){
				//do something
				int index= k;
				d_Extension[index].vi=0;
				d_Extension[index].vj=0;
				d_Extension[index].li=d_LO[i];
				d_Extension[index].lij=d_LN[k];
				d_Extension[index].lj=d_LO[d_N[k]];
				d_Extension[index].vgi=i;
				d_Extension[index].vgj=d_N[k];
				//printf("\n[%d]:%d",i,index);
				/*printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",k,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,
					d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);*/
			}
		}
	}
}


cudaError_t getAndStoreExtension(Extension *d_Extension,int *d_O,int *d_LO,unsigned int numberOfElementd_O,int *d_N,int *d_LN,unsigned int numberOfElementd_N,unsigned int Le,unsigned int Lv){

	cudaError_t cudaStatus;
	dim3 block(1024);
	dim3 grid((numberOfElementd_O+block.x-1)/block.x);

	

	kernelGetAndStoreExtension<<<grid,block>>>(d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,d_Extension);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelGetAndStoreExtension failed",cudaStatus);
		goto labelError;
	}

	
	kernelPrintExtention<<<((numberOfElementd_N+block.x-1)/block.x),block>>>(d_Extension,numberOfElementd_N); //Số lượng phần tử của d_Extension bằng số lượng phần tử của d_N nhưng chúng có kích thước khác nhau vì mỗi phần tử của d_Extension là một cấu trúc trong khi d_N là một số int.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelPrintExtention failed",cudaStatus);
		goto labelError;
	}

labelError:

	return cudaStatus;
}
