#include "getValidExtension.h"

__global__ void kernelGetValidExtension(Extension *d_Extension,int *V,int *index,unsigned int numberElementd_Extension,Extension *d_ValidExtension){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<numberElementd_Extension){
		if(V[i]==1){

			//printf("\nV[%d]:%d, index[%d]:%d,d_Extension[%d], d_Extension[%d]:%d\n",i,V[i],i,index[i],i,i,d_Extension[i].vgi);
			d_ValidExtension[index[i]].li=d_Extension[i].li;
			d_ValidExtension[index[i]].lj=d_Extension[i].lj;
			d_ValidExtension[index[i]].lij=d_Extension[i].lij;
			d_ValidExtension[index[i]].vgi=d_Extension[i].vgi;
			d_ValidExtension[index[i]].vgj=d_Extension[i].vgj;
			d_ValidExtension[index[i]].vi=d_Extension[i].vi;
			d_ValidExtension[index[i]].vj=d_Extension[i].vj;
		}

	}

}



extern "C" inline cudaError_t getValidExtension(Extension *d_Extension,int *V,int *index,unsigned int numberElementd_Extension,Extension *d_ValidExtension){
	cudaError_t cudaStatus;
	
	//printfExtension(d_Extension,numberElementd_Extension);

	dim3 block(1024);
	dim3 grid((numberElementd_Extension+block.x)/block.x);

	kernelGetValidExtension<<<grid,block>>>(d_Extension,V,index,numberElementd_Extension,d_ValidExtension);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelGetValidExtension failed");
		goto Error;
	}

	//printfExtension(d_ValidExtension,16);

Error:
	/*cudaFree(d_Extension);
	cudaFree(index);
	cudaFree(d_ValidExtension);*/
	return cudaStatus;
}

