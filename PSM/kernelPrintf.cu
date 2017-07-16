#include "kernelPrintf.h"


//__device__ void __syncthreads(void);
__global__ void kernelPrintf(int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){			
		printf("[%d]:%d ; ",i,O[i]);
	}

}


cudaError_t printInt(int* d_array,int noElem_d_Array){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_Array+block.x-1)/block.x);

	kernelPrintf<<<grid,block>>>(d_array,noElem_d_Array);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintInt failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}

__global__ void kernelprintUnsignedInt(unsigned int *O,int sizeO){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	
	if(i<sizeO){			
		printf("[%d]:%d ; ",i,O[i]);
	}

}

inline cudaError_t printUnsignedInt(unsigned int* d_array,int noElem_d_Array){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_Array+block.x-1)/block.x);

	kernelprintUnsignedInt<<<grid,block>>>(d_array,noElem_d_Array);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintInt failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}



__global__ void kernelPrintFloat(float* A,int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){
		printf("[%d]:%.0f ;",i,A[i]);
	}

}

cudaError_t printFloat(float* d_array,int numberElementOfArray){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((numberElementOfArray+block.x-1)/block.x);

	kernelPrintFloat<<<grid,block>>>(d_array,numberElementOfArray);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintExtention failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}



__global__ void kernelPrintExtention(Extension *d_Extension,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){		
		printf("\n[%d]: DFS code:(%d,%d,%d,%d,%d)  (vgi,vgj):(%d,%d)\n",i,d_Extension[i].vi,d_Extension[i].vj,d_Extension[i].li,d_Extension[i].lij,d_Extension[i].lj,d_Extension[i].vgi,d_Extension[i].vgj);
	}

}


cudaError_t printfExtension(Extension *d_E,unsigned int noElem_d_E){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_E+block.x-1)/block.x);

	kernelPrintExtention<<<grid,block>>>(d_E,noElem_d_E);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintExtention failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}


__global__ void kernelPrintUniEdge(UniEdge *d_UniqueEdge,unsigned int n){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n){		
		int li = d_UniqueEdge[i].li;
		int lij = d_UniqueEdge[i].lij;
		int lj = d_UniqueEdge[i].lj;
		printf("\n Edge %d: (%d,%d,%d)",i,li,lij,lj);		
	}
}


cudaError_t printfUniEdge(UniEdge *d_E,unsigned int noElem_d_E){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_d_E+block.x-1)/block.x);

	kernelPrintUniEdge<<<grid,block>>>(d_E,noElem_d_E);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintUniEdge failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}

__global__ void kernelPrintEmbedding(struct_Embedding *d_Embedding,int noElem_Embedding){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if (i<noElem_Embedding){		
		printf("\n[%d]: (idx:%d, vid:%d)",i,d_Embedding[i].idx,d_Embedding[i].vid);
	}
}


cudaError_t printEmbedding(struct_Embedding *d_Embedding,int noElem_Embedding){
	cudaError cudaStatus;

	dim3 block(1024);
	dim3 grid((noElem_Embedding+block.x-1)/block.x);

	kernelPrintEmbedding<<<grid,block>>>(d_Embedding,noElem_Embedding);
	cudaDeviceSynchronize();

	cudaStatus=cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\nkernelPrintEmbedding failed");
		goto Error;
	}
Error:
	
	return cudaStatus;
}
