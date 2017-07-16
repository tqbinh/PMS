#include "getSatisfyEdge.h"

__global__ void kernelGetSatisfyEdge(UniEdge *d_UniqueExtension,int indexOfSatisfyEdge,int *d_li,int *d_lij,int *d_lj,int *d_arr_labelEdge){

	d_li[0]=d_UniqueExtension[indexOfSatisfyEdge].li;
	d_lij[0]=d_UniqueExtension[indexOfSatisfyEdge].lij;
	d_lj[0]=d_UniqueExtension[indexOfSatisfyEdge].lj;

	d_arr_labelEdge[0]=d_UniqueExtension[indexOfSatisfyEdge].li;
	d_arr_labelEdge[1]=d_UniqueExtension[indexOfSatisfyEdge].lij;
	d_arr_labelEdge[2]=d_UniqueExtension[indexOfSatisfyEdge].lj;

	printf("\n d_arr_labelEdge[0]:%d", d_arr_labelEdge[0]);
	printf("\n d_arr_labelEdge[1]:%d", d_arr_labelEdge[1]);
	printf("\n d_arr_labelEdge[2]:%d", d_arr_labelEdge[2]);
}



inline cudaError_t getSatisfyEdge(UniEdge *d_UniqueExtension,int number,int indexOfSatisfyEdge,int &li,int &lij,int &lj,int* &d_arr_edgeLabel){
	cudaError_t cudaStatus;

	int *d_li=nullptr;
	int *d_lij=nullptr;
	int *d_lj=nullptr;

	cudaMalloc((void**)&d_arr_edgeLabel,sizeof(int)*3);
	cudaMemset(d_arr_edgeLabel,0,sizeof(int)*3);

	cudaStatus = cudaMalloc((void**)&d_li,sizeof(int));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_li failed");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_lij,sizeof(int));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_lij failed");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_lj,sizeof(int));
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_lj failed");
		goto Error;
	}
	

	kernelGetSatisfyEdge<<<1,1>>>(d_UniqueExtension,indexOfSatisfyEdge,d_li,d_lij,d_lj,d_arr_edgeLabel);
	cudaDeviceSynchronize();

	cudaMemcpy(&li,d_li,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&lij,d_lij,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&lj,d_lj,sizeof(int),cudaMemcpyDeviceToHost);



	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize getSatisfyEdge failed");
		goto Error;
	}
Error:
	return cudaStatus;
}
