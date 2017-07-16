#include "getExtensionFromEmbedding.h"

cudaError_t getValidExtensionFromEmbeding(Extension *&d_arrE,int &numberElement_d_arrE,struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,unsigned int maxOfVer,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,int lastColumn){
	cudaError_t cudaStatus;
	//Có bao nhiêu embedding n thì tạo bấy nhiêu thread để xử lý embedding tương ứng
	//printf("\n number of embedding:%d",n);
	//dim3 block(1024);
	//dim3 grid((n+block.x-1)/block.x);
	
	//cần tạo ra bao nhiêu Ext để lưu kết quả trả về và mỗi Ext có kích thước là bao nhiêu

	if(indexOfQ==lastColumn){
		//printf("\nThis is the last Q column");
		getValidForwardExtensionFromTheLastQ(d_arrE,numberElement_d_arrE,device_arr_Q,indexOfQ,dH,n,maxOfVer,d_O,d_LO,d_N,d_LN,numberOfElementd_O,numberOfElementd_N);
	}
	else
	{
		printf("\nThis is not last Q column");
	}



	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize getValidExtensionFromEmbedding failed");
		goto Error;
	}
Error:

	return cudaStatus;
}
