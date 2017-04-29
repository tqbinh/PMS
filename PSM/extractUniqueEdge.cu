#include "extractUniqueEdge.h"
/*
	1. Tạo |d_N| threads mỗi threads tương ứng với 1 cạnh trong cơ sở dữ liệu
	2. Threads sẽ set value =1 tại vị trí tương ứng (Lij*Lv + Lj)
	note: nếu O[i]=-1 thì xem như đỉnh không hợp lệ, chúng ta bỏ qua cạnh liên quan đến đỉnh này.
*/
extern "C" inline cudaError_t extractUniqueEdge(int *d_O,int *d_LO,unsigned int sizeOfArrayO,int *d_N,int *d_LN, unsigned int sizeOfArrayN,
												int *d_singlePattern,unsigned int numberOfElementd_singlePattern,unsigned int Lv,unsigned int Le){
	cudaError_t cudaStatus;
	//calculate block and grid
	printf("\nd_O:");
	kernelPrintf<<<1,32>>>(d_O,sizeOfArrayO);
	cudaDeviceSynchronize();
	printf("\nd_N:");
	kernelPrintf<<<1,32>>>(d_N,sizeOfArrayN);
	cudaDeviceSynchronize();
		printf("\nd_singlePattern:");
	kernelPrintf<<<1,512>>>(d_singlePattern,numberOfElementd_singlePattern);
	cudaDeviceSynchronize();

	dim3 block(512);
	dim3 grid((sizeOfArrayO+block.x-1)/block.x);

	kernelExtractUniqueEdge<<<grid,block>>>(d_O,d_LO,sizeOfArrayO,d_N,d_LN,sizeOfArrayN,d_singlePattern,Lv,Le);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchornize failed",cudaStatus);
		goto Error;
	}
	printf("\nElements of d_singlePattern: ");
	kernelPrintf<<<grid,block>>>(d_singlePattern,numberOfElementd_singlePattern);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchornize failed",cudaStatus);
		goto Error;
	}

	//Khởi tạo biến g_odata để chứa dữ liệu kết quả của prescan
	//g_odata có kích thước bằng với kích thước của d_singlePattern
	//ban đầu g_odata chứa giá trị rác, sau khi thực thi xong prescan thì kết quả sẽ được cập nhật vào g_odata
	int *g_odata=NULL;
	cudaMalloc((int**)&g_odata,numberOfElementd_singlePattern-1);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc g_odata failed",cudaStatus);
		return cudaStatus;
	}

	/*
	prescan<<<1,8>>>(g_odata,d_singlePattern,numberOfElementd_singlePattern-1);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchornize prescan failed\n",cudaStatus);
		goto Error;
	}*/
	int loop=4;
	
	scan_bel<<<1,numberOfElementd_singlePattern>>>(d_singlePattern,loop,g_odata,numberOfElementd_singlePattern);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchornize scan_bel failed\n",cudaStatus);
		goto Error;
	}

	printf("\nElements of g_odata: ");
	kernelPrintf<<<grid,block>>>(g_odata,numberOfElementd_singlePattern-1);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchornize failed",cudaStatus);
		goto Error;
	}

Error:
	cudaFree(g_odata);
	return cudaStatus;
}
