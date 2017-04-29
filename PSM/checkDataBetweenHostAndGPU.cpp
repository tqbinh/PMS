#include "checkDataBetweenHostAndGPU.h"

//********kiểm tra đảm bảo dữ liệu ở GPU giống với Host********
inline cudaError_t  checkDataBetweenHostAndGPU(int *d_O,int *d_LO,int *d_N,int *d_LN,unsigned int sizeOfarrayO,unsigned int noDeg,int *arrayO,int *arrayLO,int* arrayN,int *arrayLN,size_t nBytesO,size_t nBytesLO,size_t nBytesN,size_t nBytesLN)
{
	cudaError_t cudaStatus;
	int* host_O=new int[sizeOfarrayO];
	if(host_O==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayLO");
		exit(1);
	}

	int* host_LO=new int[sizeOfarrayO];
	if(host_LO==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem host_LO");
		exit(1);
	}
	int* host_N=new int[noDeg];
	if(host_N==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem host_N");
		exit(1);
	}
	int* host_LN=new int[noDeg];
	if(host_LN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem host_LN");
		exit(1);
	}

	//chép dữ liệu từ GPU sang host
	cudaStatus = cudaMemcpy(host_O,d_O,nBytesO,cudaMemcpyDeviceToHost);	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(host_LO,d_LO,nBytesLO,cudaMemcpyDeviceToHost);	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(host_N,d_N,nBytesN,cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(host_LN,d_LN,nBytesLN,cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Đồng bộ dữ liệu và đảm bảo không có lỗi xảy ra	
	cudaStatus= cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


	//so sánh
	if(!checkArray(arrayO,host_O,sizeOfarrayO)){
		printf("OrrayO does not match. Please, check gain\n");
		goto Error;
	}

	if(!checkArray(arrayLO,host_LO,sizeOfarrayO)){
		printf("OrrayLO does not match. Please, check gain\n");
		goto Error;
	}

	if(!checkArray(arrayN,host_N,noDeg)){
		printf("OrrayN does not match. Please, check gain\n");
		goto Error;
	}

	if(!checkArray(arrayLN,host_LN,noDeg)){
		printf("OrrayLN does not match. Please, check gain\n");
		goto Error;	
	}

Error:
	delete[] host_O;
	delete[] host_LO;
	delete[] host_N;
	delete[] host_LN;

	return cudaStatus;
}