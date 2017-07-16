#include "getLastElement.h"
#include "kernelPrintf.h"



__global__ void kernelGetLastElement(int *Arr,unsigned int noArr, int *value){
	value[0]=Arr[noArr-1];
	//printf("\n Value:%d",value[0]);
}


cudaError_t getLastElement(int *d_index,unsigned int numberElementd_index,int &numberElementd_UniqueExtension){
	cudaError_t cudaStatus;
	dim3 block(512);
	dim3 grid((numberElementd_index+block.x-1)/block.x);

	int *value;
	cudaMalloc((int**)&value,1*sizeof(int));
	
	kernelGetLastElement<<<1,1>>>(d_index,numberElementd_index,value);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}
	
	cudaMemcpy(&numberElementd_UniqueExtension,value,1*sizeof(int),cudaMemcpyDeviceToHost);
	//printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);
	

Error:
	cudaFree(value);
	//cudaFree(d_index);

	return cudaStatus;	
}

/* Kernel này trả về graphid chứa embedding cuối cùng trong mảng d_ValidExtension */
__global__ void kernelGetLastElementExtension(Extension *inputArray,unsigned int noEleInputArray,int *value,int maxOfVer){
	value[0] = (inputArray[noEleInputArray-1].vgi/maxOfVer); /*Lấy global vertex id chia cho tổng số đỉnh của đồ thị (maxOfVer). Ở đây các đồ thị luôn có số lượng đỉnh bằng nhau (maxOfVer) */
}


/* Hàm này trả về graphId chứa Embedding cuối cùng */
inline cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	int *value;
	cudaMalloc((int**)&value,sizeof(int));
	/* Lấy graphId chứa embedding cuối cùng */
	kernelGetLastElementExtension<<<1,1>>>(inputArray,numberElementOfInputArray,value,maxOfVer);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize failed",cudaStatus);
		goto Error;
	}
	
	cudaMemcpy(&outputValue,value,sizeof(int),cudaMemcpyDeviceToHost);
	//printf("\n\nnumberElementd_UniqueExtension:%d",numberElementd_UniqueExtension);
	

Error:
	cudaFree(value);
	//cudaFree(d_index);

	return cudaStatus;	

}


inline cudaError_t getSizeBaseOnScanResult(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,int &noElem_d_UniqueExtension){
	cudaError_t cudaStatus;

	cudaStatus=getLastElement(d_allPossibleExtensionScanResult,noElem_allPossibleExtension,noElem_d_UniqueExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getLastElement() of getSizeBaseOnScanResult failed",cudaStatus);
		goto Error;
	}

	printf("\n noElem_d_UniqueExtension inside function:%d",noElem_d_UniqueExtension);

	int valueOfLast=0; //giá trị phần tử cuối cùng của mảng d_allPossibleExtension
	cudaStatus=getLastElement(d_allPossibleExtension,noElem_allPossibleExtension,valueOfLast);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getLastElement() of getSizeBaseOnScanResult failed",cudaStatus);
		goto Error;
	}

	printf("\nValue of Last Element:%d ",valueOfLast);
	if (valueOfLast==1){
		noElem_d_UniqueExtension=noElem_d_UniqueExtension+1;
	}


	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() of getSizeBaseOnScanResult failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}
