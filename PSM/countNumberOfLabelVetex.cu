#include "countNumberOfLabelVetex.h"
#include <iostream>
	//1.Cấp phát một mảng số nguyên có kích thước bằng với kích thước mảng d_LO gọi là d_Lv
	//2.Cấp phát |d_LO| threads
	//3.thread thứ i sẽ đọc giá trị nhãn tại vị trí d_LO[i], rồi ghi 1 vào mảng d_Lv[d_LO[i]]
	//4. Reduction mảng d_Lv để thu được các nhãn phân biệt

cudaError_t countNumberOfLabelVetex(int* d_LO,unsigned int sizeOfArrayLO, unsigned int &numberOfSaperateVertex){
	cudaError_t cudaStatus;
	numberOfSaperateVertex=0;
	size_t nBytesLv = sizeOfArrayLO*sizeof(int);
	//cấp phát mảng d_Lv trên device
	int *d_Lv;
	cudaStatus=cudaMalloc((int**)&d_Lv,nBytesLv);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"cudaMalloc d_Lv failed");
			goto Error;
		}
		else
		{
			cudaMemset(d_Lv,0,nBytesLv);
		}

		//Cấp phát threads
		dim3 block(32);
		dim3 grid((sizeOfArrayLO+block.x-1)/block.x);
		kernelCountNumberOfLabelVertex<<<grid,block>>>(d_LO,d_Lv,sizeOfArrayLO);
		
		cudaDeviceSynchronize();
		printf("\nElements of d_Lv:");
		kernelPrintf<<<grid,block>>>(d_Lv,sizeOfArrayLO);

		int* h_Lv=NULL;
		h_Lv=(int*)malloc(nBytesLv);
		if(h_Lv==NULL){
			printf("h_Lv malloc memory fail");
			exit(1);
		}
		cudaMemcpy(h_Lv,d_Lv,nBytesLv,cudaMemcpyDeviceToHost);
		cudaStatus=cudaDeviceSynchronize();
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"cudaDeviceSynchronize fail",cudaStatus);
			goto Error;
		}
		int result=0;
		sumUntilReachZero(h_Lv,sizeOfArrayLO,result);
		numberOfSaperateVertex=result;
		/*printf("Number of label is: %d ;",numberOfSaperateVertex);*/
	//unsigned int size=sizeOfArrayLO;

	//// execution configuration
	//int blocksize = 3; // initial block size
	//
	// block.x=blocksize;
	// grid.x=(size+block.x-1)/block.x;
	//printf("grid %d block %d\n",grid.x, block.x);
	//// allocate host memory
	//size_t bytes = size * sizeof(int);
	//int *h_odata = (int *) malloc(grid.x*sizeof(int));
	//if (h_odata == NULL) {
	//	printf("\nMallocation memory h_odata failure\n");
	//	exit(1);
	//}
	//else
	//{
	//	memset(h_odata,0,grid.x*sizeof(int));
	//}



	//// allocate device memory
	//int *d_odata = NULL;
	//cudaMalloc((void **) &d_odata, grid.x*sizeof(int));

	//kernelReduce<<<grid, block>>>(d_Lv, d_odata, size);
	//cudaDeviceSynchronize();
	//cudaStatus=cudaGetLastError();
	//if(cudaStatus!= cudaSuccess){
	//	fprintf(stderr,"cudaDeviceSynchronize returned error code %d after launching addKernel!\n",cudaStatus );
	//	goto Error;
	//}


	//cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

	//int gpu_sum = 0;
	//for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i];
	//printf("\ngpu Neighbored gpu_sum: %d <<<grid %d block %d>>>\n",gpu_sum,grid.x,block.x);
	//cudaDeviceSynchronize();

	//	printf("\nElements of d_Lv after reduction:");
	//	kernelPrintf<<<1,32>>>(d_Lv,sizeOfArrayLO);

	//	//kernelPrintf<<<grid,block>>>(result,1);


Error:
	cudaFree(d_Lv);
	/// free host memory
	//free(h_odata);
	// free device memory
	//cudaFree(d_odata);
	return cudaStatus;
	return cudaStatus;
}
