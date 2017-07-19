#pragma once
#include "header.h"

//kernel khởi tạo bộ nhớ và tạo nội dung cho dQ
__global__ void kernelInitializeDataEmbedding(Embedding *dQ,int sizedQ){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<sizedQ){
		dQ[i].idx=i;
		dQ[i].vid=i+100;
	}

}

//Hàm khởi tạo bộ nhớ và tạo nội dung cho dQ
inline cudaError_t createEmbeddingElement(Embedding *&dQ,int sizedQ,int &first){
	cudaError_t cudaStatus;

	//Khởi tạo bộ nhớ cho dQ1 trên device
	size_t nBytes = sizedQ*sizeof(Embedding);
	cudaStatus=cudaMalloc((void**)&dQ,nBytes);
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dQ of createEmbeddingElement failed",cudaStatus);
		goto Error;
	}
	if(first==0)
	{
		cudaMemset(dQ,-1,nBytes);
		++first;
		return cudaStatus;
	}

	//Khởi tạo dữ liệu bất kỳ cho dQ
	dim3 block(blocksize);
	dim3 grid((sizedQ + block.x - 1)/block.x);
	kernelInitializeDataEmbedding<<<grid,block>>>(dQ,sizedQ);



	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize of createEmbeddingElement failed",cudaStatus);
		goto Error;
	}
Error:

	return cudaStatus;
}


__global__ void kernelPrint(Embedding *dQ,int sizedQ){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<sizedQ){

		printf("\n Thread %d: %p (idx:%d,vid:%d) (%p,%p)",i,dQ,dQ[i].idx,dQ[i].vid,&(dQ[i].idx),&(dQ[i].vid));
	}
}


//Hàm in nội dung Embedding *dQ khi biết kích thước
inline cudaError_t print(Embedding *dQ,int sizedQ){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((sizedQ + block.x - 1)/block.x);

	kernelPrint<<<grid,block>>>(dQ,sizedQ);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize of createEmbeddingElement failed",cudaStatus);
		goto Error;
	}


Error:

	return cudaStatus;
}

//kernel In phần tử Embedding **pdQ
__global__ void kernelPrint(Embedding **pdQ,int *d_arrSizedQ,int *d_arrPrevQ,int sizepdQ){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sizepdQ){
		printf("\n Thread %d: Value of pdQ:%p",i,pdQ+i);
		Embedding *dQ = pdQ[i];
		//printf("\n Thread %d: PrevQ: %d",i,d_arrPrevQ[i]);
		int prevQ = d_arrPrevQ[i];
		for (int j = 0; j < d_arrSizedQ[i]; j++)
		{
			printf("\n i=%d %p PrevQ:%d (idx:%d, vid:%d) ",i,dQ,prevQ,dQ[j].idx,dQ[j].vid);
		}
	}
}

//Hàm in phần tử Embedding **pdQ khi biết kích thước của dQ trong mảng h_arrSizedQ tương ứng
inline cudaError_t print(Embedding **pdQ,int *h_arrSizedQ,int *d_arrPrevQ,int sizepdQ){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((sizepdQ+block.x-1)/block.x);
	printf("\n\n Array pdQ:\n");
	kernelPrint<<<grid,block>>>(pdQ,h_arrSizedQ,d_arrPrevQ,sizepdQ);
	printf("\n");
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize of kernelPrint failed",cudaStatus);
		goto Error;
	}


Error:

	return cudaStatus;
}

//kernel lấy pointer của dQ lưu vào pdQ
__global__ void kernelgetPointer(Embedding **pdQ,Embedding *dQ){
	*pdQ=dQ;

}


__global__ void kernelCopyEmbedding(Embedding **pdQ,int sizepdQ,Embedding **d_temp){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<sizepdQ){
		d_temp[i]=pdQ[i];
	}
}

__global__ void kernelPrintDoubleEmbedding(Embedding **d_temp,int newsize){
	int i=blockDim.x * blockIdx.x + threadIdx.x;
	if (i<newsize){
		printf("\n Thread %d: %p",i,d_temp[i]);

	}

}

__global__ void kernelCopyLastEmbedding(Embedding **d_temp,Embedding *dQ,int newsize){
	d_temp[newsize-1]=dQ;
}


//Hàm lấy pointer của phần tử Embedding *dQ bằng hàm cudaMemcpy
inline cudaError_t getPointer(Embedding **&pdQ,int &sizepdQ,Embedding *dQ){
	cudaError_t cudaStatus;

	//
	int currentsize = sizepdQ;
	int newsize = ++sizepdQ;
	if (currentsize==0){
		cudaStatus=cudaMalloc((void**)&pdQ,newsize*sizeof(Embedding*));
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n cudaMalloc pdQ failed",cudaStatus);
			goto Error;
		}
		else
		{
			kernelgetPointer<<<1,1>>>(pdQ,dQ);
			cudaDeviceSynchronize();
		}
		goto Error;
	}

	//Khởi tạo mảng tạm 
	Embedding **d_temp=nullptr;
	size_t nBytes=newsize*sizeof(Embedding*);
	cudaStatus=cudaMalloc((void**)&d_temp,nBytes);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_temp in getPointer failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(d_temp,0,nBytes);
	}

	//chép mảng hiện tại 
	kernelCopyEmbedding<<<1,currentsize>>>(pdQ,currentsize,d_temp);
	cudaDeviceSynchronize();

	//chép phần tử cần thêm vào cuối mảng d_temp
	kernelCopyLastEmbedding<<<1,1>>>(d_temp,dQ,newsize);
	cudaDeviceSynchronize();

	//Hiển thị nội dung mảng d_temp

	kernelPrintDoubleEmbedding<<<1,sizepdQ>>>(d_temp,sizepdQ);
	cudaDeviceSynchronize();


	//Cấp phát lại bộ nhớ cho mảng chính với kích thước lớn hơn 1 và chép mảng d_temp vào mảng chính
	cudaFree(pdQ);

	cudaStatus=cudaMalloc((void**)&pdQ,nBytes);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc pdQ in getPointer failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(pdQ,-1,nBytes);
	}

	kernelCopyEmbedding<<<1,sizepdQ>>>(d_temp,sizepdQ,pdQ);
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize kernelgetPointer of getPointer failed",stderr);
		goto Error;
	}

Error:

	return cudaStatus;
}

//kernel chép dữ liệu kiểu int từ device sang device
__global__ void kernelCopyInt(int *d_arrSizedQ,int *d_tempArrSizedQ,int currentSize){
	int i=blockIdx.x * blockDim.x + threadIdx.x;
	if(i<currentSize){
		d_tempArrSizedQ[i]=d_arrSizedQ[i];
	}
}


__global__ void kernelCopyLastInt(int *temp,int *d_tempArrSizedQ,int newsize){
	d_tempArrSizedQ[newsize-1]=*temp;
}


inline cudaError_t copyDeviceToDeviceInt(int *d_FromIntArray,int *d_ToIntArray,int size){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((size + block.x)/block.x);

	kernelCopyInt<<<grid,block>>>(d_FromIntArray,d_ToIntArray,size);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize in copyDeviceToDeviceInt failed",stderr);
		goto Error;
	}

Error:

	return cudaStatus;
}


//Trả về mảng kích thước là một h_arrSizedQ trên device
inline cudaError_t getSizedQ(int *&d_arrSizedQ,int &sized_arrSizedQ,int sizedQ){
	cudaError_t cudaStatus;

	//Mở rộng kích thước mảng d_arrSizedQ
	int currentSize = sized_arrSizedQ;
	int newsize =++sized_arrSizedQ;
	if(currentSize==0){
		cudaStatus = cudaMalloc((void**)&d_arrSizedQ,newsize*sizeof(int));
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n cudaMalloc d_arrSizedQ in getPointer failed",cudaStatus);
			goto Error;
		}
		else
		{
			cudaMemcpy(d_arrSizedQ,&sizedQ,sizeof(int),cudaMemcpyHostToDevice);
		}


		goto Error;
	}


	size_t nBytes = newsize*sizeof(int);
	int *d_tempArrSizedQ;
	cudaStatus=cudaMalloc((void**)&d_tempArrSizedQ,nBytes);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_tempArrSizedQ in getPointer failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(d_tempArrSizedQ,-1,nBytes);
	}

	//Chép mảng cũ qua mảng mới
	cudaStatus = copyDeviceToDeviceInt(d_arrSizedQ,d_tempArrSizedQ,currentSize);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n copyDeviceToDeviceInt in getSizedQ failed");
		goto Error;
	}



	//Kiểm tra thử kết quả trên mảng tạm
	//print(d_tempArrSizedQ,currentSize);

	//Tạo một biết temp để cấp phát phần tử kiểu int trên device và chép sizedQ sang biến tạm
	int * temp;
	cudaStatus=cudaMalloc((void**)&temp,sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc temp in getPointer failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemcpy(temp,&sizedQ,sizeof(int),cudaMemcpyHostToDevice);
	}


	kernelCopyLastInt<<<1,1>>>(temp,d_tempArrSizedQ,newsize);
	cudaDeviceSynchronize();

	cudaFree(d_arrSizedQ);
	cudaMalloc((void**)&d_arrSizedQ,nBytes);
	cudaMemset(d_arrSizedQ,0,nBytes);


	copyDeviceToDeviceInt(d_tempArrSizedQ,d_arrSizedQ,newsize);
	/*
	printf("\n\n value of d_arrSizedQ array on device\n");
	print(d_arrSizedQ,newsize);*/

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize getPointer failed",cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;

}

//kernel in mảng kiểu int trên device
__global__ void kernelPrintInt(int *dArray,int sizedArray){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<sizedArray){
		printf("\n Thread %d: dArray:%d",i,dArray[i]);
	}
}

//Hàm in mảng kiểu int trên device
inline cudaError_t print(int *dArray,int sizedArray){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((sizedArray + block.x -1)/block.x);
	kernelPrintInt<<<grid,block>>>(dArray,sizedArray);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize getPointer failed",cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

//Tạo và cập nhật ColumnQ
cudaError_t makeColumnQ(Embedding *dQ,int sizedQ,Embedding **&pdQ,int &sizepdQ,int *&d_arrSizedQ,int &sized_arrSizedQ,int *&d_arrPrevQ,int &sized_arrPrevQ,int iPrevQ,int &first){
	cudaError_t cudaStatus;

	//Tạo nội dung cho các phần tử của dQ

	cudaStatus=createEmbeddingElement(dQ,sizedQ,first);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaStatuscreateEmbeddingElement in makeColumnQ failed",stderr);
		goto Error;
	}

	////In nội dung dQ
	//cudaStatus=print(dQ,sizedQ);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n print of kernel.cu failed",stderr);
	//	goto Error;
	//}


	//Lấy con trỏ của dQ lưu vào pdQ
	cudaStatus = getPointer(pdQ,sizepdQ,dQ);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getPointer in makeColumnQ failed",stderr);
		goto Error;
	}

	//Lấy kích thước của dQ lưu vào mảng d_arrSizedQ
	cudaStatus = getSizedQ(d_arrSizedQ,sized_arrSizedQ,sizedQ);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getSizedQ  in makeColumnQ failed",stderr);
		goto Error;
	}

	//Lấy prevQ
	cudaStatus = getSizedQ(d_arrPrevQ,sized_arrPrevQ,iPrevQ);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getSizedQ of kernel.cu failed",stderr);
		goto Error;
	}

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in makeColumnQ failed");
		goto Error;
	}

Error:
	return cudaStatus;
}


//kernel in nội dung embedding thứ i.
__global__ void kernelPrintEmbedding(Embedding **pdQ,int *d_arrSizedQ,int *d_arrPrevQ,int sizepdQ,int firstEmbedding,int lastColumnQ){
	Embedding *dQ = pdQ[lastColumnQ];
	int vid = dQ[firstEmbedding].vid;
	int idx = dQ[firstEmbedding].idx;
	int prevQ = d_arrPrevQ[lastColumnQ];	
	printf("\n Q%d: (idx:%d, vid:%d) prevQ:%d",lastColumnQ,idx,vid,prevQ);
	while (true)
	{
		dQ=pdQ[prevQ];
		vid = dQ[idx].vid;
		idx = dQ[idx].idx;		
		printf("\n Q%d: (idx:%d, vid:%d)",prevQ,idx,vid);	

		prevQ = d_arrPrevQ[prevQ];
		if(prevQ==-1){ 
			printf("\nEnd of Embedding\n");
			return;
		}

	}
}

//In embedding thứ i. Cần phải biết cột Q cuối để truy xuất Embedding ngược về phía trước
inline cudaError_t printEmbedding(Embedding **pdQ,int *d_arrSizedQ,int *d_arrPrevQ,int sizepdQ,int firstEmbedding,int lastColumnQ){
	cudaError_t cudaStatus;

	kernelPrintEmbedding<<<1,1>>>(pdQ,d_arrSizedQ,d_arrPrevQ,sizepdQ,firstEmbedding,lastColumnQ);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printEmbedding failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

__global__ void kernelSetValueFordQ(Extension *d_ValidExtension,int noElem_d_ValidExtension,Embedding *dQ1,Embedding *dQ2,int *d_scanResult,int li,int lij,int lj){
	int i = blockDim.x *blockIdx.x +threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li &&d_ValidExtension[i].lij == lij && d_ValidExtension[i].lj){
			dQ1[d_scanResult[i]].idx=-1;
			dQ1[d_scanResult[i]].vid=d_ValidExtension[i].vgi;

			dQ2[d_scanResult[i]].idx=d_scanResult[i];
			dQ2[d_scanResult[i]].vid=d_ValidExtension[i].vgj;
		}

	}

}


inline cudaError_t createEmbeddingRoot(Embedding **&dArrPointerEmbedding,int &noElem_dArrPointerEmbedding,int *&dArrSizedQ,int &noElem_dArrSizedQ,int *&dArrPrevQ,int &noElem_dArrPrevQ,Extension *d_ValidExtension,int noElem_d_ValidExtension,int li,int lij,int lj){
	cudaError_t cudaStatus;

	//Vì đây là lần đầu tiên tạo Embedding, chúng ta tạo 2 cột Q có kích thước bằng nhau và bằng số lượng Embedding tìm thấy trong d_ValidExtension của nhãn cạnh (li,lij,lj)
	//Tạo Q1 và Q2 trên bộ nhớ device, sau đó chép địa chỉ của nó vào biến mảng dArrPointerEmbedding. Do đó, chúng ta không huỷ bộ nhớ của Q1 và Q2 sau khi gọi hàm createEmbeddingRoot.
	Embedding *Q1=nullptr;//embedding dQ.
	Embedding *Q2=nullptr;
	int sizedQ=0;


	//Tạo bảo nhiêu mảng dQ, mỗi mảng có số lượng phần tử là bao nhiêu và nội dung mảng là gì?
	//Tạo 2 mảng dQ
	/*1.Tạo mảng M có kích thước bằng với d_ValidExtension và khởi tạo giá trị cho các phần tử trong M bằng 0.*/
	int* d_M;
	cudaStatus=cudaMalloc((int**)&d_M,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		//exit(1);
		goto Error;
	}
	else
	{
		cudaMemset(d_M,0,noElem_d_ValidExtension*sizeof(int));
	}

	/*//2. Tạo noElem_d_ValidExtension threads. Mỗi thread sẽ kiểm tra phần tử tương ứng trong mảng d_ValidExtension xem có bằng cạnh (li,lij,lj) 
	Nếu bằng thì bậc vị trí tại M lên giá trị là 1*/
	//printf("\nMang d_ValidExtension");
	//printfExtension(d_ValidExtension,noElem_d_ValidExtension);
	//cudaDeviceSynchronize();
	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelMarkExtension<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_M,li,lij,lj);
	cudaDeviceSynchronize();
	/*printf("\n\nMang d_ValidExtension");
	printfExtension(d_ValidExtension,noElem_d_ValidExtension);
	cudaDeviceSynchronize();
	printf("\nMang d-M:");
	printInt(d_M,noElem_d_ValidExtension);*/

	/* 3. Exclusive Scan d_M
	Kết quả scan lưu vào mảng d_scanResult
	*/
	int* d_scanResult;
	cudaStatus=cudaMalloc((int**)&d_scanResult,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		//exit(1);
		goto Error;
	}
	else
	{
		cudaMemset(d_scanResult,0,noElem_d_ValidExtension*sizeof(int));
	}

	cudaStatus=scanV(d_M,noElem_d_ValidExtension,d_scanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanV() d_M createForwardEmbedding failed");
		//exit(1);
		goto Error;
	}

	/*
	4. Tạo mảng Q1 và Q2 có kích thước là (scanM[LastIndex]) nếu phần tử cuối cùng của d_ValidExtension không phải là (li,lij,lj).
	Ngược lại thì Q có kích thước là (scanM[LastIndex]+1). 
	Mỗi phần tử của Q có cấu trúc là {int idx, int vid}
	*/
	bool same = false;
	kernelMatchLastElement<<<1,1>>>(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,same);
	cudaDeviceSynchronize();

	int noElem_d_Q=0;

	cudaStatus=getLastElement(d_scanResult,noElem_d_ValidExtension,noElem_d_Q);

	if (same==true){
		noElem_d_Q++;
	}

	sizedQ=noElem_d_Q;

	printf("\nnoElem_d_Q1:%d",noElem_d_Q);

	//Tạo Embedding dQ1, khi đã biết kích thước của chúng	

	cudaStatus = cudaMalloc((void**)&Q1,sizedQ*sizeof(Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dQ1 in createEmbeddingRoot() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(Q1,-1,sizedQ*sizeof(Embedding));
	}

	cudaStatus = cudaMalloc((void**)&Q2,sizedQ*sizeof(Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dQ1 in createEmbeddingRoot() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(Q2,-1,sizedQ*sizeof(Embedding));
	}
	kernelSetValueFordQ<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Q1,Q2,d_scanResult,li,lij,lj);
	cudaDeviceSynchronize();

	getPointer(dArrPointerEmbedding,noElem_dArrPointerEmbedding,Q1);
	getPointer(dArrPointerEmbedding,noElem_dArrPointerEmbedding,Q2);

	int iPrevQ=-1;
	for (int j = 0; j < 2; j++)
	{
		//Lấy kích thước của dQ lưu vào mảng d_arrSizedQ
		cudaStatus = getSizedQ(dArrSizedQ,noElem_dArrSizedQ,sizedQ);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n getSizedQ  in makeColumnQ failed",stderr);
			goto Error;
		}

		//Lấy prevQ
		cudaStatus = getSizedQ(dArrPrevQ,noElem_dArrPrevQ,iPrevQ);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n getSizedQ of kernel.cu failed",stderr);
			goto Error;
		}
		iPrevQ++;
	}


	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createEmbeddingRoot() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;

}

inline cudaError_t createEmbeddingRoot1(Embedding **&dArrPointerEmbedding,int &noElem_dArrPointerEmbedding,int *&dArrSizedQ,int &noElem_dArrSizedQ,Extension *d_ValidExtension,int noElem_d_ValidExtension,int li,int lij,int lj){
	cudaError_t cudaStatus;
	//Vì đây là lần đầu tiên tạo Embedding, chúng ta tạo 2 cột Q có kích thước bằng nhau và bằng số lượng Embedding tìm thấy trong d_ValidExtension của nhãn cạnh (li,lij,lj)
	//Tạo Q1 và Q2 trên bộ nhớ device, sau đó chép địa chỉ của nó vào biến mảng dArrPointerEmbedding. Do đó, chúng ta không huỷ bộ nhớ của Q1 và Q2 sau khi gọi hàm createEmbeddingRoot.
	Embedding *Q1=nullptr;//embedding dQ.
	Embedding *Q2=nullptr;
	int sizedQ=0;


	//Tạo bảo nhiêu mảng dQ, mỗi mảng có số lượng phần tử là bao nhiêu và nội dung mảng là gì?
	//Tạo 2 mảng dQ
	/*1.Tạo mảng M có kích thước bằng với d_ValidExtension và khởi tạo giá trị cho các phần tử trong M bằng 0.*/
	int* d_M;
	cudaStatus=cudaMalloc((int**)&d_M,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		//exit(1);
		goto Error;
	}
	else
	{
		cudaMemset(d_M,0,noElem_d_ValidExtension*sizeof(int));
	}

	/*//2. Tạo noElem_d_ValidExtension threads. Mỗi thread sẽ kiểm tra phần tử tương ứng trong mảng d_ValidExtension xem có bằng cạnh (li,lij,lj) 
	Nếu bằng thì bậc vị trí tại M lên giá trị là 1*/
	//printf("\nMang d_ValidExtension");
	//printfExtension(d_ValidExtension,noElem_d_ValidExtension);
	//cudaDeviceSynchronize();
	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	kernelMarkExtension<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_M,li,lij,lj);
	cudaDeviceSynchronize();
	/*printf("\n\nMang d_ValidExtension");
	printfExtension(d_ValidExtension,noElem_d_ValidExtension);
	cudaDeviceSynchronize();
	printf("\nMang d-M:");
	printInt(d_M,noElem_d_ValidExtension);*/

	/* 3. Exclusive Scan d_M
	Kết quả scan lưu vào mảng d_scanResult
	*/
	int* d_scanResult;
	cudaStatus=cudaMalloc((int**)&d_scanResult,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		//exit(1);
		goto Error;
	}
	else
	{
		cudaMemset(d_scanResult,0,noElem_d_ValidExtension*sizeof(int));
	}

	cudaStatus=scanV(d_M,noElem_d_ValidExtension,d_scanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanV() d_M createForwardEmbedding failed");
		//exit(1);
		goto Error;
	}

	/*
	4. Tạo mảng Q1 và Q2 có kích thước là (scanM[LastIndex]) nếu phần tử cuối cùng của d_ValidExtension không phải là (li,lij,lj).
	Ngược lại thì Q có kích thước là (scanM[LastIndex]+1). 
	Mỗi phần tử của Q có cấu trúc là {int idx, int vid}
	*/
	bool same = false;
	kernelMatchLastElement<<<1,1>>>(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,same);
	cudaDeviceSynchronize();

	int noElem_d_Q=0;

	cudaStatus=getLastElement(d_scanResult,noElem_d_ValidExtension,noElem_d_Q);

	if (same==true){
		noElem_d_Q++;
	}

	sizedQ=noElem_d_Q;

	printf("\nnoElem_d_Q1:%d",noElem_d_Q);

	//Tạo Embedding dQ1, khi đã biết kích thước của chúng	

	cudaStatus = cudaMalloc((void**)&Q1,sizedQ*sizeof(Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dQ1 in createEmbeddingRoot1() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(Q1,-1,sizedQ*sizeof(Embedding));
	}

	cudaStatus = cudaMalloc((void**)&Q2,sizedQ*sizeof(Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dQ1 in createEmbeddingRoot1() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(Q2,-1,sizedQ*sizeof(Embedding));
	}
	kernelSetValueFordQ<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,Q1,Q2,d_scanResult,li,lij,lj);
	cudaDeviceSynchronize();

	getPointer(dArrPointerEmbedding,noElem_dArrPointerEmbedding,Q1);
	getPointer(dArrPointerEmbedding,noElem_dArrPointerEmbedding,Q2);


	for (int j = 0; j < 2; j++)
	{
		//Lấy kích thước của dQ lưu vào mảng d_arrSizedQ
		cudaStatus = getSizedQ(dArrSizedQ,noElem_dArrSizedQ,sizedQ);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n getSizedQ  in makeColumnQ failed",stderr);
			goto Error;
		}	
	}


	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createEmbeddingRoot() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}


//Kernel khởi tạo giá trị cho right most path trên device */
__global__ void kernelInitializeValueForRMPath(int *dRMPath,int noElem_dRMPath){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dRMPath){
		dRMPath[i]=i;		
	}

}

/* Tạo một right most path trên device */
inline cudaError_t createRMPath(int *&dRMPath,int &noElem_dRMPath){
	cudaError_t cudaStatus;
	//Khởi tạo kích thước ban đầu của dRMPath bằng 2
	noElem_dRMPath=2;
	cudaStatus = cudaMalloc((void**)&dRMPath,noElem_dRMPath*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dRMPath failed",cudaStatus);
		goto Error;
	}

	kernelInitializeValueForRMPath<<<1,2>>>(dRMPath,noElem_dRMPath);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n CudaDeviceSynchronize() in createRMPath() failed",cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

__global__ void kernelPrintRMPath(int *dRMPath,int noElem_dRMPath){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dRMPath){
		printf("\n dRMPath[%d]: %d",i,dRMPath[i]);
	}

}


//Hàm hiển thị nội dung dRMPath trên device
inline cudaError_t printRMPath(int *dRMPath,int noElem_dRMPath){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dRMPath + block.x - 1)/block.x);

	kernelPrintRMPath<<<grid,block>>>(dRMPath,noElem_dRMPath);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n CudaDeviceSynchronize() in createRMPath() failed",cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

//kernel tìm số lượng embedding hiện tại
__global__ void kernelGetNumberOfEmbedding(int *dArrSizedQ,int noElem_dArrSizedQ,int *dNumberOfEmbedding){
	dNumberOfEmbedding[0] = dArrSizedQ[noElem_dArrSizedQ-1];
}

//Hàm tìm số lượng embedding hiện tại
inline cudaError_t findNumberOfEmbedding(int *dArrSizedQ,int noElem_dArrSizedQ,int &noElem_dArrPointerdHO){
	cudaError_t	cudaStatus;

	noElem_dArrPointerdHO=0;
	int *dNumberOfEmbedding;
	cudaStatus = cudaMalloc((void**)&dNumberOfEmbedding,sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dNumberOfEmbedding in findNumberOfEmbedding() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dNumberOfEmbedding,0,sizeof(int));
	}

	kernelGetNumberOfEmbedding<<<1,1>>>(dArrSizedQ,noElem_dArrSizedQ,dNumberOfEmbedding);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}

	cudaMemcpy(&noElem_dArrPointerdHO,dNumberOfEmbedding,sizeof(int),cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
Error:
	cudaFree(dNumberOfEmbedding);
	return cudaStatus;
}

inline cudaError_t createElementdHO(int *&dHO,int maxOfVer){
	cudaError_t	cudaStatus;

	cudaStatus = cudaMalloc((void**)&dHO,maxOfVer*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() for dHO in createElementdHO() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dHO,0,maxOfVer*sizeof(int));
	}

	//cudaDeviceSynchronize();
	//cudaStatus=cudaGetLastError();
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n cudaDeviceSynchronize() in createElementdHO() failed",cudaStatus);
	//	goto Error;
	//}
Error:
	return cudaStatus;
}

//kernel lấy pointer trỏ đến bộ nhớ mảng ở device rồi gán cho dArrPointerdHO
__global__ void	kernelAssignPointer(int **dArrPointerdHO,int pos,int *dHO){
	dArrPointerdHO[pos]=dHO;
}

//Hàm lấy pointer trỏ đến bộ nhớ mảng ở device rồi gán cho dArrPointerdHO
inline cudaError_t assignPointer(int **&dArrPointerdHO,int pos,int *dHO){
	cudaError_t cudaStatus;

	kernelAssignPointer<<<1,1>>>(dArrPointerdHO,pos,dHO);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in assignPointer() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//Kernel in mảng double pointer Int trên device
__global__ void kernelPrintDoublePointerInt(int **dArrPointerdHO,int noElem_dArrPointerdHO,unsigned int maxOfVer){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dArrPointerdHO){
		for (int j = 0; j < maxOfVer; j++)
		{
			printf("\n Thread %d: j:%d V[%d]:%d",i,j,j,dArrPointerdHO[i][j]);;
		}
	}

}


/* Hàm in mảng double pointer int (dArrPointerdHO) khi biết số lượng phần tử mảng (noElem_dArrPointerdHO) và
* Kích thước của mỗi phần tử mảng */
inline cudaError_t printDoublePointerInt(int **dArrPointerdHO,int noElem_dArrPointerdHO,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dArrPointerdHO + block.x - 1)/block.x);
	kernelPrintDoublePointerInt<<<grid,block>>>(dArrPointerdHO,noElem_dArrPointerdHO,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printDoublePointerInt() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//Hàm tạo mảng double pointer Int trên device (dArrPointerdHO) khi biết trước số lượng phần tử cần tạo và kích thước của mỗi mảng.
inline cudaError_t createdArrPointerdHO(int **&dArrPointerdHO,int noElem_dArrPointerdHO,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dArrPointerdHO,noElem_dArrPointerdHO*sizeof(int*));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() for dArrPointerdHO in createdArrPointerdHO() failed",cudaStatus);
		goto Error;
	}
	for (int i = 0; i < noElem_dArrPointerdHO; i++)
	{
		int noElem_dHO=maxOfVer;
		int *dHO=nullptr;
		cudaStatus = createElementdHO(dHO,maxOfVer);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n createElementdHO() in createdArrPointerdHO() failed",cudaStatus);
			goto Error;
		}

		int pos = i;
		assignPointer(dArrPointerdHO,pos,dHO);
	}

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createdArrPointerdHO() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;

}

//Hàm tạo phần tử dHLN trên device
inline cudaError_t createElementdHLN(int *&dHLN,int noElem_dHLN){
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dHLN,noElem_dHLN*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() dHLN in createElementdHLN() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dHLN,0,noElem_dHLN*sizeof(int));
	}

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createElementdHLN() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//Hàm tạo mảng double pointer int dHLN
inline cudaError_t createdArrPointerdHLN(int **&dArrPointerdHLN,int noElem_dArrPointerdHO,int *hNumberEdgeInEachGraph,int *hArrGraphId){
	cudaError_t cudaStatus;
	//Cấp phát bộ nhớ trên device cho dArrpointerdHLN theo số lượng embedding, cũng chính bằng số lượng phần tử của mảng dArrPointerdHO (noElem_dArrPointerdHO)
	cudaStatus = cudaMalloc((void**)&dArrPointerdHLN, noElem_dArrPointerdHO*sizeof(int*));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() dArrPointerdHLN in createdArrPointerdHLN() failed",cudaStatus);
		goto Error;
	}	


	for (int i = 0; i < noElem_dArrPointerdHO; i++)
	{
		int index = hArrGraphId[i];
		int *dHLN=nullptr;
		cudaStatus = createElementdHLN(dHLN,hNumberEdgeInEachGraph[index]);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n createElementdHLN() in createdArrPointerdHLN() failed",cudaStatus);
			goto Error;
		}		
		cudaStatus = assignPointer(dArrPointerdHLN,i,dHLN);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n assignPointer() in createdArrPointerdHLN() failed",cudaStatus);
			goto Error;
		}
	}

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createdArrPointerdHLN() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel tìm graphid của tất cả các embedding và lưu kết quả vào mảng
__global__ void kernelFindGraphIdOfAllEmbedding(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrGraphId,unsigned int maxOfVer,int noElemOfEmbedding,int *dArrSizedQ){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemOfEmbedding){
		int vid =dArrPointerEmbedding[noElem_dArrPointerEmbedding-1][i].vid;
		int graphId=vid/maxOfVer;
		dArrGraphId[i]=graphId;
		//printf("\nThread %d: vid:%d graphId:%d maxOfVer:%d",i,vid,graphId,maxOfVer);
	}

}

//Hàm tìm graphid của tất cả các embedding và lưu kết quả vào mảng
inline cudaError_t findGraphIdOfAllEmbedding(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *&hArrGraphId,unsigned int maxOfVer,int *&dArrGraphId,int noElemOfEmbedding,int *dArrSizedQ){
	cudaError_t cudaStatus;

	hArrGraphId = (int*)malloc(noElemOfEmbedding*sizeof(int));
	if(hArrGraphId==NULL){
		printf("\nMalloc hArrGraphId in findGraphIdOfAllEmbedding() failed\n");
		exit(1);
	}


	//int *dArrGraphId=nullptr;
	cudaStatus = cudaMalloc((void**)&dArrGraphId,noElemOfEmbedding*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() dArrGraphId in findGraphIdOfAllEmbedding() failed",cudaStatus);
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((noElemOfEmbedding + block.x -1)/block.x);

	kernelFindGraphIdOfAllEmbedding<<<grid,block>>>(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrGraphId,maxOfVer,noElemOfEmbedding,dArrSizedQ);
	cudaDeviceSynchronize();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindGraphIdOfAllEmbedding in findGraphIdOfAllEmbedding() failed",cudaStatus);
		goto Error;
	}

	cudaMemcpy(hArrGraphId,dArrGraphId,noElemOfEmbedding*sizeof(int),cudaMemcpyDeviceToHost);

	/*printf("\n**********hArrGraphId ***********\n");
	for (int j = 0; j < noElemOfEmbedding; j++)
	{
	printf("\n hArrGraphId[%d]:%d",j,hArrGraphId[j]);
	}*/

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindGraphIdOfAllEmbedding in findGraphIdOfAllEmbedding() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel in mảng double pointer int dArrPointerdHLN
__global__ void kernelPrintdArrPointerdHLN(int **dArrPointerdHLN,int noElem_dArrPointerdHO,int *dNumberEdgeInEachGraph,int *dArrGraphId){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dArrPointerdHO){
		int n =dNumberEdgeInEachGraph[dArrGraphId[i]];
		for (int j = 0; j < n; j++)
		{
			printf("\n Thread %d: j:%d dArrPointerdHLN[%d][%d]:%d",i,j,i,j,dArrPointerdHLN[i][j]);
		}
	}

}

//Hàm in mảng double pointer int dArrPointerdHLN
inline cudaError_t printdArrPointerdHLN(int **dArrPointerdHLN,int noElem_dArrPointerdHO,int *dNumberEdgeInEachGraph,int *dArrGraphId){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem_dArrPointerdHO+block.x - 1)/block.x);
	kernelPrintdArrPointerdHLN<<<grid,block>>>(dArrPointerdHLN,noElem_dArrPointerdHO,dNumberEdgeInEachGraph,dArrGraphId);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindGraphIdOfAllEmbedding in findGraphIdOfAllEmbedding() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel tạo mảng dArrNumberEdgeOfEachdHLN dựa vào graphId đã thu thập được theo thứ tự của từng embedding lưu trong mảng dArrGraphId
__global__ void kernelCreatedArrNumberEdgeOfEachdHLN(int *dArrNumberEdgeOfEachdHLN,int noElemOfEmbedding,int *dArrGraphId,int *dNumberEdgeInEachGraph){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemOfEmbedding){
		dArrNumberEdgeOfEachdHLN[i]= dNumberEdgeInEachGraph[dArrGraphId[i]];
	}

}


//Hàm tạo mảng dArrNumberEdgeOfEachdHLN dựa vào graphId đã thu thập được theo thứ tự của từng embedding lưu trong mảng dArrGraphId
inline cudaError_t createdArrNumberEdgeOfEachdHLN(int *&dArrNumberEdgeOfEachdHLN,int noElemOfEmbedding,int *dArrGraphId,int *dNumberEdgeInEachGraph){
	cudaError_t cudaStatus;

	//Cấp phát bộ nhớ cho mảng dArrNumberEdgeOfEachdHLN với số lượng phần tử bằng với số lượng embedding
	cudaStatus = cudaMalloc((void**)&dArrNumberEdgeOfEachdHLN, noElemOfEmbedding*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() dArrNumberEdgeOfEachdHLN in createdArrPointerdHLN() failed",cudaStatus);
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((noElemOfEmbedding+block.x - 1)/block.x);
	kernelCreatedArrNumberEdgeOfEachdHLN<<<grid,block>>>(dArrNumberEdgeOfEachdHLN,noElemOfEmbedding,dArrGraphId,dNumberEdgeInEachGraph);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelCreatedArrNumberEdgeOfEachdHLN in createdArrNumberEdgeOfEachdHLN() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel in nội dung của mảng dArrPointerdHLN khi biết số lượng cạnh của mỗi phần tử tương ứng của embedding được lưu trong mảng dArrNumberEdgeOfEachdHLN
__global__ void kernelprintDoublePointerInt(int **dArrPointerdHLN,int noElemOfEmbedding,int *dArrNumberEdgeOfEachdHLN){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElemOfEmbedding){
		int length = dArrNumberEdgeOfEachdHLN[i];
		for (int j = 0; j < length; j++)
		{
			printf("\n Thread %d: j:%d dArrPointerdHLN[%d][%d]:%d",i,j,i,j,dArrPointerdHLN[i][j]);
		}
	}

}


//Overloading function printDoublePointerInt() để in nội dung mảng dArrPointerdHLN dựa vào số lượng embedding và số lượng cạnh trong mỗi phần tử
inline cudaError_t printDoublePointerInt(int **dArrPointerdHLN,int noElemOfEmbedding,int *dArrNumberEdgeOfEachdHLN){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElemOfEmbedding + block.x -1)/block.x);

	kernelprintDoublePointerInt<<<grid,block>>>(dArrPointerdHLN,noElemOfEmbedding,dArrNumberEdgeOfEachdHLN);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize()  in printDoublePointerInt() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

__global__ void kernelAssignValueForGraphHistory(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrPrevQ,int noElemOfEmbedding,int *d_O,int *d_N,unsigned int maxOfVer,int **dArrPointerdHO,int **dArrPointerdHLN,int *dArrNumberEdgeOfEachdHLN){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//Mỗi một embedding sẽ cập nhật graphHistory tương ứng của nó (gồm 2 mảng: dArrPointerdHO(mảng các đỉnh của embedding mà thread i đang xử lý) và dArrPointerdHLN(mảng các cạnh tương ứng với ánh xạ đỉnh).)
	if(i<noElemOfEmbedding){
		int vid = dArrPointerEmbedding[noElem_dArrPointerEmbedding-1][i].vid; //Từ cột Q cuối cùng, chúng ta lấy ra được vid của 6 embedding tương ứng
		int indexOfFirstVertexInGraph = vid-(vid%maxOfVer); //the first global id vertex in graph
		int toVid = vid;//đỉnh to của cạnh thuộc embedding
		int idxOfVertex= (vid%maxOfVer); //Vị trí của phần tử đỉnh cần cập nhật trong mảng dArrPointerdHO[i][idxOfVertex];
		dArrPointerdHO[i][idxOfVertex]=2; //Cập nhật đỉnh đã thuộc right most path của embedding trong mảng dArrPointerdHO tương ứng.
		int prevQ= dArrPrevQ[noElem_dArrPointerEmbedding-1]; 
		int newi=dArrPointerEmbedding[noElem_dArrPointerEmbedding-1][i].idx; //lấy index gán cho newi

		while (true)
		{			

			vid = dArrPointerEmbedding[prevQ][newi].vid; //truy xuất phần tử phía trước theo prevQ và newi
			int fromVid=vid; //đỉnh from của cạnh thuộc embedding


			int idxEdge = d_O[vid]-d_O[indexOfFirstVertexInGraph]; //vị trí cạnh cần cập nhật được khởi tạo bằng giá trị index của vid đang xét trừ đi giá trị index của đỉnh đầu tiên trong đồ thị đó.
			int indexOfdN=d_O[fromVid];

			while (d_N[indexOfdN]!=toVid){
				idxEdge=idxEdge+1;
				indexOfdN++;
			}

			int fromVidR=toVid;
			int toVidR=fromVid;
			int indexOfEdgeR=d_O[fromVidR]-d_O[indexOfFirstVertexInGraph];
			indexOfdN=d_O[fromVidR];
			while(d_N[indexOfdN]!=toVidR){
				indexOfEdgeR++;
				indexOfdN++;
			}


			//Nếu không phải là đỉnh đầu tiên thì phải cộng vào idxEdge một lượng bằng tổng bậc của các đỉnh trước đó
			//Tổng bậc của các đỉnh trước đó chính bằng 

			idxOfVertex = (vid%maxOfVer); //Đánh dấu đỉnh thuộc Embedding
			dArrPointerdHO[i][idxOfVertex]=2;


			dArrPointerdHLN[i][idxEdge]=2;//Đánh dấu cạnh thuộc Embedding. vì đây là đơn đồ thị vô hướng nên cạnh AB cũng bằng cạnh BA,do đó ta phải đánh dấu cạnh BA cũng thuộc embedding.
			dArrPointerdHLN[i][indexOfEdgeR]=2;


			if(dArrPrevQ[prevQ]==-1) return; //nếu là cột Q đầu tiên thì dừng lại vì đã duyệt xong embedding
			newi=dArrPointerEmbedding[prevQ][newi].idx; //ngược lại thì lấy index của cột Q phía trước
			prevQ=dArrPrevQ[prevQ]; //Lấy Q phía trước
			toVid=fromVid; //cập nhật lại đỉnh to.
		}


	}
}

//Xây dựng graphHistory cho tất cả các embedding
inline cudaError_t createGraphHistory(Embedding **dArrPointerEmbedding,int *dArrSizedQ,int *dArrPrevQ,int noElem_dArrPointerEmbedding,int noElem_dArrSizedQ,int noElem_dArrPrevQ,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int maxOfVer,int **&dArrPointerdHO,int &noElem_dArrPointerdHO,int **&dArrPointerdHLN,int *&dArrNumberEdgeOfEachdHLN,int *hNumberEdgeInEachGraph,int noElem_hNumberEdgeInEachGraph,int *dNumberEdgeInEachGraph){
	cudaError_t cudaStatus;

	//số lượng embedding chính bằng giá trị của biến noElem_dArrPointerdHO
	cudaStatus = findNumberOfEmbedding(dArrSizedQ,noElem_dArrSizedQ,noElem_dArrPointerdHO);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n findNumberOfEmbedding() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
	int noElemOfEmbedding=noElem_dArrPointerdHO;
	//In nội dung số lượng phần tử embedding vừa tìm được
	//printf("\nNumber Of Embedding: %d",noElem_dArrPointerdHO);

	/* Tạo graphHistory
	*	1. Tạo mảng dArrPointerdHO
	*	2. Tạo mảng dArrPointerdHLN
	*	3. Tạo mảng dArrNumberEdgeOfEachdHLN: mảng này mô tả số cạnh của mỗi phần tử trong mảng dArrPointerdHLN
	*	Bước 2 và 3 có thể được thực hiện một cách độc lập, nên có thể xử lý song song ở bước này.
	*	4. Cập nhật nội dung cho 3 mảng trên.
	*/

	//1. Tạo 5 mảng có số lượng phần tử là  maxOfVer trên device, và chép pointer của các mảng bỏ vào phần tử dArrPointerEmbedding
	cudaStatus = createdArrPointerdHO(dArrPointerdHO,noElem_dArrPointerdHO,maxOfVer);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n createdArrPointerdHO() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
	//In nội dung của mảng vừa tạo được
	/*printf("\n ********** dArrPointerdHO *****************\n");
	printDoublePointerInt(dArrPointerdHO,noElem_dArrPointerdHO,maxOfVer);
	if(cudaStatus!=cudaSuccess){
	fprintf(stderr,"\n printDoublePointerInt() in createGraphHistory() failed",cudaStatus);
	goto Error;
	}*/

	//2. Tạo dArrPointerHLN
	/* Tìm số lượng cạnh của mỗi embedding
	* Biết được global vertex id của embedding thì chúng ta biết được graphId của embedding đó
	* Biết được graphID thì suy ra được số lượng cạnh của embedding.
	* Trước tiên nên tính số lượng cạnh của mỗi đồ thị trong CSDL và lưu chúng vào một mảng <-- Làm được
	* Sau đó duyệt qua các vid của embedding ở last column Q để biết được graphID mà embedding thuộc vào
	*/
	//Tính graphId của từng embedding và lưu vào mảng 
	int *dArrGraphId=nullptr; //Mảng này dùng để in nội dung của mảng dArrPointerdHLN
	int *hArrGraphId=nullptr; //Lấy graphId ở mảng này mang đi tra trong mảng hNumberEdgeInEachGraph để lấy số lượng cạnh cho embedding đó để tạo dArrPointerdHLN
	cudaStatus = findGraphIdOfAllEmbedding(dArrPointerEmbedding,noElem_dArrPointerEmbedding,hArrGraphId,maxOfVer,dArrGraphId,noElemOfEmbedding,dArrSizedQ);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n findGraphIdOfAllEmbedding() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
	cudaStatus =createdArrPointerdHLN(dArrPointerdHLN,noElem_dArrPointerdHO,hNumberEdgeInEachGraph,hArrGraphId);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n createdArrPointerdHLN() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}

	//In nội dung mảng dArrPointerdHLN
	/*printf("\n***************** dArrPointerdHLN ***************\n");

	cudaStatus = printdArrPointerdHLN(dArrPointerdHLN,noElem_dArrPointerdHO,dNumberEdgeInEachGraph,dArrGraphId);
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
	fprintf(stderr,"\n printdArrPointerdHLN() in createGraphHistory() failed",cudaStatus);
	goto Error;
	}*/

	//3. Tạo mảng dArrNumberEdgeOfEachdHLN

	cudaStatus = createdArrNumberEdgeOfEachdHLN(dArrNumberEdgeOfEachdHLN,noElemOfEmbedding,dArrGraphId,dNumberEdgeInEachGraph);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n createdArrNumberEdgeOfEachdHLN() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}

	//printf("\n**************dArrNumberEdgeOfEachdHLN**************\n");
	//printInt(dArrNumberEdgeOfEachdHLN,noElemOfEmbedding);

	//printf("\n**************dArrNumberEdgeOfEachdHLN**************\n");
	//printDoublePointerInt(dArrPointerdHLN,noElemOfEmbedding,dArrNumberEdgeOfEachdHLN);

	//4.1 Cập nhật nội dung cho graphHistory
	/* Cần có cơ sở dữ liệu để ánh xạ đỉnh và cạnh phù hợp vào mảng dArrPointerEmbedding (chứa idx và vid), dArrPointerdHLN (chứa cạnh)
	*	Mỗi một thread sẽ chịu trách nhiệm cập nhật dữ liệu cho 1 embedding
	*/


	kernelAssignValueForGraphHistory<<<1,noElemOfEmbedding>>>(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrPrevQ,noElemOfEmbedding,d_O,d_N,maxOfVer,dArrPointerdHO,dArrPointerdHLN,dArrNumberEdgeOfEachdHLN);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel tính số cạnh trong mỗi đồ thị trong CSDL và lưu vào biến mảng tương ứng.
__global__ void kernelGetNumberOfEdgeInGraph(int *d_O,int numberOfElementd_N,unsigned int numberOfGraph,unsigned int maxOfVer,int *dNumberEdgeInEachGraph){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<numberOfGraph){
		int graphId=i;
		int idxFrom = graphId*maxOfVer;		
		int idxFirstNext = (graphId+1)*maxOfVer;
		int r=0;
		if (graphId!=(numberOfGraph-1)){
			r=d_O[idxFirstNext]-d_O[idxFrom];
		}else
		{
			r=numberOfElementd_N-d_O[idxFrom];
		}
		dNumberEdgeInEachGraph[i]=r;	
	}
}

//Hàm tính số cạnh của tất cả các đồ thị trong CSDL, kết quả lưu vào một mảng tương ứng
inline cudaError_t getNumberOfEdgeInGraph(int *d_O,int numberOfElementd_N,unsigned int maxOfVer,int *&hNumberEdgeInEachGraph,int *&dNumberEdgeInEachGraph,unsigned int numberOfGraph){
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dNumberEdgeInEachGraph,numberOfGraph*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc() in dNumberEdgeInEachGraph createGraphHistory() failed",cudaStatus);
		goto Error;
	}

	dim3 block(blocksize);
	dim3 grid((numberOfGraph + block.x-1)/block.x);

	kernelGetNumberOfEdgeInGraph<<<grid,block>>>(d_O,numberOfElementd_N,numberOfGraph,maxOfVer,dNumberEdgeInEachGraph);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize()  kernelGetNumberOfEdgeInGraph in getNumberOfEdgeInGraph() failed",cudaStatus);
		goto Error;
	}

	//printf("\n *************dNumberEdgeInEachGraph********\n" );
	//printInt(dNumberEdgeInEachGraph,numberOfGraph);

	hNumberEdgeInEachGraph = (int*)malloc(numberOfGraph*sizeof(int));
	if(hNumberEdgeInEachGraph==NULL){
		printf("\n Malloc hNumberEdgeInEachGraph in getNumberOfEdgeInGraph() failed" );
		exit(1);
	}

	cudaMemcpy(hNumberEdgeInEachGraph,dNumberEdgeInEachGraph,numberOfGraph*sizeof(int),cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in createGraphHistory() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel in tất cả column Q của embedding
__global__ void kernelprintAllEmbeddingColumn(Embedding **dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrPointerEmbedding){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dArrPointerEmbedding){
		Embedding* Q = dArrPointerEmbedding[i];
		int lenght = dArrSizedQ[i];
		for (int j = 0; j < lenght; j++)
		{
			printf("\n Thread %d: j:%d (idx:%d vid:%d)",i,j,Q[j].idx,Q[j].vid);
		}
	}

}


//Hàm in tất cả các column Q của embedding
inline cudaError_t printAllEmbeddingColumn(Embedding **dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrPointerEmbedding){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dArrPointerEmbedding + block.x - 1)/block.x);
	printf("\n****************** All Columm in Embedding dArrPointerEmbedding *************\n");
	kernelprintAllEmbeddingColumn<<<grid,block>>>(dArrPointerEmbedding,dArrSizedQ,noElem_dArrPointerEmbedding);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printAllEmbeddingColumn() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel in một embedding khi biết vị trí Row của nó trong last column Q.
__global__ void kernelprintEmbeddingFromPos(Embedding **dArrPointerEmbedding,int posColumn,int posRow){
	Embedding *Q =dArrPointerEmbedding[posColumn];
	printf("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);
	while (true)
	{
		posRow = Q[posRow].idx;
		posColumn=posColumn-1;		
		Q=dArrPointerEmbedding[posColumn];
		printf("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);
		posRow=Q[posRow].idx;
		if(posRow==-1) return;
	}
}


//Hàm in một embedding khi biết vị trí Row của nó trong last column Q.
inline cudaError_t printEmbeddingFromPos(Embedding **dArrPointerEmbedding,int posColumn,int posRow){
	cudaError_t cudaStatus;
	printf("\n ****Embeding from posColumn: %d posRow:%d **************\n",posColumn,posRow);
	kernelprintEmbeddingFromPos<<<1,1>>>(dArrPointerEmbedding,posColumn,posRow);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printEmbeddingFromPos() failed",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}

//kernel tìm bậc của các vid trên cột Q và lưu kết quả vào mảng dArrDegreeOfVid
__global__ void kernelCalDegreeOfVid(Embedding **dArrPointerEmbedding,int idxQ,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,float *dArrDegreeOfVid){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_Embedding){
		int vid = dArrPointerEmbedding[idxQ][i].vid;
		float degreeOfV =0;
		int nextVid;
		int graphid;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		if (vid==numberOfElementd_O-1){ //nếu như đây là đỉnh cuối cùng trong d_O
			degreeOfV=numberOfElementd_N-d_O[vid]; //thì bậc của đỉnh vid chính bằng tổng số cạnh trừ cho giá trị của d_O[vid].
		}
		else
		{
			nextVid = vid+1; //xét đỉnh phía sau có khác 1 hay không?
			graphid=vid/maxOfVer;
			if(d_O[nextVid]==-1 && graphid==lastGraphId){
				degreeOfV=numberOfElementd_N-d_O[vid];
			}
			else if(d_O[nextVid]==-1 && graphid!=lastGraphId){
				nextVid=(graphid+1)*maxOfVer;
				degreeOfV=d_O[nextVid]-d_O[vid];
			}
			else
			{
				degreeOfV=d_O[nextVid]-d_O[vid];
			}							
		}
		dArrDegreeOfVid[i]=degreeOfV;
	}

}

//Hàm tìm bậc của các đỉnh trên column Q và lưu kết quả vào mảng dArrDegreeOfVid
inline cudaError_t findDegreeOfVer(Embedding **dArrPointerEmbedding,int idxQ,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N, unsigned int maxOfVer,float *&dArrDegreeOfVid){
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&dArrDegreeOfVid,noElem_Embedding*sizeof(float));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc dArrDegreeOfVid in findMaxDegreeOfVer() failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrDegreeOfVid,0,noElem_Embedding*sizeof(float));
	}

	dim3 block(blocksize);
	dim3 grid((noElem_Embedding + block.x -1)/block.x);
	kernelCalDegreeOfVid<<<grid,block>>>(dArrPointerEmbedding,idxQ,d_O, numberOfElementd_O,noElem_Embedding,numberOfElementd_N, maxOfVer,dArrDegreeOfVid);	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();	
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() of kernelCalDegreeOfVid in findDegreeOfVer() failed",cudaStatus);
		goto Error;
	}

Error:

	return cudaStatus;
}


//Hàm tìm bậc lớn nhất của các đỉnh vid trong cột Q và lưu kết quả vào biến maxDegreeOfVer và float *dArrDegreeOfVid
inline cudaError_t findMaxDegreeOfVer(Embedding **dArrPointerEmbedding,int idxQ,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,int &maxDegreeOfVer,float *&dArrDegreeOfVid){
	cudaError_t cudaStatus;

	//Lấy bậc của các đỉnh vid trong cột Q và lưu vào mảng dArrDegreeOfVid có số lượng phần tử bằng số lượng phần tử của embedding

	cudaStatus = findDegreeOfVer(dArrPointerEmbedding,idxQ,d_O, numberOfElementd_O,noElem_Embedding,numberOfElementd_N, maxOfVer,dArrDegreeOfVid);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\findDegreeOfVer() in findMaxDegreeOfVer() failed");
		goto Error;
	}

	printf("\n*******dArrDegreeOfVid*************\n");
	printFloat(dArrDegreeOfVid,noElem_Embedding);

	//Tìm bậc lớn nhất và lưu kết quả vào biến maxDegreeOfVer
	float *h_max;
	h_max = (float*)malloc(sizeof(float));
	if(h_max==NULL){
		printf("\nMalloc h_max failed");
		exit(1);
	}

	float *d_max;
	int *d_mutex;
	cudaStatus=cudaMalloc((void**)&d_max,sizeof(float));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_max failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_max,0,sizeof(float));
	}

	cudaStatus=cudaMalloc((void**)&d_mutex,sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_mutex failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_mutex,0,sizeof(int));
	}

	dim3 gridSize = 256;
	dim3 blockSize = 256;
	find_maximum_kernel<<<gridSize, blockSize>>>(dArrDegreeOfVid, d_max, d_mutex, noElem_Embedding);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize find_maximum_kernel in findMaxDegreeOfVer() failed");
		goto Error;
	}

	// copy from device to host
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

	//report results
	maxDegreeOfVer = (int)(*h_max); //bậc lớn nhất của các đỉnh trong 1 cột Q
	printf("\nMax degree of vid in Q column is: %d",maxDegreeOfVer);




Error:
	free(h_max);
	cudaFree(d_max);
	//cudaFree(dArrDegreeOfVid); Giữ lại bậc của các đỉnh trong cột Q để thuận lợi cho việc tìm các mở rộng ở bước kế tiếp
	return cudaStatus;
}

//kernel tìm các mở rộng hợp lệ và ghi nhận vào mảng dArrV và dArrExtension tương ứng.
__global__ void kernelFindValidForwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int noElem_Embedding,int *d_O,int *d_LO,int *d_N,int *d_LN,float *dArrDegreeOfVid,int maxDegreeOfVer,struct_V *dArrV,EXT *dArrExtension,int idxQ,int minLabel,int maxid){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i<noElem_Embedding){
		int posColumn =noElem_dArrPointerEmbedding-1;
		int posRow=i;
		int col = posColumn;
		int row = posRow;
		Embedding *Q=dArrPointerEmbedding[idxQ];
		int vid = Q[i].vid;
		int degreeVid=__float2int_rn(dArrDegreeOfVid[i]);
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid];
		int labelFromVid = d_LO[vid];
		int toVid;
		int labelToVid;
		bool b=true;
		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối
		{			
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			posColumn=col;
			posRow=row;
			Q=dArrPointerEmbedding[posColumn];
			printf("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//1. Trước tiên kiểm tra nhãn của labelToVid có nhỏ hơn minLabel hay không. Nếu nhỏ hơn thì return
			if(labelToVid<minLabel) continue;
			//2. kiểm tra xem đỉnh toVid có tồn tại trong embedding hay không nếu tồn tại thì return
			//Duyệt qua embedding column từ Q cuối đến Q đầu, lần lượt lấy vid so sánh với toVid

			//printf("\n Q[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[1][0]
			if(toVid==Q[posRow].vid) continue;
			//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);

			while (true)
			{
				posRow = Q[posRow].idx;//0
				posColumn=posColumn-1;		//0
				Q=dArrPointerEmbedding[posColumn];
				//printf("\n posColumn[%d] Row[%d] (idx:%d vid:%d)",posColumn,posRow,Q[posRow].idx,Q[posRow].vid);//Q[0][0]
				//printf("\nj:%d toVid:%d Q.vid:%d",j,toVid,Q[posRow].vid);
				if(toVid==Q[posRow].vid) {
					b=false; break;
				}
				posRow=Q[posRow].idx;//-1
				//printf("\nposRow:%d",posRow);
				if(posRow==-1) break;
			}
			if (b==false){b=true; continue;}
			int indexOfd_arr_V=i*maxDegreeOfVer+j;
			//printf("\nThread %d: m:%d",i,maxDegreeOfVer);
			int indexOfd_LN=indexToVidIndN;
			dArrV[indexOfd_arr_V].valid=1;
			printf("\ndArrV[%d].valid:%d",indexOfd_arr_V,dArrV[indexOfd_arr_V].valid);
			//cập nhật dữ liệu cho mảng dArrExtension
			dArrExtension[indexOfd_arr_V].vgi=vid;
			dArrExtension[indexOfd_arr_V].vgj=toVid;
			dArrExtension[indexOfd_arr_V].lij=d_LN[indexOfd_LN];
			printf("\n");
			printf("d_LN[%d]:%d ",indexOfd_LN,d_LN[indexOfd_LN]);
			dArrExtension[indexOfd_arr_V].li=labelFromVid;
			dArrExtension[indexOfd_arr_V].lj=labelToVid;
			dArrExtension[indexOfd_arr_V].vi=idxQ;
			dArrExtension[indexOfd_arr_V].vj=maxid+1;
			dArrExtension[indexOfd_arr_V].posColumn=col;
			dArrExtension[indexOfd_arr_V].posRow=row;
		}
	}
}

//kernel in mảng struct_V *dArrV trên device
__global__ void kernelprintdArrV(struct_V *dArrV,int noElem_dArrV,EXT *dArrExtension){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dArrV){
		int vi = dArrExtension[i].vi;
		int vj = dArrExtension[i].vj;
		int li = dArrExtension[i].li;
		int lij = dArrExtension[i].lij;
		int lj = dArrExtension[i].lj;
		printf("\n dArrV[%d].backward:%d ,dArrV[%d].valid:%d Extension:(vgi:%d,vgj:%d) (vi:%d vj:%d li:%d lij:%d lj:%d)",i,dArrV[i].backward,i,dArrV[i].valid,dArrExtension[i].vgi,dArrExtension[i].vgj,vi,vj,li,lij,lj);
	}

}

//Hàm in mảng struct_V *dArrV
inline cudaError_t printdArrV(struct_V *dArrV,int noElem_dArrV,EXT *dArrExtension){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV + block.x -1 )/block.x);
	kernelprintdArrV<<<grid,block>>>(dArrV,noElem_dArrV,dArrExtension);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in printdArrV() failed", cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;
}
//kernel trích phần tử valid từ mảng dArrV và lưu vào mảng dArrValid
__global__ void kernelExtractValidFromdArrV(struct_V *dArrV,int noElem_dArrV,int *dArrValid){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<noElem_dArrV){
		dArrValid[i]=dArrV[i].valid;
	}
}

//kernel trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
__global__ void kernelExtractValidExtensionTodExt(EXT *dArrExtension,int *dArrValid,int *dArrValidScanResult,int noElem_dArrV,EXT *dExt,int noElem_dExt){
	int i =blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dArrV){
		if(dArrValid[i]==1){
			dExt[dArrValidScanResult[i]].vi = dArrExtension[i].vi;
			dExt[dArrValidScanResult[i]].vj = dArrExtension[i].vj;
			dExt[dArrValidScanResult[i]].li = dArrExtension[i].li;
			dExt[dArrValidScanResult[i]].lij = dArrExtension[i].lij;
			dExt[dArrValidScanResult[i]].lj = dArrExtension[i].lj;
			dExt[dArrValidScanResult[i]].vgi = dArrExtension[i].vgi;
			dExt[dArrValidScanResult[i]].vgj = dArrExtension[i].vgj;
			dExt[dArrValidScanResult[i]].posColumn = dArrExtension[i].posColumn;
			dExt[dArrValidScanResult[i]].posRow = dArrExtension[i].posRow;
		}

	}

}

//Kernel in nội dung mảng EXT *dExt
__global__ void kernelPrintdExt(EXT *dExt,int noElem_dExt){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_dExt){		
		int vi=dExt[i].vi;
		int vj=dExt[i].vj;
		int li= dExt[i].li;
		int lij=dExt[i].lij;
		int lj=dExt[i].lj;
		int vgi=dExt[i].vgi;
		int vgj=dExt[i].vgj;
		int posColumn= dExt[i].posColumn;
		int posRow=dExt[i].posRow;
		printf("\n Thread %d (vi:%d vj:%d li:%d lij:%d lj:%d) (vgi:%d vgj:%d) (posColumn:%d posRow:%d)",i,vi,vj,li,lij,lj,vgi,vgj,posColumn,posRow);
	}

}

//Hàm in dExt
inline cudaError_t printdExt(EXT *dExt,int noElem_dExt){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dExt+block.x -1)/block.x);
	kernelPrintdExt<<<grid,block>>>(dExt,noElem_dExt);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelPrintdExt in printdExt() failed", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

//Hàm trích các mở rộng hợp lệ từ mảng dArrExtension sang mảng dExt
inline cudaError_t extractValidExtensionTodExt(EXT *dArrExtension,struct_V *dArrV,int noElem_dArrV,EXT *&dExt,int &noElem_dExt){
	cudaError_t cudaStatus;
	//1. Trích dữ liệu ra mảng dArrvalid
	int *dArrValid = nullptr;

	cudaStatus = cudaMalloc((void**)&dArrValid, noElem_dArrV*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrValid in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	dim3 block(blocksize);
	dim3 grid((noElem_dArrV + block.x -1)/block.x);
	kernelExtractValidFromdArrV<<<grid,block>>>(dArrV,noElem_dArrV,dArrValid);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelExtractValidFromdArrV in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	//In nội dung dArrValid
	printf("\n********dArrValid******\n");
	printInt(dArrValid,noElem_dArrV);

	//2. Scan mảng dArrValid để lấy kích thước của mảng cần tạo
	int *dArrValidScanResult = nullptr;
	
	cudaStatus = cudaMalloc((void**)&dArrValidScanResult,sizeof(int)*noElem_dArrV);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n CudaMalloc dArrValidScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrValidScanResult,0,sizeof(int)*noElem_dArrV);
	}


	cudaStatus = scanV(dArrValid,noElem_dArrV,dArrValidScanResult);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n scanV dArrValid in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung kết quả dArrValidScanResult
	printf("\n********dArrValidScanResult******\n");
	printInt(dArrValidScanResult,noElem_dArrV);

	//3. Lấy kích thước của mảng EXT *dExt;
	noElem_dExt=0;
	cudaStatus=getSizeBaseOnScanResult(dArrValid,dArrValidScanResult,noElem_dArrV,noElem_dExt);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n getSizeBaseOnScanResult in extractValidExtensionToExt() failed");
		goto Error;
	}

	//In nội dung noElem_dExt
	printf("\n******** noElem_dExt ******\n");
	printf("\n noElem_dExt:%d",noElem_dExt);


	//4. Khởi tạo mảng dExt có kích thước noElem_dExt rồi trích dữ liệu từ dArrExtension sang dựa vào dArrValid.
	cudaStatus = cudaMalloc((void**)&dExt,sizeof(EXT)*noElem_dExt);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dExt,0,sizeof(EXT)*noElem_dExt);
	}
	dim3 blockb(blocksize);
	dim3 gridb((noElem_dArrV+blockb.x -1)/blockb.x);
	kernelExtractValidExtensionTodExt<<<gridb,blockb>>>(dArrExtension,dArrValid,dArrValidScanResult,noElem_dArrV,dExt,noElem_dExt);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelExtractValidExtensionTodExt in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
	
	//In mảng dExt;
	printf("\n********** dExt **********\n");
	cudaStatus =printdExt(dExt,noElem_dExt);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n printdExt() in extractValidExtensionTodExt() failed", cudaStatus);
		goto Error;
	}
		
Error:
	cudaFree(dArrValid);
	cudaFree(dArrValidScanResult);
	return cudaStatus;
}

//Hàm Tìm tất cả các mở rộng hợp lệ forward từ các đỉnh trên cột Q và lưu vào mảng dExt và noElem_dExt
inline cudaError_t forwardExtensionQ(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrSizedQ,int noElem_Embedding,int idxQ,EXT *&dExt,int &noElem_dExt,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,unsigned int maxOfVer,int minLabel,int maxid){
	cudaError_t cudaStatus;

	//Tìm bậc lớn nhất của các đỉnh vid trong cột Q
	int maxDegreeOfVer=0;
	float *dArrDegreeOfVid=nullptr; //Được sử dụng để tìm các mở rộng từ các vid trên column Q
	cudaStatus = findMaxDegreeOfVer(dArrPointerEmbedding,idxQ,d_O,numberOfElementd_O,noElem_Embedding, numberOfElementd_N,maxOfVer,maxDegreeOfVer,dArrDegreeOfVid);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n findMaxDegreeOfVer() in forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}

	//Tạo mảng dArrV có số lượng phần tử bằng số lượng embedding nhân với bậc lớn nhất của các vid vừa tìm được
	//Tạo mảng d_arr_V có kích thước: maxDegree_vid_Q * |Q|
	//	Lưu ý, mảng d_arr_V phải có dạng cấu trúc đủ thể hiện cạnh mở rộng có hợp lệ hay không và là forward extension hay backward extension.
	//	struct struct_V
	//	{
	//		int valid; //default: 0, valid: 1
	//		int backward; //default: 0- forward; backward: 1
	//	}

	struct_V *dArrV;
	int noElem_dArrV=maxDegreeOfVer*noElem_Embedding;
	cudaStatus=cudaMalloc((void**)&dArrV,noElem_dArrV*sizeof(struct_V));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrV in  failed");
		goto Error;
	}
	else
	{
		cudaMemset(dArrV,0,noElem_dArrV*sizeof(struct_V));
	}

	//Các mở rộng hợp lệ sẽ được ghi nhận vào mảng dArrV, đồng thời thông tin của cạnh mở rộng gồm dfscode, vgi, vgj và row pointer của nó cũng được xây dựng
	//và lưu trữ trong mảng EXT *dExtension, mảng này có số lượng phần tử bằng với mảng dArrV. Sau đó chúng ta sẽ rút trích những mở rộng hợp lệ này và lưu vào dExt. 
	//Để xây dựng dfscode (vi,vj,li,lij,lj) thì chúng ta cần:
	// - Dựa vào giá trị của right most path để xác định vi
	// - Dựa vào maxid để xác định vj
	// - Dựa vào CSDL để xác định các thành phần còn lại.
	//Chúng ta có thể giải phóng bộ nhớ của dExtension sau khi đã trích các mở rộng hợp lệ thành công.


	EXT *dArrExtension= nullptr;
	cudaStatus = cudaMalloc((void**)&dArrExtension,noElem_dArrV*sizeof(EXT));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrExtension forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dArrExtension,0,noElem_dArrV*sizeof(EXT));
	}

	printf("\nnoElem_dArrV:%d",noElem_dArrV );


	//Gọi kernel với các đối số: CSDL, bậc của các đỉnh, dArrV, dArrExtension,noElem_Embedding,maxDegreeOfVer,idxQ,dArrPointerEmbedding,minLabel,maxid
	dim3 block(blocksize);
	dim3 grid((noElem_Embedding+block.x - 1)/block.x);
	kernelFindValidForwardExtension<<<grid,block>>>(dArrPointerEmbedding,noElem_dArrPointerEmbedding,noElem_Embedding,d_O,d_LO,d_N,d_LN,dArrDegreeOfVid,maxDegreeOfVer,dArrV,dArrExtension,idxQ,minLabel,maxid);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelFindValidForwardExtension in forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}
	//In mảng dArrV để kiểm tra thử
	/*printf("\n****************dArrV*******************\n");
	cudaStatus = printdArrV(dArrV,noElem_dArrV,dArrExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n printdArrV() in forwardExtensionQ() failed",cudaStatus);
		goto Error;
	}*/
	//Chép kết quả từ dArrExtension sang dExt

	cudaStatus =extractValidExtensionTodExt(dArrExtension,dArrV,noElem_dArrV,dExt,noElem_dExt);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n extractValidExtensionTodExt() in forwardExtensionQ() failed");
		goto Error;
	}

Error:

	cudaFree(dArrExtension);
	cudaFree(dArrV);
	return cudaStatus;
}

//kernel lấy chép địa chỉ của dExt lưu vào dArrPointerExt
__global__ void kernelGetPointerExt(EXT **dArrPointerExt,EXT *dExt,int pos){
	dArrPointerExt[pos]=dExt;
}


//Tìm tất cả các mở rộng hợp lệ forward và lưu vào mảng dArrPointerExt
inline cudaError_t forwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrSizedQ,int *dRMPath,int noElem_dRMPath,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,unsigned int maxOfVer,EXT **&dArrPointerExt,int &noElem_dArrPointerExt,int minLabel,int maxid,int *&dArrNoElemPointerExt){
	cudaError_t cudaStatus;

	//Lấy số lượng embedding
#pragma region "get noElem_Embedding"
	int noElem_Embedding = 0;
	cudaStatus = findNumberOfEmbedding(dArrSizedQ,noElem_dArrSizedQ,noElem_Embedding);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n findNumberOfEmbedding() in forwardExtension() failed",cudaStatus);
		goto Error;
	}

	//printf("\n noElem_Embedding:%d",noElem_Embedding);
#pragma endregion

	//Duyệt qua các column Q thuộc dRMPath và tìm các mở rộng hợp lệ từ chúng
	int *hRMPath =(int*)malloc(sizeof(int)*noElem_dRMPath);
	if (hRMPath==NULL){
		printf("\n malloc hRMPath in forwardExtension() failed");
		exit(1);
	}

	cudaStatus = cudaMemcpy(hRMPath,dRMPath,sizeof(int)*noElem_dRMPath,cudaMemcpyDeviceToHost);
	if (cudaStatus !=cudaSuccess){
		fprintf(stderr,"\n cudaMemcpy dRMPath --> hRMPath failed",cudaStatus);
		goto Error;
	}

	printf("\n ***************** hRMPath **************\n");
	for (int i = 0; i < noElem_dRMPath; i++)
	{
			printf("\n hRMPath[%d]:%d",i,hRMPath[i]);

	}

	cudaStatus = cudaMalloc((void**)&dArrPointerExt,noElem_dRMPath*sizeof(EXT*));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrPointerExt in forwardExtension() failed",cudaStatus);
		goto Error;		
	}
	else
	{
		cudaMemset(dArrPointerExt,0,noElem_dRMPath*sizeof(EXT*));
	}

	int *hArrNoElemPointerExt;
	hArrNoElemPointerExt = (int*)malloc(sizeof(int)*noElem_dRMPath);
	if(hArrNoElemPointerExt==NULL){
		printf("\nMalloc hArrNoElemPointerExt in kernel.cu failed");
		goto Error;
	}
	else
	{
		memset(hArrNoElemPointerExt,0,sizeof(int)*noElem_dRMPath);
	}

	cudaStatus = cudaMalloc((void**)&dArrNoElemPointerExt,sizeof(int)*noElem_dRMPath);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrNoElemPointerExt in forwardExtension() failed",cudaStatus);
		goto Error;		
	}
	else
	{
		cudaMemset(dArrNoElemPointerExt,0,sizeof(int)*noElem_dRMPath);
	}

	for (int i = noElem_dRMPath-1; i>=0  ; i--)
	{
		int idxQ=hRMPath[i];
		printf("\n*********idxQ:%d***************\n",idxQ);
		EXT *dExt=nullptr; //Những mở rộng hợp lệ sẽ được trích sang mảng dExt
		int noElem_dExt=0;
		cudaStatus = forwardExtensionQ(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrSizedQ,noElem_dArrSizedQ,noElem_Embedding,idxQ,dExt,noElem_dExt,d_O,d_LO,d_N,d_LN, numberOfElementd_O, numberOfElementd_N, maxOfVer,minLabel,maxid);
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n forwardExtensionQ() in forwardExtension() failed",cudaStatus);
			goto Error;		
		}
		//Chép pointer của dExt bỏ vào mảng dArrPointerExt
		hArrNoElemPointerExt[i]=noElem_dExt;
		kernelGetPointerExt<<<1,1>>>(dArrPointerExt,dExt,i);
		cudaDeviceSynchronize();
		cudaStatus=cudaGetLastError();
		if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n cudaDeviceSynchronize() kernelGetPointerExt in forwardExtension() failed",cudaStatus);
			goto Error;
		}
	}

	//chép dữ liệu từ hArrNoElemPointerExt sang dArrNoElemPointerExt
	cudaStatus = cudaMemcpy(dArrNoElemPointerExt,hArrNoElemPointerExt,sizeof(int)*noElem_dRMPath,cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
			fprintf(stderr,"\n cudaMemcpy() hArrNoElemPointerExt sang dArrNoElemPointerExt in forwardExtension() failed",cudaStatus);
			goto Error;
		}


	//cudaDeviceSynchronize();
	//cudaStatus=cudaGetLastError();
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\n cudaDeviceSynchronize() in forwardExtension() failed",cudaStatus);
	//	goto Error;
	//}
Error:
	return cudaStatus;

}


//kernel in mảng dArrPointerExt
__global__ void kernelprintdArrPointerExt(EXT **dArrPointerExt,int *dArrNoElemPointerExt,int noElem_dArrPointerExt){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_dArrPointerExt){
		int noElem_dExt=dArrNoElemPointerExt[i];
		printf("\nThread %d: noElem_dExt:%d",i,noElem_dExt);
		if(noElem_dExt>0){
			EXT* dExt= dArrPointerExt[i];
			printf("\n dExt_value:%p dExt_address:%p ",dExt,&dExt);
			
			int length = dArrNoElemPointerExt[i];
			for (int i = 0; i < length; i++)
			{
				int vi=dExt[i].vi;
				int vj=dExt[i].vj;
				int li= dExt[i].li;
				int lij=dExt[i].lij;
				int lj=dExt[i].lj;
				int vgi=dExt[i].vgi;
				int vgj=dExt[i].vgj;
				int posColumn= dExt[i].posColumn;
				int posRow=dExt[i].posRow;
				printf("\n Thread %d (vi:%d vj:%d li:%d lij:%d lj:%d) (vgi:%d vgj:%d) (posColumn:%d posRow:%d)",i,vi,vj,li,lij,lj,vgi,vgj,posColumn,posRow);
			}
		}
	}
}

//Hàm in mảng dArrPointerExt
inline cudaError_t printdArrPointerExt(EXT **dArrPointerExt,int *dArrNoElemPointerExt,int noElem_dArrPointerExt){
	cudaError_t cudaStatus;
	
	dim3 block(blocksize);
	dim3 grid((noElem_dArrPointerExt + block.x - 1)/block.x);
	kernelprintdArrPointerExt<<<grid,block>>>(dArrPointerExt,dArrNoElemPointerExt,noElem_dArrPointerExt);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize() in printArrPointerdExt() failed");
		goto Error;
	}
Error:
	return cudaStatus;
}

//Hàm giải phóng bộ nhớ Ext** dArrPointerExt và dArr
inline cudaError_t cudaFreeArrPointerExt(EXT **&dArrPointerExt,int *&dArrNoElemPointerExt,int noElem_dArrPointerExt){
	cudaError_t cudaStatus;
	EXT **hArrPointerExt=nullptr;
	hArrPointerExt = (EXT**)malloc(sizeof(EXT*)*noElem_dArrPointerExt);
	if(hArrPointerExt==NULL){
		printf("\n malloc hArrPointerExt in cudaFreeArrpointerExt failed"),
		exit(1);
	}

	cudaStatus = cudaMemcpy(hArrPointerExt,dArrPointerExt,noElem_dArrPointerExt*sizeof(EXT*),cudaMemcpyDeviceToHost);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\cudaMemcpy() in printArrPointerdExt() failed");
		goto Error;
	}

	int length = noElem_dArrPointerExt;
	for (int i = 0; i < length; i++)
	{
		if (hArrPointerExt[i]!=NULL){
			cudaFree(hArrPointerExt[i]);
		}
	}
	cudaFree(dArrPointerExt);
	cudaFree(dArrNoElemPointerExt);

	
Error:
	return cudaStatus;
}


//Hàm giải phóng bộ nhớ Embedding *dArrPointerEmbedding và dArrSizeQ
inline cudaError_t cudaFreeArrPointerEmbedding(Embedding **&dArrPointerEmbedding,int *&dArrSizedQ,int noElem_dArrPointerEmbedding){
	cudaError_t cudaStatus;
	Embedding **hArrPointerExt=nullptr;
	hArrPointerExt = (Embedding**)malloc(sizeof(Embedding*)*noElem_dArrPointerEmbedding);
	if(hArrPointerExt==NULL){
		printf("\n malloc hArrPointerExt in cudaFreeArrPointerEmbedding() failed"),
		exit(1);
	}

	cudaStatus = cudaMemcpy(hArrPointerExt,dArrPointerEmbedding,noElem_dArrPointerEmbedding*sizeof(Embedding*),cudaMemcpyDeviceToHost);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\cudaMemcpy() in cudaFreeArrPointerEmbedding() failed");
		goto Error;
	}

	int length = noElem_dArrPointerEmbedding;
	for (int i = 0; i < length; i++)
	{
		if (hArrPointerExt[i]!=NULL){
			cudaFree(hArrPointerExt[i]);
		}
	}
	cudaFree(dArrPointerEmbedding);
	cudaFree(dArrSizedQ);	
Error:
	return cudaStatus;
}

//Kernel ánh xạ nhãn cạnh sang vị trí tương ứng trong dArrAllPossibleExtension và set giá trị tại đó bằng 1
__global__ void kernelassigndAllPossibleExtension(EXT **dArrPointerExt,int posdArrPointerExt,int Lv,int Le,int *dArrAllPossibleExtension,int noElem_PointerExt){
	int i = blockDim.x *blockIdx.x + threadIdx.x;
	if(i<noElem_PointerExt){
		int lij,lj;
		lij=dArrPointerExt[posdArrPointerExt][i].lij;
		lj=dArrPointerExt[posdArrPointerExt][i].lj;
		int idx=lij*Lv+lj;
		dArrAllPossibleExtension[idx]=1;
	}
}

//Hàm duyệt qua các phần tử trong mảng dExt và set giá trị 1 tại vị trí tương ứng trong mảng kết quả dArrAllPossibleExtension
inline cudaError_t assigndAllPossibleExtension(EXT **dArrPointerExt,int posdArrPointerExt,int Lv,int Le,int *dArrAllPossibleExtension,int noElem_PointerExt){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_PointerExt+block.x -1)/block.x);
	kernelassigndAllPossibleExtension<<<grid,block>>>(dArrPointerExt, posdArrPointerExt, Lv, Le,dArrAllPossibleExtension,noElem_PointerExt);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize kernelassigndAllPossibleExtension in assigndAllPossibleExtension() failed");
		goto Error;
	}

	//In nội dung dArrAllPossibleExtension
	cudaStatus = printInt(dArrAllPossibleExtension,Lv*Le);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n printInt(dArrAllPossibleExtension,Lv*Le) in assigndAllPossibleExtension() failed");
		goto Error;
	}

	
Error:
	return cudaStatus;
}

//Kernel gán giá trị cho mảng dArrUniEdge
__global__ void kernelassigndArrUniEdge(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *dArrUniEdge,int Lv,int *dFromLi){
	int i = blockDim.x*blockIdx.x +threadIdx.x;
	if(i<noElem_dArrAllPossibleExtension){
		if(dArrAllPossibleExtension[i]==1){
			int li,lij,lj;
			li=dFromLi[0];
			lij = i/Lv;
			lj=i%Lv;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].li=li;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lij=lij;
			dArrUniEdge[dArrAllPossibleExtensionScanResult[i]].lj=lj;
		}
	}
}

//Hàm gán giá trị cho mảng dArrUniEdge
inline cudaError_t assigndArrUniEdge(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *&dArrUniEdge,int Lv,int *dFromLi){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_dArrAllPossibleExtension+block.x-1)/block.x);

	kernelassigndArrUniEdge<<<grid,block>>>(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,dArrUniEdge,Lv,dFromLi);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in assigndArrUniEdge() failed");
		goto Error;
	}	
Error:
	return cudaStatus;}

//kernel lấy nhãn from Li
__global__ void kernelGetFromLabel(EXT **dArrPointerExt,int pos,int *dFromLi){
	dFromLi[0]	= dArrPointerExt[pos][0].li;
}

//kernel getPointerUniEdge
__global__ void kernelGetPointerUniEdge(UniEdge **dArrPointerUniEdge,UniEdge *dArrUniEdge,int pos){
	dArrPointerUniEdge[pos]=dArrUniEdge;
}


//Hàm trích các mở rộng duy nhất và lưu kết quả vào mảng dArrPointerUniEdge, mỗi phần tử của nó là một pointer trỏ đến mảng dArrUniEdge trên device
inline cudaError_t extractUniExtension(EXT **dArrPointerExt,int noElem_dArrPointerExt,int Lv,int Le,UniEdge **&dArrPointerUniEdge,int noElem_dArrPointerUniEdge,int *&dArrNoELemPointerUniEdge,int *hArrNoElemPointerExt,int *dArrNoElemPointerExt){
	cudaError_t cudaStatus;
	/*Duyệt qua từng EXTk để thực hiện rút trích và lưu kết quả vào UniEdge **dArrPointerUniEdge
	* Mỗi phần tử của mảng UniEdge **dArrPointerUniEdge là một pointer, chính là kết quả của 1 lần xử lý EXTk
	*	Trích các unique forward extention lưu vào dUniqueEdgeForward
	*	Trích các unique backward extension lưu vào dUniqueEdgeBackward (Backward Extension chỉ tồn tại ở EXTk cuối)
	*/

	//1. Khởi tạo mảng UniEdge **dArrPointerUniEdge với số lượng phần tử bằng kích thước dRMPath				
	//Cấp phát bộ nhớ cho mảng dArrPointerUniEdge
	cudaStatus=cudaMalloc((void**)&dArrPointerUniEdge,sizeof(UniEdge*)*noElem_dArrPointerUniEdge);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrPointerUniEdge in extractUniExtension() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dArrPointerUniEdge,0,sizeof(UniEdge*)*noElem_dArrPointerUniEdge);
	}
	//Cấp phát bộ nhớ cho mảng dArrNoELemPointerUniEdge
	cudaStatus=cudaMalloc((void**)&dArrNoELemPointerUniEdge,sizeof(int)*noElem_dArrPointerUniEdge);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc dArrPointerUniEdge in extractUniExtension() failed",cudaStatus);
		goto Error;
	}
	else
	{
		cudaMemset(dArrNoELemPointerUniEdge,0,sizeof(int)*noElem_dArrPointerUniEdge);
	}
	
	int *hArrNoELemPointerUniEdge=(int*)malloc(sizeof(int)*noElem_dArrPointerUniEdge); //Nơi lưu trữ tạm thời phải được giải phóng cuối hàm này, dữ liệu sẽ được chép sang bộ nhớ dArrNoELemPointerUniEdge
	if(hArrNoELemPointerUniEdge==NULL){
		printf("\n Malloc hArrNoELemPointerUniEdge in extractUniExtension() failed");
		exit(1);
	}
	else
	{
		memset(hArrNoELemPointerUniEdge,0,sizeof(int)*noElem_dArrPointerUniEdge);
	}


	for (int i = 0; i < noElem_dArrPointerExt; i++)
	{		
		//Khai báo bộ nhớ dArrAllPossibleExtension và số lượng phần tử của nó
		int *dArrAllPossibleExtension =nullptr; //Phải được giải phóng bên trong vòng for sau khi dùng xong
		int noElem_dArrAllPossibleExtension = Lv*Le;

		//printf("\n hArrNoElemPointerExt:%d",hArrNoElemPointerExt[i]);
		//Nếu số lượng phần tử tại EXTk lớn hơn bằng minsup thì mới duyệt.
		//Ngon hơn nữa thì xét Số lượng phần tử phân biệt trong EXTk >= minsup thì mới duyệt
		if(hArrNoElemPointerExt[i]>0){

			int *dFromLi;
			cudaStatus = cudaMalloc((void**)&dFromLi,sizeof(int));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc dFromLi in extractUniExtension failed");
				goto Error;
			}
			else
			{
				cudaMemset(dFromLi,0,sizeof(int));
			}


			//lấy nhãn Li lưu vào biến dFromLi
			kernelGetFromLabel<<<1,1>>>(dArrPointerExt,i,dFromLi);
			cudaDeviceSynchronize();
			cudaStatus=cudaGetLastError();
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaDeviceSynchronize()  kernelGetFromLabel in extracUniExtension failed");
				goto Error;
			}
						
			//Hiển thị nội dung nhãn dFromLi
			printf("\n ****dFrom *******\n");
			cudaStatus =printInt(dFromLi,1);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n printInt(dFromLi,1) in extracUniExtension failed");
				goto Error;
			}


			UniEdge * dArrUniEdge=nullptr;
			int noElem_dArrUniEdge=0;

			//Khởi tạo một mảng dArrAllPossileExtension có kích thước bằng Lv*Le với giá trị là zero
			cudaStatus=cudaMalloc((void**)&dArrAllPossibleExtension,noElem_dArrAllPossibleExtension*sizeof(int));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc((void**)&dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
				goto Error;
			}
			else
			{
				cudaMemset(dArrAllPossibleExtension,0,noElem_dArrAllPossibleExtension*sizeof(int));
			}
			//Gọi hàm assigndAllPossibleExtension để ánh xạ (li,lij,lj) sang vị trí trên mảng dArrAllPossibleExtension và set 1 value tại index đó.
			//Gọi kernel gồm hArrNoElemPointerExt[i] threads, mỗi thread sẽ đọc nhãn li,lij,lj và ánh xạ thành vị trí tương ứng trên mảng dArrAllPossibleExtension
			//đồng thời set giá trị 1 tại vị trí trên mảng dArrAllPossibleExtension.
			//
			cudaStatus = assigndAllPossibleExtension(dArrPointerExt,i,Lv,Le,dArrAllPossibleExtension,hArrNoElemPointerExt[i]);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n assigndAllPossibleExtension in extractUniExtension() failed",cudaStatus);
				goto Error;
			}
			
			//Scan mảng dArrAllPossibleExtension để biết kích thước của mảng dArrUniEdge và ánh xạ từ vị trí trong dArrAllPossibleExtension thành nhãn để lưu vào dArrUniEdge
			int *dArrAllPossibleExtensionScanResult =nullptr;
			cudaStatus = cudaMalloc((void**)&dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension*sizeof(int));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc dArrAllPossibleExtensionScanResult in extractUniExtension() failed",cudaStatus);
				goto Error;
			}
			cudaStatus = scanV(dArrAllPossibleExtension,noElem_dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n scanV dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
				goto Error;
			}

			//Tính kích thước của dArrUniEdge và lưu vào noElem_dArrUniEdge
			cudaStatus =getSizeBaseOnScanResult(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,noElem_dArrUniEdge);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n scanV dArrAllPossibleExtension in extractUniExtension() failed",cudaStatus);
				goto Error;
			}

			//HIển thị giá trị của noElem_dArrUniEdge
			printf("\n******noElem_dArrUniEdge************\n");
			//printf("\n noElem_dArrUniEdge:%d",noElem_dArrUniEdge);

			//Cấp phát bộ nhớ cho dArrUniEdge
			cudaStatus = cudaMalloc((void**)&dArrUniEdge,noElem_dArrUniEdge*sizeof(UniEdge));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc dArrUniEdge in extractUniExtension() failed",cudaStatus);
				goto Error;
			}

			//Gọi hàm để ánh xạ dữ liệu từ dArrAllPossibleExtension sang mảng dArrUniEdge
			/* Input Data:	dArrAllPossibleExtension, dArrAllPossibleExtensionScanResult,  */
			cudaStatus =assigndArrUniEdge(dArrAllPossibleExtension,dArrAllPossibleExtensionScanResult,noElem_dArrAllPossibleExtension,dArrUniEdge,Lv,dFromLi);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n assigndArrUniEdge in extractUniExtension() failed",cudaStatus);
				goto Error;
			}

			//In nội dung mảng dArrUniEdge
			printf("\n**********printf************");
			printfUniEdge(dArrUniEdge,noElem_dArrUniEdge);

			//Lưu lại số lượng cạnh duy nhất
			hArrNoELemPointerUniEdge[i]=noElem_dArrUniEdge;

			kernelGetPointerUniEdge<<<1,1>>>(dArrPointerUniEdge,dArrUniEdge,i);
			cudaDeviceSynchronize();

			
			cudaFree(dArrAllPossibleExtensionScanResult);
			cudaFree(dFromLi);
		}		//end if
		cudaFree(dArrAllPossibleExtension);
	} //end for 

	cudaMemcpy(dArrNoELemPointerUniEdge,hArrNoELemPointerUniEdge,sizeof(int)*noElem_dArrPointerUniEdge,cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in extractUniExtension() failed");
		goto Error;
	}
Error:
	free(hArrNoELemPointerUniEdge);
	return cudaStatus;
}

//kernel in nội dung mảngdArrPointerUniEdge
__global__ void kernelprintArrPointerUniEdge(UniEdge **dArrPointerUniEdge,int *dArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElem_dArrPointerUniEdge){
		if(dArrNoELemPointerUniEdge[i]!=0){
			UniEdge * dArrUniEdge = dArrPointerUniEdge[i];
			int n = dArrNoELemPointerUniEdge[i];
			for (int j = 0; j < n; j++)
			{
				printf("\n Thread %d: j:%d (li:%d lij:%d lj:%d)",i,j,dArrUniEdge[j].li,dArrUniEdge[j].lij,dArrUniEdge[j].lj);
			}
		}
	}
}


//Hàm in nội dung mảngdArrPointerUniEdge
inline cudaError_t printArrPointerUniEdge(UniEdge **dArrPointerUniEdge,int *dArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElem_dArrPointerUniEdge+block.x-1)/block.x);
	
	kernelprintArrPointerUniEdge<<<grid,block>>>(dArrPointerUniEdge,dArrNoELemPointerUniEdge,noElem_dArrPointerUniEdge);
	cudaDeviceSynchronize();
	cudaStatus= cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() kernelprintArrPointerUniEdge in printArrPointerUniEdge() failed");
		goto Error;
	}
Error:
	return cudaStatus;
}

//Hàm giải phóng bộ nhớ Ext** dArrPointerUniEdge và dArrNoELemPointerUniEdge
inline cudaError_t cudaFreeArrPointerUniEdge(UniEdge **&dArrPointerUniEdge,int *&dArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge){
	cudaError_t cudaStatus;
	UniEdge **hArrPointerUniEdge=nullptr;
	hArrPointerUniEdge = (UniEdge**)malloc(sizeof(EXT*)*noElem_dArrPointerUniEdge);
	if(hArrPointerUniEdge==NULL){
		printf("\n malloc hArrPointerExt in cudaFreeArrpointerExt failed"),
		exit(1);
	}

	cudaStatus = cudaMemcpy(hArrPointerUniEdge,dArrPointerUniEdge,noElem_dArrPointerUniEdge*sizeof(EXT*),cudaMemcpyDeviceToHost);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\cudaMemcpy() in printArrPointerdExt() failed");
		goto Error;
	}

	int length = noElem_dArrPointerUniEdge;
	for (int i = 0; i < length; i++)
	{
		if (hArrPointerUniEdge[i]!=NULL){
			cudaFree(hArrPointerUniEdge[i]);
		}
	}
	cudaFree(dArrPointerUniEdge);
	cudaFree(dArrNoELemPointerUniEdge);

Error:
	return cudaStatus;
}




__global__ void kernelExtractPointerUniEdge(UniEdge **dPointerArrUniEdge,UniEdge **dArrPointerUniEdge,int pos){
	dPointerArrUniEdge[0] = dArrPointerUniEdge[pos];
	printf("\nPointer UniEdge:%p",dArrPointerUniEdge[pos]);
}

__global__ void kernelExtractPointerExt(EXT **dPointerArrExt,EXT **dArrPointerExt,int pos,unsigned int noElemdArrExt){
	dPointerArrExt[0] = dArrPointerExt[pos];
	printf("\nPointer:%p",dArrPointerExt[pos]);
}

__global__ void kernelfindBoundary(EXT **dPointerArrExt,unsigned int noElemdArrExt,unsigned int *dArrBoundary,unsigned int maxOfVer){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	EXT *dArrExt = dPointerArrExt[0];
	if(i<noElemdArrExt-1){		
		unsigned int graphIdAfter=dArrExt[i+1].vgi/maxOfVer;
		unsigned int graphIdCurrent=dArrExt[i].vgi/maxOfVer;
		if(graphIdAfter!=graphIdCurrent){
			dArrBoundary[i]=1;
		}
	}
}

inline cudaError_t findBoundary(EXT **dPointerArrExt,unsigned int noElemdArrExt,unsigned int *&dArrBoundary,unsigned int maxOfVer){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);


	kernelfindBoundary<<<grid,block>>>(dPointerArrExt,noElemdArrExt,dArrBoundary,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in findBoundary() failed",cudaStatus);
		goto Error;
	}
Error:
	
	return cudaStatus;
}

__global__ void kernelPrint(EXT **dArrExt,unsigned int noElemdArrExt){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt){
		EXT *arrExt = dArrExt[0];
		printf("\nPointer ext:%p",dArrExt[0]);
		printf("\n vgi:%d vgj:%d",arrExt[i].vgi,arrExt[i].vgj);
	}
}

__global__ void kernelPrintUE(UniEdge **dPointerArrUniEdge,unsigned int noElem){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElem){
		UniEdge *arrUniEdge = dPointerArrUniEdge[0];
		printf("\nPointer ue:%p",dPointerArrUniEdge[0]);
		printf("\n UniEdge: li:%d, lij:%d, lj:%d)",arrUniEdge[i].li,arrUniEdge[i].lij,arrUniEdge[i].lj);
	}

}



__global__ void kernelFilldF(UniEdge **dPointerArrUniEdge,unsigned int pos,EXT **dPointerArrExt,unsigned int noElemdArrExt,unsigned int *dArrBoundaryScanResult,float *dF){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<noElemdArrExt){
		UniEdge *dUniEdge = dPointerArrUniEdge[0];
		int li = dUniEdge[pos].li;
		int lij = dUniEdge[pos].lij;
		int lj = dUniEdge[pos].lj;
		EXT *dArrExt = dPointerArrExt[0];
		int Li = dArrExt[i].li;
		int Lij = dArrExt[i].lij;
		int Lj = dArrExt[i].lj;
		printf("\nThread %d: UniEdge(li:%d lij:%d lj:%d) (Li:%d Lij:%d Lj:%d)",i,li,lij,lj,Li,Lij,Lj);

		if(li==Li && lij==Lij && lj==Lj){
			dF[dArrBoundaryScanResult[i]]=1;
		}
	}
}


inline cudaError_t calcSupport(UniEdge **dPointerArrUniEdge,unsigned int pos,EXT **dPointerArrExt,unsigned int noElemdArrExt,unsigned int *dArrBoundaryScanResult,float *dF,unsigned int noElemdF,float &support,unsigned int noElemdArrUniEdge){
	cudaError_t cudaStatus;
	dim3 block(blocksize);
	dim3 grid((noElemdArrExt+block.x-1)/block.x);

	printf("\n**********dPointerArrExt***********\n");
	kernelPrint<<<1,noElemdArrExt>>>(dPointerArrExt,noElemdArrExt);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelPrintExt  in computeSupportv2() failed");
		goto Error;
	}

	printf("\n**********dPointerArrUniEdge***********\n");
	kernelPrintUE<<<1,noElemdArrUniEdge>>>(dPointerArrUniEdge,noElemdArrUniEdge);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n kernelPrintUE  in computeSupportv2() failed");
		goto Error;
	}

	kernelFilldF<<<grid,block>>>(dPointerArrUniEdge,pos,dPointerArrExt,noElemdArrExt,dArrBoundaryScanResult,dF);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n 	cudaDeviceSynchronize() kernelFilldF in calcSupport() failed",cudaStatus);
		goto Error;
	}
	printf("\n**********dF****************\n");
	printFloat(dF,noElemdF);

	reduction(dF,noElemdF,support);

	printf("\n******support********");
	printf("\n Support:%f",support);
	
	cudaMemset(dF,0,noElemdF*sizeof(float));
Error:				
	return cudaStatus;
}


//Hàm tính độ hỗ trợ computeSupportv2
inline cudaError_t computeSupportv2(EXT **dArrPointerExt,int *dArrNoElemPointerExt,int *hArrNoElemPointerExt,int noElem_dArrPointerExt,UniEdge **dArrPointerUniEdge,int *dArrNoELemPointerUniEdge,int *hArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge,unsigned int **&hArrPointerSupport,unsigned int *&hArrNoElemPointerSupport,unsigned int noElem_hArrPointerSupport,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	//Cấp phát bộ nhớ cho hArrPointerSupport. Mỗi phần tử là một địa chỉ trỏ đến 1 mảng kiểu unsigned int
	hArrPointerSupport = (unsigned int**)malloc(sizeof(unsigned int*)*noElem_hArrPointerSupport);
	if(hArrPointerSupport==NULL){
		printf("\n malloc hArrPointerSupport in kernel.cu failed");
		exit(1);
	}
	else
	{
		memset(hArrPointerSupport,0,sizeof(unsigned int*)*noElem_hArrPointerSupport);
	}

	hArrNoElemPointerSupport = (unsigned int*)malloc(sizeof(unsigned int)*noElem_hArrPointerSupport);
	if(hArrNoElemPointerSupport==NULL){
		printf("\n malloc hArrNoelemPointerSupport in computeSupportv2() failed");
		exit(1);
	}
	else
	{
		memset(hArrNoElemPointerSupport,0,sizeof(unsigned int)*noElem_hArrPointerSupport);
	}
	
	//Duyệt qua mảng các pointer trỏ đến mảng chứa các cạnh duy nhất. Mỗi vòng lặp j sẽ ứng với một segment EXTk, và mỗi EXTk sẽ có một boundary 
	for (int j = 0; j < noElem_dArrPointerUniEdge ; j++)
	{
		
		//Mảng dArrBoundary dùng để lưu trữ boundary của EXTk (ở đây là EXT thứ j theo như vòng lặp for bên dưới)
		unsigned int *dArrBoundary=nullptr;
		unsigned int *dArrBoundaryScanResult=nullptr;
		unsigned int noElemdArrBoundary=0; //Bằng với hArrNoElemPointerExt[j]
		if(hArrNoELemPointerUniEdge[j]>0){ //Nếu tồn tại unique edge tại dArrPointerUniEdge j đang xét thì tìm boundary của EXTk j tương ứng
			UniEdge **dPointerArrUniEdge=nullptr;
			unsigned int noElemdArrUniEdge = hArrNoELemPointerUniEdge[j];
			cudaStatus = cudaMalloc((void**)&dPointerArrUniEdge,sizeof(UniEdge*));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\ncudaMalloc dPointerArrUniEdge in computeSupportv2() failed",cudaStatus);
				goto Error;
			}

			
			EXT **dPointerArrExt = nullptr;
			unsigned int noElemdArrExt = hArrNoElemPointerExt[j];
			cudaStatus = cudaMalloc((void**)&dPointerArrExt,sizeof(EXT*));
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\ncudaMalloc dPointerArrExt in computeSupportv2() failed",cudaStatus);
				goto Error;
			}
			//Hoạt động rút trích diễn ra song song
						
			kernelExtractPointerUniEdge<<<1,1>>>(dPointerArrUniEdge,dArrPointerUniEdge,j); //Trích phần tử  trong mảng dArrPointerUniEdge lưu vào biến dArrUniEdge để tiện tính toán
			cudaDeviceSynchronize();
			kernelExtractPointerExt<<<1,1>>>(dPointerArrExt,dArrPointerExt,j,noElemdArrExt); //Trích phần tử trong mảng dArrPointerExt lưu vào biến dArrExt để tiện tính toán
			cudaDeviceSynchronize();
			cudaStatus = cudaGetLastError();
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaDeviceSynchronize() kernelExtractPointerExt kernelExtractPointerUniEdge in computeSupportv2() failed",cudaStatus);
				goto Error;
			}

			printf("\n**********dPointerArrExt***********\n");
			kernelPrint<<<1,noElemdArrExt>>>(dPointerArrExt,noElemdArrExt);
			cudaDeviceSynchronize();
			cudaStatus=cudaGetLastError();
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n kernelPrintExt  in computeSupportv2() failed");
				goto Error;
			}

			printf("\n**********dPointerArrUniEdge***********\n");
			kernelPrintUE<<<1,noElemdArrUniEdge>>>(dPointerArrUniEdge,noElemdArrUniEdge);
			cudaDeviceSynchronize();
			cudaStatus=cudaGetLastError();
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n kernelPrintUE  in computeSupportv2() failed");
				goto Error;
			}

			
					
#pragma region "find Boundary and scan Boundary"
			noElemdArrBoundary = noElemdArrExt;
			cudaStatus=cudaMalloc((void**)&dArrBoundary,sizeof(unsigned int)*noElemdArrBoundary);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc dArrBoundary in computeSupportv2() failed");
				goto Error;
			}
			else
			{
				cudaMemset(dArrBoundary,0,sizeof(unsigned int)*noElemdArrBoundary);
			}

			cudaStatus=cudaMalloc((void**)&dArrBoundaryScanResult,sizeof(unsigned int)*noElemdArrBoundary);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n cudaMalloc dArrBoundary in computeSupportv2() failed");
				goto Error;
			}
			else
			{
				cudaMemset(dArrBoundaryScanResult,0,sizeof(unsigned int)*noElemdArrBoundary);
			}

			//Tìm boundary của EXTk và lưu kết quả vào mảng dArrBoundary
			cudaStatus = findBoundary(dPointerArrExt,noElemdArrExt,dArrBoundary,maxOfVer);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n findBoundary() in computeSupportv2() failed");
				goto Error;
			}

			printf("\n ************* dArrBoundary ************\n");
			cudaStatus=printUnsignedInt(dArrBoundary,noElemdArrBoundary);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n printUnsignedInt in computeSupportv2() failed", cudaStatus);
				goto Error;
			}

			//Scan dArrBoundary lưu kết quả vào dArrBoundaryScanResult
			cudaStatus=scanV(dArrBoundary,noElemdArrBoundary,dArrBoundaryScanResult);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\n Exclusive scan dArrBoundary in computeSupportv2() failed",cudaStatus);
				goto Error;
			}

			printf("\n**************dArrBoundaryScanResult****************\n");
			printUnsignedInt(dArrBoundaryScanResult,noElemdArrBoundary);

			float *dF=nullptr;
			unsigned int noElemdF = 0;

			cudaStatus = cudaMemcpy(&noElemdF,&dArrBoundaryScanResult[noElemdArrBoundary-1],sizeof(unsigned int),cudaMemcpyDeviceToHost);
			if(cudaStatus !=cudaSuccess){
				fprintf(stderr,"\n cudamemcpy dF failed",cudaStatus);
				goto Error;
			}
			noElemdF++;
			printf("\n*****noElemdF******\n");
			printf("noElemdF:%d",noElemdF);

			cudaStatus = cudaMalloc((void**)&dF,sizeof(unsigned int)*noElemdF);
			if(cudaStatus!=cudaSuccess){
				fprintf(stderr,"\ncudaMalloc dF failed",cudaStatus);
				goto Error;
			}
			else
			{
				cudaMemset(dF,0,sizeof(float)*noElemdF);
			}
#pragma endregion "end of finding Boundary"

			hArrNoElemPointerSupport[j]=noElemdArrUniEdge;
			unsigned int * hArrSupport = (unsigned int*)malloc(sizeof(unsigned int)*noElemdArrUniEdge);
			if(hArrSupport==NULL){
				printf("\n Malloc hArrSupport in computeSupportv2() failed");
				exit(1);
			}
			else
			{
				memset(hArrSupport,0,sizeof(unsigned int)*noElemdArrUniEdge);
			}
			//Duyệt và tính độ hỗ trợ của các cạnh
			for (int i = 0; i < noElemdArrUniEdge; i++)
			{					
				float support=0;
				cudaStatus =calcSupport(dPointerArrUniEdge,i,dPointerArrExt,noElemdArrExt,dArrBoundaryScanResult,dF,noElemdF,support,noElemdArrUniEdge);
				if(cudaStatus !=cudaSuccess){
					fprintf(stderr,"\n calcSupport failed",cudaStatus);
					goto Error;
				}				
				hArrSupport[i]=support;
			}			
		    hArrPointerSupport[j]=hArrSupport;
			/*printf("\n***************hArrPointerSupport*************\n");
			for (int i = 0; i < noElemdArrUniEdge; i++)
			{
				printf("\n support:%d ",hArrSupport[i]);
			}*/
		}			
	}


	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaDeviceSynchronize() in computeSupportv2() failed",cudaStatus);
		goto Error;
	}
Error:
	
	return cudaStatus;
}
