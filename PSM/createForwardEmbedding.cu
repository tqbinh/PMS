#include "createForwardEmbedding.h"

//__global__ void kernelPrintdArrayQ(struct_QQ Q){
//	printf("\nInside kernelPrintArrayQ:");
//	struct_Q temp;
//	for (int i = 0; i < Q.size; i++)
//	{		
//		temp=Q.structQ[i];
//		printf("\ntemp.size:%d",temp._size);
//		printf("\ntemp._prevQ:%d",temp._prevQ);
//		printf("\ntemp._d_arr_Q:%p",temp._d_arr_Q);
//		printf("\ntemp._d_arr_Q:%p",temp._d_arr_Q);
//		struct_Embedding **temp2=temp._d_arr_Q;
//		printf("\ntemp2[0][0]: (idx:%p",temp2);
//		//for (int j = 0; j < 4; j++)
//		//{
//		//	printf("\ntemp2[0][%d]: (idx:%d",j,temp2);
//		//}
//	}
//}

//struct_QQ convertToStruct(thrust::device_vector<struct_Q> &dArray){
//	struct_QQ Q;
//	Q.structQ=thrust::raw_pointer_cast(&dArray[0]);
//	Q.size=(int)dArray.size();
//	return Q;
//}

//convert thrust device vector to struct_Q
//struct_Q convertDeviceVectorToStruct(thrust::device_vector<struct_Embedding*>&dVecQ,struct_Embedding *d_Q1,int noElemOfd_Q1,int prevQ){
//	struct_Q Q;
//	//Q._d_arr_Q = thrust::raw_pointer_cast(&dVecQ[0]);
//	Q._d_arr_Q = &d_Q1;
//	Q._prevQ=prevQ;
//	Q._size=noElemOfd_Q1;
//	printf("\nInside converDeviceVectorToStruct function:");
//	printf("\nQ._d_arr_Q:%p",*Q._d_arr_Q);
//	return Q;
//}


//__global__ void kernelVector(struct_Q Q,struct_Embedding *d_Q1,int noElem_d_Q){
//	Q._d_arr_Q=&(d_Q1);
//	Q._prevQ=d_Q1->prevQ;
//	Q._size=noElem_d_Q;
//	printf("\nInside kernelVector:");
//	printf("\nd_Q1:%p",d_Q1);
//	printf("\nQ._d_arr_Q:%p",*(Q._d_arr_Q));
//	printf("\nElement of array d_Q:");
//	//printf("\nQ._d_arr_Q[0] value :(idx:%d, vid:%d)",(*(Q._d_arr_Q))->idx,(*(Q._d_arr_Q))->vid);
//	//printf("\nQ._d_arr_Q[1] value:(idx:%d, vid:%d)",((*(Q._d_arr_Q))+1)->idx,((*(Q._d_arr_Q))+1)->vid);
//	//printf("\nQ._d_arr_Q[2] value:(idx:%d, vid:%d)",((*(Q._d_arr_Q))+2)->idx,((*(Q._d_arr_Q))+2)->vid);
//	//printf("\nQ._d_arr_Q[3] value:(idx:%d, vid:%d)",((*(Q._d_arr_Q))+3)->idx,((*(Q._d_arr_Q))+3)->vid);
//	for (int i = 0; i < noElem_d_Q; i++)
//	{
//		printf("\nQ._d_arr_Q[%d] value:(idx:%d, vid:%d)",i,((*(Q._d_arr_Q))+i)->idx,((*(Q._d_arr_Q))+i)->vid);		
//	}
//}


//__global__ void kernelPrintVectorQ(thrust::device_vector<struct_Embedding**> vecQ,int sizeVecQ,int noElem_d_Q){
//
//	int i=blockDim.x*blockIdx.x + threadIdx.x;
//	if (i<sizeVecQ){
//	printf("\nArray is:%p",thrust::raw_pointer_cast(&*vecQ[i]));
//		//cout<<vecQ[0][0]->idx;
//	}
//
//	//int t = blockDim.x*blockIdx.x + threadIdx.x;
//	//if (t<noElem_d_Q){
//	//	for (int j = 0; j < vecQ.size(); j++)
//	//	{
//	//		for (int k = 0; k < noElem_d_Q; k++)
//	//		{
//	//			printf("\nvecQ[%d][%d]: (idx:%d",j,k,vecQ[j][k]->idx);
//	//		}
//	//	}
//
//	//}
//
//}



__global__ void kernelPrintEmbeddingFromLastQ(struct_Embedding **d_arr_Q,int position,int noElem_LastQ){
	int i= threadIdx.x + blockIdx.x * blockDim.x;
	if(i<noElem_LastQ){
		printf("\nd_arr_Q[%d][%d]: (prevQ:%d, idx:%d,vid:%d)",position,i,d_arr_Q[position][i].prevQ,d_arr_Q[position][i].idx,d_arr_Q[position][i].vid);
		//int newi=d_arr_Q[position][i].idx;
		//printf("\nd_arr_Q[%d][%d]: (prevQ:%d, idx:%d,vid:%d)",position,newi,d_arr_Q[position][newi].prevQ,d_arr_Q[position][newi].idx,d_arr_Q[position][newi].vid);		
		int prevQ=d_arr_Q[position][i].prevQ;
		int newi=d_arr_Q[position][i].idx;
		while (true)
		{
			
			printf("\nd_arr_Q[%d][%d]: (prevQ:%d, idx:%d,vid:%d)",prevQ,newi,d_arr_Q[prevQ][newi].prevQ,d_arr_Q[prevQ][newi].idx,d_arr_Q[prevQ][newi].vid);		
			
			if(d_arr_Q[prevQ][newi].prevQ==-1) return;
			newi=d_arr_Q[prevQ][newi].idx;
			prevQ=d_arr_Q[prevQ][newi].prevQ;
		}
	}

}


__global__ void kernelCopy(struct_Embedding *d_Q1,struct_Embedding *d_Q2,struct_Embedding **d_arr_Q){
	d_arr_Q[0]=d_Q1;
	d_arr_Q[1]=d_Q2;
	printf("\n\nInside KernelCopy:");
	printf("\nd_Q1:%p",d_Q1);
	printf("\nd_Q2:%p",d_Q2);
	printf("\nd_arr_Q[0]:%p",d_arr_Q[0]);
	printf("\nd_arr_Q[1]:%p",d_arr_Q[1]);

}



__global__ void kernelPrintArrayEmbedding(struct_Embedding **d_arr_Q,int n,int noElem_Embedding){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		printf("\nd_arr_Q[%d]:%p",i,d_arr_Q[i]);
		for(int j=0;j<noElem_Embedding;j++){
			printf("\n[%d][%d]: (prevQ:%d, idx:%d,vid:%d)",i,j,d_arr_Q[i][j].prevQ,d_arr_Q[i][j].idx,d_arr_Q[i][j].vid);
		}
	}

}


__global__ void kernelMarkExtension(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_M,int li,int lij,int lj){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			d_M[i]=1;
		}		
	}
}


__global__ void kernelMatchLastElement(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj,bool same){
	int lastIndex=noElem_d_ValidExtension-1;
	if(d_ValidExtension[lastIndex].li==li && d_ValidExtension[lastIndex].lij==lij && d_ValidExtension[lastIndex].lj==lj){
		same=true;
	}

}

__global__ void kernelCreateForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanResult,int li,int lij,int lj,struct_Embedding *d_Q1,struct_Embedding *d_Q2){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			d_Q1[d_scanResult[i]].prevQ=-1;
			d_Q1[d_scanResult[i]].vid=d_ValidExtension[i].vgi;

			d_Q2[d_scanResult[i]].prevQ=0;
			d_Q2[d_scanResult[i]].idx=d_scanResult[i];
			d_Q2[d_scanResult[i]].vid=d_ValidExtension[i].vgj;
		}
	}

}

__global__ void kernelPrintd_array_Q(struct_Embedding** d_arr_Q){
	printf("\nd_arr_Q:%p",d_arr_Q);
	printf("\nd_arr_Q[0]:%p",d_arr_Q[0]);
	printf("\nd_arr_Q[1]:%p",d_arr_Q[1]);
}

__global__ void kernelcp(struct_Q *device_arr_Q,int noElem_device_arr_Q,int positionUpdate,struct_Embedding *d_Q,int noElem_d_Q,int prevQ){
	if(positionUpdate<noElem_device_arr_Q && positionUpdate>=0 ){
	device_arr_Q[positionUpdate]._size=noElem_d_Q;
	device_arr_Q[positionUpdate]._prevQ=prevQ;
	device_arr_Q[positionUpdate]._d_arr_Q=d_Q;
	}
}

__global__ void printStructQ(struct_Q *device_arr_Q,int noElem){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if(i<noElem){
		printf("\ndevice_arr_Q[%d]._size:%d",i,device_arr_Q[i]._size);
		printf("\ndevice_arr_Q[%d]._prevQ:%d",i,device_arr_Q[i]._prevQ);
		printf("\ndevice_arr_Q[%d]._d_arr_Q:%p",i,device_arr_Q[i]._d_arr_Q);
		for (int j = 0; j < device_arr_Q[i]._size; j++)
		{
			printf("\n(idx:%d, vid:%d)",(device_arr_Q[i]._d_arr_Q)[j].idx,(device_arr_Q[i]._d_arr_Q)[j].vid);
		}
	}
}


__global__ void PrintAllEmbedding(struct_Q *device_arr_Q,int position,int noElemOfLastColumn){
	int i= threadIdx.x + blockIdx.x * blockDim.x;
	if(i<device_arr_Q[position]._size && position!=0){
		printf("\ndevice_arr_Q[%d]: (prevQ:%d, idx:%d,vid:%d)",position,device_arr_Q[position]._prevQ,device_arr_Q[position]._d_arr_Q[i].idx,device_arr_Q[position]._d_arr_Q[i].vid);
		int prevQ=device_arr_Q[position]._prevQ;
		int newi=device_arr_Q[position]._d_arr_Q[i].idx;
		while (true)
		{			
			printf("\nd_arr_Q[%d]: (prevQ:%d, idx:%d,vid:%d)",prevQ,device_arr_Q[prevQ]._prevQ,device_arr_Q[prevQ]._d_arr_Q[newi].idx,device_arr_Q[prevQ]._d_arr_Q[newi].vid);		
			
			if(device_arr_Q[prevQ]._prevQ==-1) return;
			newi=device_arr_Q[prevQ]._d_arr_Q[i].idx;
			prevQ=device_arr_Q[prevQ]._prevQ;
		}
	}

}

__global__ void kernelGetInformationLastElement(struct_Q *d_arr_Q,int positionLastElement,int *sizeOfLastElement){
	sizeOfLastElement[0]=d_arr_Q[positionLastElement]._size;
	printf("\nsizeOfLastElement:%d",sizeOfLastElement[0]);
}


cudaError_t createForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj){
	cudaError_t cudaStatus;

	thrust::device_vector<struct_Embedding*> dVecQ(1);
	thrust::device_vector<struct_Q> dArrayQ(1);
	

	/*//GPU step: Duyệt qua mảng d_ValidExtension và đánh dấu 1 tại những vị trí có cạnh bằng (li,lij,lj) trong mảng M tương ứng
		1. Tạo mảng M có kích thước bằng với d_ValidExtension và gán giá trị ban đầu cho các phần tử trong M bằng 0.
		2. Tạo noElem_d_ValidExtension threads. Mỗi thread sẽ kiểm tra phần tử tương ứng trong mảng d_ValidExtension xem có bằng cạnh (li,lij,lj) 
			Nếu bằng thì bậc vị trí tại M lên giá trị là 1
		3. Exclusive Scan M để thu được vị trí index cũng như kích thước của mảng Q1 và Q2
		4. Tạo mảng Q1 và Q2 có kích thước là (scanM[LastIndex]) nếu phần tử cuối cùng của d_ValidExtension không phải là (li,lij,lj).
			Ngược lại thì Q có kích thước là (scanM[LastIndex]+1). 
			Mỗi phần tử của Q có cấu trúc là {int idx, int vid}
		5. Tạo mảng các cấu trúc Q1 và Q2 với kích thước tìm được đồng thời gán giá trị cho các phần tử của mảng là -1.
		6. Lưu các embedding của cạnh (li,lij,lj) vào Q1 và Q2, cụ thể như sau:
			6.1. vgi vào vid của Q1
			6.2. vgj vào vid của Q2
			6.3. d_scanResult[i] vào idx Q2
		7. Làm sao duyệt qua được tất cả các Embedding khi có Q2?
	*/

	/*1.Tạo mảng M có kích thước bằng với d_ValidExtension và gán giá trị ban đầu cho các phần tử trong M bằng 0.*/
	int* d_M;
	cudaStatus=cudaMalloc((int**)&d_M,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		exit(1);
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
	dim3 block(1024);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);
	
	kernelMarkExtension<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_M,li,lij,lj);
	cudaDeviceSynchronize();
	printf("\n\nMang d_ValidExtension");
	printfExtension(d_ValidExtension,noElem_d_ValidExtension);
	cudaDeviceSynchronize();
	printf("\nMang d-M:");
	printInt(d_M,noElem_d_ValidExtension);

	/* 3. Exclusive Scan d_M
		Kết quả scan lưu vào mảng d_scanResult
	*/
	int* d_scanResult;
	cudaStatus=cudaMalloc((int**)&d_scanResult,noElem_d_ValidExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc M failed");
		exit(1);
	}
	else
	{
		cudaMemset(d_scanResult,0,noElem_d_ValidExtension*sizeof(int));
	}

	cudaStatus=scanV(d_M,noElem_d_ValidExtension,d_scanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanV() d_M createForwardEmbedding failed");
		exit(1);
	}

	/*
	4. Tạo mảng Q1 và Q2 có kích thước là (scanM[LastIndex]) nếu phần tử cuối cùng của d_ValidExtension không phải là (li,lij,lj).
			Ngược lại thì Q có kích thước là (scanM[LastIndex]+1). 
			Mỗi phần tử của Q có cấu trúc là {int idx, int vid}
	*/
	bool same = false;
	kernelMatchLastElement<<<1,1>>>(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,same);

	int noElem_d_Q=0;
	
	cudaStatus=getLastElement(d_scanResult,noElem_d_ValidExtension,noElem_d_Q);

	if (same==true){
		noElem_d_Q++;
	}

	
	printf("\nnoElem_d_Q1:%d",noElem_d_Q);

	/*
		5. Tạo mảng các cấu trúc Q1 và Q2 với kích thước tìm được đồng thời gán giá trị cho các phần tử của mảng là -1.
	*/
	struct_Embedding *d_Q1=NULL;
	cudaStatus=cudaMalloc((struct_Embedding**)&d_Q1,noElem_d_Q*sizeof(struct_Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc Embedding failed");
		exit(1);
	}
	else
	{
		cudaMemset(d_Q1,-1,noElem_d_Q*sizeof(struct_Embedding));
	}

	struct_Embedding *d_Q2=NULL;
	cudaStatus=cudaMalloc((struct_Embedding**)&d_Q2,noElem_d_Q*sizeof(struct_Embedding));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc Embedding failed");
		exit(1);
	}
	else
	{
		cudaMemset(d_Q2,-1,noElem_d_Q*sizeof(struct_Embedding));
	}
	
	/*
		6. Lưu các embedding của cạnh (li,lij,lj) vào Q1 và Q2, cụ thể như sau:
			6.1. vgi vào vid của Q1
			6.2. vgj vào vid của Q2
			6.3. d_scanResult[i] vào idx Q2
	*/

	kernelCreateForwardEmbedding<<<grid,block>>>(d_ValidExtension,noElem_d_ValidExtension,d_scanResult,li,lij,lj,d_Q1,d_Q2);
	cudaDeviceSynchronize();

	printf("\nEmbedding:\nd_Q1");
	printEmbedding(d_Q1,noElem_d_Q);
	printf("\nd_Q2:");
	printEmbedding(d_Q2,noElem_d_Q);

	//wrap_pointer from raw_pointer to device pointer
	thrust::device_ptr<struct_Embedding> dev_ptr(d_Q1);
	thrust::device_vector<struct_Embedding> Vec(dev_ptr,dev_ptr+noElem_d_Q);
	printf("\nSo luong phan tu cua Vec:%d",Vec.size());
	
	printf("\nd_Q1:%p",d_Q1);
	printf("\ndev_prt:%p",dev_ptr);
	for (int i = 0; i < Vec.size(); i++)
	{
		//printf("\nVec[%d]: (raw_pointer:%p",i,(thrust::raw_pointer_cast(&Vec[i])));
		printf("\nVec[%d]:%p",i,Vec[i]);
	}

	//unwrap pointer from device pointer to raw pointer
	struct_Embedding *raw_ptr = thrust::raw_pointer_cast(&Vec[0]);
	//printf("\nraw_ptr:%p",raw_ptr);
	//printf("\n\nValue of raw_ptr:");
	//printEmbedding(raw_ptr,noElem_d_Q);
	 
	
	//Tạo mảng d_arr_Q, mỗi phần tử của d_arr_Q sẽ trỏ tới địa chỉ của vùng nhớ được trỏ tới bởi d_Q1 và d_Q2
	struct_Embedding **d_arr_Q=NULL;
	cudaStatus=cudaMalloc((void**)&d_arr_Q,sizeof(struct_Embedding*)*2);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_arr_Q failed");
		exit(1);
	}
	
	//kernelCopy<<<1,1>>>(d_Q1,d_Q2,d_arr_Q); 
	
	cudaStatus=cudaMemcpy(d_arr_Q,&d_Q1,sizeof(struct_Embedding*)*1,cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMemcpy d_arr_Q failed");
		//goto Error;
		exit(1);
	}
	else
	{
		printf("\nCopy successful");
	}
	
	cudaStatus=cudaMemcpy(d_arr_Q+1,&(d_Q2),sizeof(struct_Embedding*)*1,cudaMemcpyHostToDevice);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMemcpy d_arr_Q failed");
		//goto Error;
		exit(1);
	}
	else
	{
		printf("\nCopy successful");
	}

	//printf("\nd_Q1:%p",d_Q1);
	//printf("\nd_Q2:%p",d_Q2);
	//kernelPrintd_array_Q<<<1,1>>>(d_arr_Q);

	//cudaDeviceSynchronize();

	//Copy d_arr_Q to d_arr_new_Q: from device to device memory
	//printf("\n\n d_arr_Q:");
	//kernelPrintArrayEmbedding<<<1,2>>>(d_arr_Q,2,noElem_d_Q);
	
	/*
	//Tạo một mảng mới có kích thước bằng d_arr_Q và sao chép d_arr_Q sang mảng mới
	struct_Embedding **d_arr_new_Q=NULL;
	cudaStatus=cudaMalloc((void**)&d_arr_new_Q,sizeof(struct_Embedding*)*2);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMallocd_arr_new_Q failed");
		//goto Error;
		exit(1);
	}

	
	cudaStatus=cudaMemcpy(d_arr_new_Q,d_arr_Q,sizeof(struct_Embedding*)*2,cudaMemcpyDeviceToDevice);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\\ncudaMemcpy failed");
		//goto Error;
		exit(1);
	}
	cudaDeviceSynchronize();
	printf("\nd_arr_new_Q:");
	kernelPrintd_array_Q<<<1,1>>>(d_arr_new_Q);
	cudaFree(d_arr_Q);
	kernelPrintArrayEmbedding<<<1,2>>>(d_arr_new_Q,2,noElem_d_Q);
	*/
	
	//Tạo một mảng device_Array_Q có kiểu là struct_Q
	struct_Q *device_arr_Q=NULL;
	cudaMalloc((void**)&device_arr_Q,sizeof(struct_Q)*2);
	//Vì device_arr_Q là một device pointer nên để truy cập phần tử của chúng thì chúng ta cần phải sử dụng kernel
	//Chúng ta tạo kernel để chép dữ liệu từ d_Q1 vào device_array_Q
	int prevQ=-1;	
	int positionUpdate=0;
	kernelcp<<<1,1>>>(device_arr_Q,2,positionUpdate,d_Q1,noElem_d_Q,prevQ);
	//cudaMemset(d_Q1,0,sizeof(struct_Embedding)*noElem_d_Q); 
	positionUpdate=1;
	prevQ=0;
	kernelcp<<<1,1>>>(device_arr_Q,2,positionUpdate,d_Q2,noElem_d_Q,prevQ);
	//printf("\n\nPrint device_arr_Q:");
	//printStructQ<<<1,2>>>(device_arr_Q,2);



	printf("\nPrint information of size of the last element of d_arr_Q:");	
	int positionLastElement = 1;
	int *dsizeOfLastElement,*hsizeOfLastElement;
	hsizeOfLastElement=(int*)malloc(sizeof(int));
	cudaMalloc((void**)&dsizeOfLastElement,sizeof(int));
	cudaMemset(dsizeOfLastElement,0,sizeof(int));
	
	kernelGetInformationLastElement<<<1,1>>>(device_arr_Q,positionLastElement,dsizeOfLastElement);
	cudaDeviceSynchronize();
	cudaMemcpy(hsizeOfLastElement,dsizeOfLastElement,sizeof(int),cudaMemcpyDeviceToHost);
	printf("\nhsizeOfLastElement:%d",hsizeOfLastElement[0]);

		//Làm sao để truy xuất tất cả các Embeddings khi truyền vào một mảng cấu trúc struct_Q: device_arr_Q
	printf("\n\nPrint all embedding from the last element of device_arr_Q");
	PrintAllEmbedding<<<1,hsizeOfLastElement[0]>>>(device_arr_Q,1,hsizeOfLastElement[0]);

	////Làm sao mở rộng kích thước của mảng device_arr_Q
	//struct_Q *device_arr_newQ=NULL;
	//cudaMalloc((void**)&device_arr_newQ,sizeof(struct_Q)*3);
	//device_arr_newQ=device_arr_Q;
	////printf("\nDevice array of new Q:");
	////printStructQ<<<1,3>>>(device_arr_newQ,3);
	//kernelcp<<<1,1>>>(device_arr_newQ,3,2,d_Q1,noElem_d_Q);




	//Truy xuat tat ca embedding from d_Q2
	//kernelPrintEmbeddingFromLastQ<<<1,noElem_d_Q>>>(d_arr_Q,1,noElem_d_Q);
	//dVecQ[0]=d_Q1;
	//printf("\ndVecQ[0]:%p",dVecQ[0]);
	//convertDeviceVectorToStruct(dVecQ);
	
	
	
	

	//kernelVector<<<1,1>>>(convertDeviceVectorToStruct(dVecQ),d_Q1,noElem_d_Q);
	//Chuyển dVecQ đang chứa d_Q1 thành struct_Q, sau đó đưa nó vào phần tử đầu tiên của dVecQ là dArrayQ[0]
	//dArrayQ[0]=(convertDeviceVectorToStruct(dVecQ,d_Q1,noElem_d_Q,-1));
	//dArrayQ.resize(2);
	//dArrayQ[1]=(convertDeviceVectorToStruct(dVecQ,d_Q2,noElem_d_Q,1));

	//printf("\ndArrayQ[0]:%p",dArrayQ[0]);
	//printf("\ndArrayQ[1]:%p",dArrayQ[1]);
	//convertToStruct(dArrayQ);
	
	//struct_QQ structQ=convertToStruct(dArrayQ);

	//kernelPrintdArrayQ<<<1,1>>>(structQ);

	//dVecQ.resize(2);
	//dVecQ[1]=d_Q2;


	//cudaDeviceSynchronize();
	//cudaStatus=cudaGetLastError();
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\ncudaDeviceSynchronize() failed");
	//	goto Error;
	//}
//Error:
	//cudaFree(d_M);
	//cudaFree(d_Q1);
	//cudaFree(d_Q2);
	return cudaStatus;
}
