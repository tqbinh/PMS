#include "calcSupport.h"
#include "kernelPrintf.h"




__global__ void kernelCalcSupport(int li,int lij,int lj,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,float *d_F){
	int i= blockIdx.x*blockDim.x + threadIdx.x;
	if(i<noElem_d_ValidExtension){
		if(d_ValidExtension[i].li==li && d_ValidExtension[i].lij==lij && d_ValidExtension[i].lj==lj){
			int index=d_scanB_Result[i];
			d_F[index]=1;
		}		
	}
}



cudaError_t calcSupport(Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,float *d_F,unsigned int noElem_F,unsigned int minsup,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg,vector<int> &h_satisfyEdge,vector<int> &h_satisfyEdgeSupport){
	cudaError_t cudaStatus;

	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x-1)/block.x);

	//chép dữ liệu của mảng d_UniqueExtension sang host

	Extension *h_UniqueExtension;
	h_UniqueExtension = new Extension[noElem_d_UniqueExtension];
	if(h_UniqueExtension==NULL){
		printf("\n!!!Memory Problem h_UniqueExtension");
		exit(1);
	}else{
		memset(h_UniqueExtension,0, noElem_d_UniqueExtension*sizeof(Extension));
	}

	cudaMemcpy(h_UniqueExtension,d_UniqueExtension,noElem_d_UniqueExtension*sizeof(Extension),cudaMemcpyDeviceToHost);
	

	for (int i=0;i<noElem_d_UniqueExtension;i++){	
		int li,lij,lj;
		li=h_UniqueExtension[i].li;
		lij=h_UniqueExtension[i].lij;
		lj=h_UniqueExtension[i].lj;		
			
		kernelCalcSupport<<<grid,block>>>(li,lij,lj,d_ValidExtension,noElem_d_ValidExtension,d_scanB_Result,d_F);
		cudaDeviceSynchronize();
		printf("\n[%d] d_F:",i);
		printFloat(d_F,noElem_F);
		float support=0;
		reduction(d_F,noElem_F,support);
		//printf("  Support:%.0f\n",support);

		//Kiểm tra xem độ hỗ trợ có thoả minsup hay không? 
		//Nếu thoả minsup thì kiểm tra xem pattern P có phải là nhỏ nhất hay không? (Đây là hoạt động tuần tự được thực thi trên CPU)
		//Nếu là nhỏ nhất thì mới đi tạo embedding cho pattern P.
		if(support>=minsup){
			//1. Nếu độ hỗ trợ thoả minSup thì xây dựng DFS_Code cho cạnh đó --> cần phải thoả cấu trúc (vi,vj,Li,Lij,Lj)
			//Xây dựng DFS_Code trên device hay trên host? --> xây dựng DFS_Code trên host vì quá trình minDFS_Code diễn tra trên CPU chứ không phải GPU
			//h_frequentEdge[i]=1;
			h_satisfyEdge.push_back(i);		
			h_satisfyEdgeSupport.push_back(support);
			
			//xây dựng embedding cho mở rộng thoả minsup
			//printf("\n***********support of (%d,%d,%d) >= %d --> create embeddings for DFS_CODE************",li,lij,lj,minsup);
			/*cudaStatus=createForwardEmbedding(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,minsup,maxOfVer,numberOfGraph,noDeg);
			if (cudaStatus!=cudaSuccess){
				fprintf(stderr,"\ncreateForwardEmbedding failed");
				goto Error;
			}*/
		}
		
		cudaMemset(d_F,0,noElem_F*sizeof(int));
	}
		
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize function failed");
		goto Error;
	}

Error:
	return cudaStatus;
}


__global__ void kernelGetGraphIdContainEmbedding(int *d_arr_edgeLabel,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *d_arr_graphIdContainEmbedding,unsigned int maxOfVer){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<noElem_d_ValidExtension){
		if(	d_ValidExtension[i].li == d_arr_edgeLabel[0] && d_ValidExtension[i].lij == d_arr_edgeLabel[1] && 	d_ValidExtension[i].lj == d_arr_edgeLabel[2]){
			int graphid = (d_ValidExtension[i].vgi/maxOfVer);
			d_arr_graphIdContainEmbedding[graphid]=1;
		}
	}
}


__global__ void kernelGetGraph(int *d_arr_graphIdContainEmbedding,int noEle_d_arr_graphIdContainEmbedding,int *d_kq,int *d_scanResult){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<noEle_d_arr_graphIdContainEmbedding){
		if(d_arr_graphIdContainEmbedding[i]!=0){
			d_kq[d_scanResult[i]]=i;
		}
	}
}


inline cudaError_t getGraphIdContainEmbedding(int *d_arr_edgeLabel,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *&h_graphIdContainEmbedding,int &noElem_h_arr_graphIdContainEmbedding,unsigned int maxOfVer){
	cudaError_t cudaStatus;

	//Từ global id của đỉnh (vgi hoặc vgj) trong d_ValidExtension chúng ta sẽ tính được graphID chứa mở rộng đó.
	//Các mở rộng trong d_validExtension đã được sắp xếp theo thứ tự từ graphID 0 đến graphId cuối cùng một cách tự nhiên
	//Cần có noElem_d_ValidExtension threads để thực hiện so sánh với nhãn (li,lij,lj), nếu bằng nhau thì sẽ tính graphID=(vgi/maxOfVer) của nó.
	//Set giá trị của mảng d_arr_graphid[graphID]=1;
	//scan mảng d_arr_graphID để thu được index
	//Duyệt qua mảng d_arr_graphid, tại vị trí nào bằng 1 thì ghi giá trị i vào index trong mảng index vừa tính được
	//copy mảng này bỏ vào mảng kết quả.
	dim3 block(blocksize);
	dim3 grid((noElem_d_ValidExtension+block.x)/block.x);
	int noEle_d_arr_graphIdContainEmbedding;
	getLastElementExtension(d_ValidExtension,noElem_d_ValidExtension,noEle_d_arr_graphIdContainEmbedding,maxOfVer);
	noEle_d_arr_graphIdContainEmbedding++;
	//printf("\n noEle_d_arr_graphIdContainEmbedding: %d",noEle_d_arr_graphIdContainEmbedding);

	int *d_arr_graphIdContainEmbedding=NULL;
	cudaStatus=cudaMalloc((void**)&d_arr_graphIdContainEmbedding,noEle_d_arr_graphIdContainEmbedding*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_arr_graphIdContainEmbedding failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_arr_graphIdContainEmbedding,0,noEle_d_arr_graphIdContainEmbedding*sizeof(int));
	}

	//Gọi hàm kernelGetGraphIdContainEmbedding để đánh dấu vị trí đồ thị trong mảng d_arr_graphIdContainEmbedding là 1  
	kernelGetGraphIdContainEmbedding<<<grid,block>>>(d_arr_edgeLabel,d_ValidExtension,noElem_d_ValidExtension,d_arr_graphIdContainEmbedding,maxOfVer);
	cudaDeviceSynchronize();
	printf("\n*************d_arr_graphIdContainEmbedding***************\n");
	printInt(d_arr_graphIdContainEmbedding,noEle_d_arr_graphIdContainEmbedding);

	int *d_scanResult;
	cudaMalloc((void**)&d_scanResult,sizeof(int)*noEle_d_arr_graphIdContainEmbedding);

	scanV(d_arr_graphIdContainEmbedding,noEle_d_arr_graphIdContainEmbedding,d_scanResult);

	printf("\n ************* d_scanResult *************\n");
	printInt(d_scanResult,noEle_d_arr_graphIdContainEmbedding);

	int noElem_kq;	
	getLastElement(d_scanResult,noEle_d_arr_graphIdContainEmbedding,noElem_kq);
	noElem_kq++;

	int *d_kq;
	cudaMalloc((void**)&d_kq,sizeof(int)*noElem_kq);
	
	dim3 blocka(blocksize);
	dim3 grida((noEle_d_arr_graphIdContainEmbedding + blocka.x -1)/blocka.x);

	kernelGetGraph<<<grida,blocka>>>(d_arr_graphIdContainEmbedding,noEle_d_arr_graphIdContainEmbedding,d_kq,d_scanResult);
	cudaDeviceSynchronize();

	printf("\n*********** d_kq ***********\n");
	printInt(d_kq,noElem_kq);

	h_graphIdContainEmbedding=(int*)malloc(sizeof(int)*noElem_kq);
	if(h_graphIdContainEmbedding==NULL){
		printf("\nMalloc h_graphIdContainEmbedding failed");
		exit(1);
	}


	cudaStatus = cudaMemcpy(h_graphIdContainEmbedding,d_kq,sizeof(int)*noElem_kq,cudaMemcpyDeviceToHost);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMemcpy failed",cudaStatus);
		goto Error;
	}

	noElem_h_arr_graphIdContainEmbedding = noElem_kq;

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize function failed");
		goto Error;
	}

Error:
	return cudaStatus;
}
