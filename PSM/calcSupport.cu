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



cudaError_t calcSupport(Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,float *d_F,unsigned int noElem_F,unsigned int minsup,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg){
	cudaError_t cudaStatus;

	dim3 block(1024);
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
			//xây dựng embedding cho mở rộng thoả minsup
			printf("\n***********support of (%d,%d,%d) >= %d --> create embeddings for DFS_CODE************",li,lij,lj,minsup);
			cudaStatus=createForwardEmbedding(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,minsup,maxOfVer,numberOfGraph,noDeg);
			if (cudaStatus!=cudaSuccess){
				fprintf(stderr,"\ncreateForwardEmbedding failed");
				goto Error;
			}
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
