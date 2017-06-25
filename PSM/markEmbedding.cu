#pragma once
#include "markEmbedding.cuh"


__global__ void kernelMarkEmbedding(cHistory **dH,struct_Q *device_arr_Q,int lastColumn,int n,unsigned int maxOfVer,int *d_O,int *d_N){
	int i= blockDim.x*blockIdx.x + threadIdx.x; //mỗi thread i sẽ xử lý một embedding
	if(i<n){
		int vid = device_arr_Q[lastColumn]._d_arr_Q[i].vid; // Từ cột Q cuối cùng, mỗi thread i sẽ xử lý embedding thứ i
		int indexOfFirstVertexInGraph = vid-(vid%maxOfVer);
		int toVid = vid;//đỉnh to của cạnh thuộc embedding
		int idxOfdH= (vid%maxOfVer);
		dH[i]->d_arr_HO[idxOfdH]=2;
		int prevQ=device_arr_Q[lastColumn]._prevQ;
		int newi=device_arr_Q[lastColumn]._d_arr_Q[i].idx;
		while (true)
		{			
			//printf("\nd_arr_Q[%d]: (prevQ:%d, idx:%d,vid:%d)",prevQ,device_arr_Q[prevQ]._prevQ,device_arr_Q[prevQ]._d_arr_Q[newi].idx,device_arr_Q[prevQ]._d_arr_Q[newi].vid);		
			
			vid = device_arr_Q[prevQ]._d_arr_Q[newi].vid;
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
						
			idxOfdH = (vid%maxOfVer); //Đánh dấu đỉnh thuộc Embedding
			dH[i]->d_arr_HO[idxOfdH]=2;

			dH[i]->d_arr_HLN[idxEdge]=2;//Đánh dấu cạnh thuộc Embedding. vì đây là đơn đồ thị vô hướng nên cạnh AB cũng bằng cạnh BA,do đó ta phải đánh dấu cạnh BA cũng thuộc embedding.
			dH[i]->d_arr_HLN[indexOfEdgeR]=2;

			if(device_arr_Q[prevQ]._prevQ==-1) return; //nếu là cột Q đầu tiên thì dừng lại vì đã duyệt xong embedding
			newi=device_arr_Q[prevQ]._d_arr_Q[i].idx; //ngược lại thì lấy index của cột Q phía trước
			prevQ=device_arr_Q[prevQ]._prevQ; //Lấy Q phía trước
			toVid=fromVid; //cập nhật lại đỉnh to.
		}
	}

}


cudaError_t markEmbedding(cHistory **dH,struct_Q *device_arr_Q,int lastColumn,int n,unsigned int maxOfVer,int *d_O,int *d_N){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((n+block.x-1)/block.x);
	/*printf("\****************ndH arr***********"); //kiểm tra thử dữ liệu của mảng dH trên device xem có đúng không
	kernelPrintdeviceH<<<1,1>>>(dH,n);*/
	kernelMarkEmbedding<<<grid,block>>>(dH,device_arr_Q,lastColumn,n,maxOfVer,d_O,d_N);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize markEmbedding failed");
		goto Error;
	}

	//printf("\****************ndH arr***********"); //kiểm tra thử dữ liệu của mảng dH trên device sau khi đã đánh dấu các embedding thuộc right most path
	//kernelPrintdeviceH<<<1,1>>>(dH,n);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize markEmbedding failed");
		goto Error;
	}
Error:

	return cudaStatus;
}
