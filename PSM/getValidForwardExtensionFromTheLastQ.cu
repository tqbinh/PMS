#include "getValidForwardExtensionFromTheLastQ.h"

__global__ void kernelPrintd_arr_V(struct_V *d_arr_V,int numberElementOf_d_arr_V){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<numberElementOf_d_arr_V){
		//if(d_arr_V[i].valid==1){
			printf("\n Thread %d: valid: %d, d_backward: %d",i,d_arr_V[i].valid,d_arr_V[i].backward);
		//}		
	}
}


__global__ void kernelFindValidForwardFromLastQ(struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,int *d_O,int *d_LO,int *d_N,struct_V *d_arr_V,float *d_arr_degreeOfVerticesInQColumn, int maxOfVer,int m){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<n){
		int minLabel = d_LO[device_arr_Q[0]._d_arr_Q[0].vid];
		printf("\n minLabel: %d",minLabel);
		// diplay array dH
		/*		
		//dH[i]->printmn();
		printf("\n dh[%d]->m:%d",i,dH[i]->m);
		printf("\n dh[%d]->n:%d",i,dH[i]->n);
		for (int j = 0; j < dH[i]->n; j++) //display d_arr_HO
		{
		printf("\n dH[%d]->d_arr_HO[%d]:%d",i,j,dH[i]->d_arr_HO[j]);
		}
		for (int j = 0; j < dH[i]->m; j++) //display d_arr_HLN
		{
		printf("\n dH[%d]->d_arr_HLN[%d]:%d",i,j,dH[i]->d_arr_HLN[j]);
		}
		*/
		int vid = device_arr_Q[indexOfQ]._d_arr_Q[i].vid; //lấy vid của cột Q
		//int indexOfPrevQ = device_arr_Q[indexOfQ]._d_arr_Q[i].idx; //Tạm thời không lấy index của Q phía trước
		int degreeVid = __float2int_rn(d_arr_degreeOfVerticesInQColumn[i]); //lấy bậc của vid đó, do bậc là kiểu float nên phải convert sang kiểu int
		printf("\n Thread %d: vid:%d have degree: %d",i,vid,degreeVid);
		//Duyệt qua các đỉnh kề với đỉnh vid dựa vào số lần duyệt là bậc
		int indexToVidIndN=d_O[vid];
		int toVid;
		int labelToVid;
		for (int j = 0; j < degreeVid; j++,indexToVidIndN++) //Duyệt qua tất cả các đỉnh kề với đỉnh vid, nếu đỉnh không thuộc embedding thì --> cạnh cũng không thuộc embedding vì đây là Q cuối
		{			
			toVid=d_N[indexToVidIndN]; //Lấy vid của đỉnh cần kiểm tra
			labelToVid = d_LO[toVid]; //lấy label của đỉnh cần kiểm tra
			//printf("\nThread %d, j: %d has ToVidLabel:%d",i,j,labelToVid);
			//kiểm tra xem đỉnh toVid đã tồn tại trong embedding hay chưa (khác zero là thuộc embedding)
			int indexOfToVidInEmbedding=(toVid%maxOfVer);
			//printf("\n Thread %d, for j: %d, dH[%d]->d_arr_HO[%d]:%d",i,j,i,indexOfToVidInEmbedding,dH[i]->d_arr_HO[indexOfToVidInEmbedding]);
			if(dH[i]->d_arr_HO[indexOfToVidInEmbedding]==0){ //Nếu giá trị tương ứng trên Embedding bằng zero thì xét xem label của nó có thoả lớn hơn hoặc bằng minLabel hay không
				if(labelToVid>=minLabel){ //nếu thoả thì sẽ set mảng V tương ứng là 1 và chỉ định nó là forward
					int indexOfd_arr_V=i*m+j;
					d_arr_V[indexOfd_arr_V].valid=1;					
				}
			}
		}
	}
}


__global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}


__global__ void kernelFindDegreeOfVertex(int *d_O,int *d_N,int numberOfElementd_O,int numberOfElementd_N,struct_Q *device_arr_Q,int indexOfQ,int n,float *d_arr_degreeOfVerticesInQColumn,int maxOfVer){
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i<n){
		float degreeOfV =0;
		int nextVid;
		int graphid;
		int lastGraphId=(numberOfElementd_O-1)/maxOfVer;
		int vid =device_arr_Q[indexOfQ]._d_arr_Q[i].vid;
		if(d_O[vid]==-1){
			printf("\ndevice_arr_Q is not correct, vertex id %vid is not exist in database");
			return;
		}

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
		//printf("\nThread:%d : Degree of %d is %f",i,vid,degreeOfV);
		d_arr_degreeOfVerticesInQColumn[i]=degreeOfV;
		//printf("\nThread %d: d_arr_degreeOfVerticesInQColumn[%d]:%f",i,i,d_arr_degreeOfVerticesInQColumn[i]);
	}		
}


cudaError_t getValidForwardExtensionFromTheLastQ(struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,unsigned int maxOfVer,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((n+block.x-1)/block.x);

	//1. Tìm bậc lớn nhất m của các vid thuộc device_arr_Q[indexOfQ] đang xét.
	//1.1 Khởi tạo một mảng số nguyên có kích thước bằng số lượng embedding
	float *d_arr_degreeOfVerticesInQColumn;
	cudaStatus = cudaMalloc((void**)&d_arr_degreeOfVerticesInQColumn,n*sizeof(float));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_arr_degreeOfVerticeInQColumn failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_arr_degreeOfVerticesInQColumn,0,n*sizeof(float));
	}

	//1.2 Tính bậc của các đỉnh vid trong Q column và lưu vào d_arr_OfVerticeInQColumn
	kernelFindDegreeOfVertex<<<grid,block>>>(d_O,d_N,numberOfElementd_O,numberOfElementd_N,device_arr_Q,indexOfQ,n,d_arr_degreeOfVerticesInQColumn,maxOfVer);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize kernelFindDegreeOfVertex failed");
		goto Error;
	}

	//2. Tìm bậc lớn nhất của vid trong Q column chính là tìm giá trị lớn nhất trong mảng d_arr_degreeOfVerticesInQColumn
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
	find_maximum_kernel<<<gridSize, blockSize>>>(d_arr_degreeOfVerticesInQColumn, d_max, d_mutex, n);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize find_maximum_kernel failed");
		goto Error;
	}

	// copy from device to host
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

	//report results
	int m = (int)(*h_max); //bậc lớn nhất của các đỉnh trong 1 cột Q
	printf("\nMax degree of vid in Q column is: %d",m);

	/*
	//3. Tạo mảng d_arr_V có kích thước: maxDegree_vid_Q * |Q|
			Lưu ý, mảng d_arr_V phải có dạng cấu trúc đủ thể hiện cạnh mở rộng có hợp lệ hay không và là forward extension hay backward extension.
			struct struct_V
			{
				int valid; //default: 0, valid: 1
				int backward; //default: 0- forward; backward: 1
			}
			*/
	struct_V *d_arr_V;
	int numberElementOf_d_arr_V=m*n;
	cudaStatus=cudaMalloc((void**)&d_arr_V,numberElementOf_d_arr_V*sizeof(struct_V));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n cudaMalloc d_arr_V failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_arr_V,0,numberElementOf_d_arr_V*sizeof(struct_V));
	}

	/*
	//4. Tìm các mở rộng của vid và đánh dấu những mở rộng hợp lệ vào mảng d_arr_V
		o Bậc của các đỉnh trong Q column được lưu trữ trong mảng d_arr_degreeOfVerticesInQColumn--> chúng ta không cần tính bậc của vid
		o cHistory được lưu trữ trong dH là một cấu trúc gồm mảng d_HO và d_HLN cho biết cạnh và đỉnh đã thuộc embedding
		o Thread thứ i sẽ sử dụng các phần tử tương ứng index_d_arr_V từ [i*m,(i+1)*m - 1]
		o Mỗi lần lặp bậc của vid thì biến tạm sẽ tăng lên 1 để chỉ vùng nhớ tương ứng trên d_arr_V
		o Nếu đỉnh phải cùng của DFS_Code kết nối trực tiếp với đỉnh đầu tiên của DFS_Code thì không tồn tại backward edge (chỉ đúng trong đơn đồ thị vô hướng).
	*/
	kernelFindValidForwardFromLastQ<<<grid,block>>>(device_arr_Q,indexOfQ,dH,n,d_O,d_LO,d_N,d_arr_V,d_arr_degreeOfVerticesInQColumn,maxOfVer,m);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize kernelFindValidForwardFromLastQ failed");
		goto Error;
	}
	
	//Hiển thị kết quả mảng d_arr_V với số lượng phần tử numberElementOf_d_arr_V
	kernelPrintd_arr_V<<<1,numberElementOf_d_arr_V>>>(d_arr_V,numberElementOf_d_arr_V);
	
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize getValidExtensionFromEmbedding failed");
		goto Error;
	}
Error:

	return cudaStatus;
}
