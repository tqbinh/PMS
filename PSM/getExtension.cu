#include "getExtension.h"


__global__ void kernelUpdateData(ArrHistory *d_arrH,cHistory *h_d_Hi,int i){
	d_arrH->vecA[i]=h_d_Hi;
}


__global__ void kernelPrintd_d_obj(cHistory **d_d_obj,int n,int *d_arr_number_HLN){
	for (int i = 0; i < n; i++)
	{
		printf("\nd_d_obj[%d] n:%d m:%d d_arr_HO:%p",i,d_d_obj[i]->n,d_d_obj[i]->m,d_d_obj[i]->d_arr_HO);
		/*for (int j = 0; j < d_arr_number_HLN[i]; j++)
		{
			printf("\nd_HLN:%d",d_d_obj[i]->d_arr_HO[j]);
		}*/

	}
}


__global__ void kernelPrintd_h(cHistory **d_h,int noEle){
	for (int i = 0; i < noEle; i++)
	{
		d_h[0]->print();
	}

}

__global__ void kernelPrintArrHistory(ArrHistory *d_arrH,int n){	
	printf("\n***inside kernelPrintArrHistory" );
	printf("\nvalue of n:%d",d_arrH->n);
	printf("\npointer vecA:%p",d_arrH->vecA);
	printf("\nvalue n of vecA:%d",d_arrH->vecA[0]->n);
		//d_arrH->print();
	//printf("\nd_arrH:%p",d_arrH);
		//for (int i = 0; i < n; i++)
		//{
		//	//printf("\n vecA[%d].m:%p",i,d_arrH->vecA[i]->m);
		//	//printf("\n vec[%d]:%p",i,d_arrH->vecA[i]->d_arr_HO);
		//	//d_arrH->vecA[i]->print();
		//}
}


void cHistory::print(){
	for (int i = 0; i < n; i++)
	{
		
		printf("\nd_arr_HO[%d]:%d - %p",i,d_arr_HO[i],&d_arr_HO[i]);
	}
	for (int i = 0; i < m; i++)
	{
		printf("\nd_arr_HLN[%d]:%d - %p",i,d_arr_HLN[i],&d_arr_HLN[i]);
	}
}

void cHistory::printmn(){
	printf("\n n:%d m:%d",n,m);
}


cHistory::cHistory(){
	n=0;
	m=0;
	d_arr_HO=NULL;
	d_arr_HLN=NULL;
}


cHistory::cHistory(int _n,int _m){
	n=_n;
	m=_m;

	d_arr_HO = (int*)malloc(sizeof(int)*n);
	if(d_arr_HO==NULL){
		printf("\nMalloc d_arr_HO failed");
		exit(1);
	}
	else
	{
		memset(d_arr_HO,0,sizeof(int)*n);
	}

	d_arr_HLN=(int*)malloc(sizeof(int)*m);
	if(d_arr_HLN==NULL){
		printf("\nMalloc d_arr_HLN failed");
		exit(1);
	}
	else
	{
		memset(d_arr_HLN,0,sizeof(int)*m);
	}
}

ArrHistory::ArrHistory(){
	n=0;
	vecA=NULL;
}

ArrHistory::ArrHistory(int _n){
	
	n=_n; //số lượng phần tử của vecA
	vecA=NULL;
	vecA=new cHistory*[n]; //dynamic array (size = n) of pointer to an object cHistory
	if(vecA==NULL){
		printf("\nMalloc failed");
		exit(1);
	}	
}

void ArrHistory::print()	
{
	printf("\n Number of element of vecA is:%d \n Address of vecA:%p \n Below is the pointer to array of object:",n,vecA);
	for (int i = 0; i < n; i++)
	{
		printf("\n Value of vecA[%d]:%p",i,vecA[i]); //vecA[i] lưu trữ địa chỉ của một đối tượng cHistory. hay nói cách khác là nó trỏ đến đối tượng cHistory
		vecA[i]->print();
	}
}

__global__ void kernelFindNumberOfEdgeInAGraph(int *d_arr_number_HLN,struct_Q *device_arr_Q,int numberEmbedding,int lastColumn,unsigned int maxOfVer,int *d_O,unsigned int numberOfGraph,unsigned int noDeg){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<numberEmbedding){
		//printf("\n vid:%d",device_arr_Q[lastColumn]._d_arr_Q[i].vid);
		int graphId=(device_arr_Q[lastColumn]._d_arr_Q[i].vid)/maxOfVer;
		//printf("\n vid:%d, graphid:%d",device_arr_Q[lastColumn]._d_arr_Q[i].vid,graphId);
		int idxFrom = graphId*maxOfVer;		
		int idxFirstNext = (graphId+1)*maxOfVer;
		int r=0;
		//printf("\n i:%d, vid:%d, r:%d",i,device_arr_Q[lastColumn]._d_arr_Q[i].vid,r);
		if (graphId!=(numberOfGraph-1)){
			//printf("\nidxFirstNext:%d",idxFirstNext);
			r=d_O[idxFirstNext]-d_O[idxFrom];
		}else
		{
			r=noDeg-d_O[idxFrom];
		}
		//printf("\n i:%d, vid:%d, r:%d",i,device_arr_Q[lastColumn]._d_arr_Q[i].vid,r);
		d_arr_number_HLN[i]=r;
	}
}


cudaError_t	findNumberOfEdgeInAGraph(int *d_arr_number_HLN,struct_Q *device_arr_Q,int numberEmbedding,int lastColumn,unsigned int maxOfVer,int *d_O,unsigned int numberOfGraph,unsigned int noDeg){
	cudaError_t cudaStatus;

	dim3 block(1024);
	dim3 grid((numberEmbedding+block.x-1)/block.x);
	printf("\nCall kernelFindNumberOfEdgeInAGraph");
	kernelFindNumberOfEdgeInAGraph<<<grid,block>>>(d_arr_number_HLN,device_arr_Q,numberEmbedding,lastColumn,maxOfVer,d_O,numberOfGraph,noDeg);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus!=cudaSuccess)
	{
		fprintf(stderr,"\ncudaDeviceSynchronize findNumberOfEdgeInAGraph failed");
		goto Error;
	}
Error:
	return cudaStatus;
}


__global__ void	kernelGetnoEle_Embedding(struct_Q *device_arr_Q,int lastColumn,int *noEle_Embeddings){
	noEle_Embeddings[0]=device_arr_Q[lastColumn]._size;
	printf("\nInside kernelGetnoEle_Embedding:%d",noEle_Embeddings[0]);
}

__global__ void kernelPrintdeviceH(cHistory **device_H,int numberEmbeddings){
	for (int i = 0; i < numberEmbeddings; i++)
	{
		printf("\n**************dH[%d]:%p**********",i,&device_H[i]);		
		////device_H[i]->print();
		//printf("\n n:%d",device_H[i]->n);
		//printf("\n m:%d",device_H[i]->m);
		//for (int j = 0; j < device_H[i]->n; j++)
		//{
		//	printf("\nHO[%d]:%d",j,device_H[i]->d_arr_HO[j]);
		//}
		//for (int j = 0; j < device_H[i]->m; j++)
		//{
		//	printf("\nHLN[%d]:%d",j,device_H[i]->d_arr_HLN[j]);
		//}
		device_H[i]->printmn();
		device_H[i]->print();
	}
}




cudaError_t getExtension(struct_Q *device_arr_Q,int lastColumn,vector<struct_DFS> &P,vector<int> &RMPath,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int minsup,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg){
	cudaError_t cudaStatus;

	/*
		Dữ liệu truyền vào gồm: CSDL(d_O,d_LO,d_N,d_LN), ngưỡng minsup, pattern P và các Embeddings của P(device_arr_Q), Right Most Path (vector<int> RMPath)
		Làm sao để tìm tất cả các mở rộng hợp lệ từ tất cả các đỉnh thuộc RMPath của Embedding 
		Chúng ta thực hiện tìm mở rộng lần lượt từ Qk đến Q0 (Chỉ xét các Q thuộc RMPath của P).
		B1. Tìm bậc lớn nhất của tất cả các vid của Q đang xét (GPU step). Kết quả lưu vào biến maxDegree_vid_Q
		B2. Tạo mảng d_arr_V có kích thước: maxDegree_vid_Q * |Q|
			Lưu ý, mảng d_arr_V phải có dạng cấu trúc đủ thể hiện cạnh mở rộng có hợp lệ hay không và là forward extension hay backward extension.
			struct struct_V
			{
				int valid; //default: 0, valid: 1
				int BK; //default: 0- forward; backward: 1
			}
		B3. Dựa vào CSDL để tìm những mở rộng hợp lệ, thông tin backward và forward được ghi nhận vào d_arr_V
		
		- Valid Forward: cạnh mở rộng luôn phải lớn hơn hoặc bằng cạnh đầu tiên của DFS_CODE
			+ Lớn hơn nếu nó có đỉnh from lớn hơn
			+ Hoặc nhãn cạnh lớn hơn
			+ hoặc nhãn đỉnh "to" lớn hơn
			+ và đỉnh "to" không thuộc embedding
		- Valid Backward: cạnh mở rộng luôn phải lớn hơn hoặc bằng cạnh kết nối với đỉnh "to" của mở rộng.
			+ Đỉnh "to" của mở rộng phải thuộc right most path
			+ Cạnh mở rộng không thuộc embedding và phải có nhãn cạnh lớn hơn hoặc bằng hoặc phải có một đỉnh lớn hơn hoặc bằng cạnh lớn nhất kề với đỉnh "to" của
			mở rộng.
		
		Nhưng vấn đề là làm sao kiểm tra mở rộng đó có thuộc embedding hay không?
		- Cần phải xây dựng một embedding và ánh xạ nó với CSDL hiện có để ghi nhận lại những cạnh và đỉnh đã thuộc embedding và right most path
		- Hoạt động này có thể được thực hiện một cách song song hay không?
		1. Chúng ta biết được số lượng embedding, suy ra chúng ta có thể biết được cần phải xây dựng bao nhiêu ánh xạ (history mapping).
			- Khởi tạo bộ nhớ d_arr_History, mỗi phần tử của d_arr_History là một đồ thị
		Mọi hoạt động cấp phát bộ nhớ đều phải được thực hiện ở host. Do đó, cần phải biết kích thước cần cấp phát bộ nhớ
			+ Duyệt qua embedding
				o Dựa vào vid để biết được embedding thuộc đồ thị nào

	*/

	//1. Lấy số lượng embedding từ device_arr_Q và lưu kết quả vào biến noEle_Embeddings
	//	 Lấy kích thước mảng dO, dLN và lưu vào mảng
	int *noEle_Embeddings=NULL;
	int *noEle_hEmbeddings=(int*)new int[1];
	cudaMalloc((void**)&noEle_Embeddings,sizeof(int));
	
	kernelGetnoEle_Embedding<<<1,1>>>(device_arr_Q,lastColumn,noEle_Embeddings);
	
	cudaMemcpy(noEle_hEmbeddings,noEle_Embeddings,sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize() kernelGetnoEle_Embedding failed");
		goto Error;
	} 

	printf("\nnoEle_Embeddings:%d",*noEle_hEmbeddings);
	
	//Tạo mảng số nguyên có kích thước bằng số lượng embedding
	//Mangr d_arr_number_HLN lưu trữ số lượng phần tử của mảng d_HLN trong object cHistory của embedding tương ứng.

	int * d_arr_number_HLN;
	cudaStatus=cudaMalloc((void**)&d_arr_number_HLN,sizeof(int)*noEle_hEmbeddings[0]);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,":\ncudaMalloc d_arr_number_HLN failed");
		goto Error;
	}
	else
	{
		cudaMemset(d_arr_number_HLN,0,sizeof(int)*noEle_hEmbeddings[0]);
	}

	findNumberOfEdgeInAGraph(d_arr_number_HLN,device_arr_Q,noEle_hEmbeddings[0],lastColumn,maxOfVer,d_O,numberOfGraph,noDeg);

	//Chép kết quả qua host
	int *h_arr_number_HLN;
	h_arr_number_HLN=(int*)malloc (sizeof(int)*noEle_hEmbeddings[0]);
	cudaMemcpy(h_arr_number_HLN,d_arr_number_HLN,(sizeof(int)*noEle_hEmbeddings[0]),cudaMemcpyDeviceToHost);

	for (int i = 0; i < noEle_hEmbeddings[0]; i++)
	{
		printf("\nh_arr_number_HLN[%d]:%d",i,h_arr_number_HLN[i]);
	}


	//2. Tạo một con trỏ đối tượng h_arrH, member của nó là h_arrH->vecA một mảng các pointer trỏ đến các đối tượng cHistory (cHistory **h_arrH->vecA)
	ArrHistory h_arrH(noEle_hEmbeddings[0]); //6 là số lượng embedding, giá trị tham số truyền vào sẽ được sử dụng để tạo ra 6 phần tử vecA có kiểu (cHistory**)
	//h_arrH.print();

	////Đứng đây thì làm sao khởi tạo cho các đối tượng bên trong (khởi tạo cho các con trỏ trỏ đến mảng các đối tượng) khi đã biết kích thước m và n của object
	for (int i = 0; i < noEle_hEmbeddings[0]; i++) 
	{
		//printf("\n************ %d ************",i);
		h_arrH.vecA[i]= (cHistory*) new cHistory(maxOfVer,h_arr_number_HLN[i]);
		//h_arrH.vecA[i]->print();
	}
	//h_arrH.print();

	//3. Bắt đầu quá trình sao chép mảng cHistory sang device
	int n=noEle_hEmbeddings[0]; 
	int numberElement_darrHO=maxOfVer;
	//int numberElement_darrHLN[]={2,4}; // chính là mảng h_arr_number_HLN chỉ số lượng phần tử của mảng d_arr_HLN trong đối tượng cHistory
	
	//cHistory **h = (cHistory**)malloc(sizeof(cHistory)*n); // chính là mảng h_arrH
	//for (int i = 0; i < n; i++)
	//{
	//	h[i] = new cHistory(numberElement_darrHO,numberElement_darrHLN[i]);
	//}

	cHistory *h1=new cHistory[n]; //Do các embedding lưu trong h_arrH.vecA[i] là không liên tục nhau trên bộ nhớ, Do đó tạo h1 với các bộ nhớ liên tục và chép dữ liệu của h_arrH.vecA[i] sang h1
	for (int i = 0; i < n; i++)
	{
		h1[i].n=h_arrH.vecA[i]->n;
		h1[i].m=h_arrH.vecA[i]->m;
		h1[i].d_arr_HO = (int*) malloc(sizeof(int)*numberElement_darrHO);
		for (int j = 0; j < numberElement_darrHO; j++)
		{
			h1[i].d_arr_HO[j]=h_arrH.vecA[i]->d_arr_HO[j];
		}
		h1[i].d_arr_HLN=(int*)malloc(sizeof(int)*h_arr_number_HLN[i]);
		for (int j = 0; j < h_arr_number_HLN[i]; j++)
		{
			h1[i].d_arr_HLN[j]=h_arrH.vecA[i]->d_arr_HLN[j];
		}
	}

	for (int i = 0; i < n; i++)
	{
		printf("\n********%d***********",i);
		h1[i].printmn();
	}


	cHistory **dH; //dH dùng để lưu kết quả cuối cùng của mảng cHistory
	cudaMalloc((void**)&dH,sizeof(cHistory*)*n);

	//Do không thể cấp phát bộ nhớ cho các member của dH một các trực tiếp trên device nên chúng ta sẽ cấp phát thông qua một biến khác device_H
	cHistory **device_H=(cHistory**)malloc(sizeof(cHistory*)*n);

	for (int j = 0; j < n; j++)
	{
		cHistory h2(numberElement_darrHO,h_arr_number_HLN[j]);
		h2.n=h1[j].n;
		h2.m=h1[j].m;
		for (int i = 0; i < h1[j].n; i++)
		{
			h2.d_arr_HO[i]=h1[j].d_arr_HO[i];
		}
		for (int i = 0; i < h1[j].m; i++)
		{
			h2.d_arr_HLN[i]=h1[j].d_arr_HLN[i];
		}
		//h2.print();	
		//Bây giờ làm sao chép đối tượng này sang bộ nhớ device?
		//Tạo một con trỏ đối tượng 
		
		cudaMalloc((void**)&device_H[j],sizeof(cHistory));
		cudaMemcpy(device_H[j],&h2,sizeof(cHistory),cudaMemcpyHostToDevice); //copy h bỏ vào d_h

		int *temp_dO,*temp_dHLN;	//khởi tạo bộ nhớ tạm trên device, gán dữ liệu cho bộ nhớ tạm này. Sau đó, gán chép bộ nhớ này cho các pointer bên trong.
		cudaMalloc((void**)&temp_dO,sizeof(int)*numberElement_darrHO);
		cudaMalloc((void**)&temp_dHLN,sizeof(int)*h_arr_number_HLN[j]);


		cudaMemcpy(temp_dO,h2.d_arr_HO,sizeof(int)*numberElement_darrHO,cudaMemcpyHostToDevice);
		cudaMemcpy(temp_dHLN,h2.d_arr_HLN,sizeof(int)*h_arr_number_HLN[j],cudaMemcpyHostToDevice);


		cudaMemcpy(&(device_H[j]->d_arr_HO),&(temp_dO),sizeof(int*),cudaMemcpyHostToDevice);
		cudaMemcpy(&(device_H[j]->d_arr_HLN),&(temp_dHLN),sizeof(int*),cudaMemcpyHostToDevice);

		/*printf("\naddress of d_h:%p",&device_H[j]);
		printf("\n**********j=%d***********",j);
		kernelPrintd_h<<<1,1>>>(device_H[j]);
		cudaDeviceSynchronize();
		if(cudaGetLastError()!=cudaSuccess){
			fprintf(stderr,"\ncudaDeviceSynchronize kernelPrintd_h has been failed");
			goto Error;
		}*/

	}

	cudaMemcpy(dH,device_H,sizeof(cHistory*)*n,cudaMemcpyHostToDevice);

	//kernelPrintdeviceH<<<1,1>>>(dH,n);

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize kernelPrintdeviceH d has been failed");
		goto Error;
	}

	//Cách chép 1 đối tượng cHistory sang device
		/* 
	//Rắc rối quá. Bây giờ mình chỉ làm cho 1 đối tượng cHistory
	cHistory h(6,12); //n=6 và m=12 đây là 2 giá trị tham số mà chúng ta cần phải trích ra từ embedding device_arr_Q. Sẽ được viết hàm sau.
	
	//h.print();	
	//Bây giờ làm sao chép đối tượng này sang bộ nhớ device?
	//Tạo một con trỏ đối tượng 
	cHistory *d_h;
	cudaMalloc((void**)&d_h,sizeof(cHistory));
	cudaMemcpy(d_h,&h,sizeof(cHistory),cudaMemcpyHostToDevice); //copy h bỏ vào d_h

	int *temp_dO,*temp_dHLN;	//khởi tạo bộ nhớ tạm trên device, gán dữ liệu cho bộ nhớ tạm này. Sau đó, gán chép bộ nhớ này cho các pointer bên trong.
	cudaMalloc((void**)&temp_dO,sizeof(int)*6);
	cudaMalloc((void**)&temp_dHLN,sizeof(int)*12);

	cudaMemcpy(temp_dO,h.d_arr_HO,sizeof(int)*6,cudaMemcpyHostToDevice);
	cudaMemcpy(temp_dHLN,h.d_arr_HLN,sizeof(int)*12,cudaMemcpyHostToDevice);

	// cudaMemcpy(&(d_c->data), &hostdata, sizeof(int *), cudaMemcpyHostToDevice);

	cudaMemcpy(&(d_h->d_arr_HO),&(temp_dO),sizeof(int*),cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_h->d_arr_HLN),&(temp_dHLN),sizeof(int*),cudaMemcpyHostToDevice);

	printf("\nsizeof(h):%d",sizeof(h));
	printf("\nsizeof(d_h):%d",sizeof(d_h));


	kernelPrintd_h<<<1,1>>>(d_h);
	*/

/*
//4. Đã có dH
	Bây giờ chúng ta duyệt qua các embedding và đánh dấu những đỉnh vào cạnh thuộc Embedding là 1
	Những đỉnh và cạnh nào thuộc Embedding thì đánh dấu là 2.
	Input: cHistory **dH, structQ *device_arr_Q, int lastColumn,vector<int> RMPath
*/	


cudaStatus = markEmbedding(dH,device_arr_Q,lastColumn,RMPath,n,maxOfVer,d_O,d_N);
	

	

	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize() has been failed");
		goto Error;
	}
Error:

	return cudaStatus;
}
