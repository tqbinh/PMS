#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <map>
#include "conio.h"
#include "kernelPrintf.h"
#include "gspan.h"
#include "kernelCountLabelInGraphDB.h"
#include "kernelMarkInvalidVertex.h"
#include "markInvalidVertex.h"
#include "checkArray.h"
#include "displayArray.h"
#include "checkDataBetweenHostAndGPU.h"
#include "access_d_LO_from_idx_of_d_O.h"
#include "countNumberOfLabelVetex.h"
#include "countNumberOfEdgeLabel.h"
#include "extractUniqueEdge.h"
#include "ExtensionStructure.h"
#include "getAndStoreExtension.h"
#include "validEdge.h"
#include "scanV.h"
#include "getLastElement.h"
#include "getValidExtension.h"
#include "getUniqueExtension.h"
#include "calcLabelAndStoreUniqueExtension.h"
#include "calcBoundary.h"
#include "calcSupport.h"
#include "getSatisfyEdge.h"
#include "header.h"

//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>	
using namespace std;

#define blocksize 512

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1); \
} \
}

//declare prototype
//void displayArray(int*,const unsigned int);
//bool checkArray(int*, int*, const int);
//__device__ void __syncthreads(void);

int main(int argc, char * const  argv[])
{	

#pragma region "Load data in to database. OutPut: d_O,d_LO,d_N,d_LN"

	//*************************** Load Graph database with some parameters ***********************

	//unsigned int minsup = 34;
	unsigned int minsup = 2;
	unsigned int maxpat = 2;
	//unsigned int maxpat = 0x00000000;
	unsigned int minnodes = 0;
	bool where = true;
	bool enc = false;
	bool directed = false;

	//int opt;
	char* fname;
	//fname = "Klesscus";
	fname = "Klessorigin";
	//fname = "KlessoriginCust1";
	//fname= "G0G1G2_custom";
	

	gSpan gspan;	
	ofstream fout("result.txt");

	//Chuyển dữ liệu từ fname sang TRANS
	gspan.run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);


	unsigned int maxOfVer;
	unsigned int numberOfGraph;
	maxOfVer=gspan.findMaxVertices();
	numberOfGraph=gspan.noGraphs();
	int sizeOfarrayO=maxOfVer*numberOfGraph;

	//printf("\nMaximun number of vertices: %d",maxOfVer);

	int* arrayO = new int[sizeOfarrayO]; //Tạo mảng arrayO có kích thước D*m
	if(arrayO==NULL){
		printf("\n!!!Memory Problem ArrayO");
		exit(1);
	}else{
		memset(arrayO, -1, sizeOfarrayO*sizeof(int)); // gán giá trị cho các phần tử mảng bằng -1
	}

	unsigned int noDeg; //Tổng bậc của tất cả các đỉnh trong csdl đồ thị TRANS
	noDeg = gspan.sumOfDeg();
	//cout<<noDeg;
	unsigned int sizeOfArrayN=noDeg;
	int* arrayN = new int[sizeOfArrayN]; //Mảng arrayN lưu trữ id của các đỉnh kề với đỉnh tương ứng trong mảng arrayO.
	if(arrayN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayN");
		exit(1);
	}else
	{
		memset(arrayN, -1, noDeg*sizeof(int));
	}


	int* arrayLO = new int[sizeOfarrayO]; //Mảng arrayLO lưu trữ label cho tất cả các đỉnh trong TRANS.
	if(arrayLO==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayLO");
		exit(1);
	}else
	{
		memset(arrayLO, -1, sizeOfarrayO*sizeof(int));
	}


	int* arrayLN = new int[noDeg]; //Mảng arrayLN lưu trữ label của tất cả các cạnh trong TRANS
	if(arrayLN==NULL){ //kiểm tra cấp phát bộ nhớ cho mảng có thành công hay không
		printf("\n!!!Memory Problem ArrayLN");
		exit(1);
	}else
	{
		memset(arrayLN, -1, noDeg*sizeof(int));
	}


	gspan.importDataToArray(arrayO,arrayLO,arrayN,arrayLN,sizeOfarrayO,noDeg,maxOfVer);

	cout<<"ArrayO:";
	displayArray(arrayO,sizeOfarrayO);
	cout<<"\nArrayLO:";
	displayArray(arrayLO,sizeOfarrayO);
	cout<<"\nArrayN:";
	displayArray(arrayN,noDeg);
	cout<<"\nArrayLN:";
	displayArray(arrayLN,noDeg);

	//kích thước của dữ liệu
	size_t nBytesO = sizeOfarrayO*sizeof(int);
	size_t nBytesLO = sizeOfarrayO*sizeof(int);
	size_t nBytesN = noDeg*sizeof(int);
	size_t nBytesLN = noDeg*sizeof(int);


	//****cấp phát vùng nhớ trên GPU***
	//1. khai báo biến trên GPU
	int *d_O;
	int *d_LO;
	int *d_N; //Số lượng phần tử của d_N bằng noDeg
	int *d_LN;

	//2. Kiểm tra lỗi khi cấp phát
	//Khai báo biến cudaStatusAllocate
	cudaError_t cudaStatusAllocate;

	cudaStatusAllocate =cudaMalloc((int**) &d_O,nBytesO);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		//goto labelError;
		exit(1);
	}

	cudaStatusAllocate =cudaMalloc((int**) &d_LO,nBytesLO);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		//goto labelError;
		exit(1);
	}
	cudaStatusAllocate =cudaMalloc((int**) &d_N,nBytesN);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		//goto labelError;
		exit(1);
	}
	cudaStatusAllocate =cudaMalloc((int**) &d_LN,nBytesLN);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		//goto labelError;
		exit(1);
	}


	//chép dữ liệu từ bốn mảng O,LO,N,LN từ Host sang GPU. Đây chính là CSDL đồ thị dùng để khai thác trên GPU
	cudaStatusAllocate = cudaMemcpy(d_O,arrayO,nBytesO,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto labelError;
		exit(1);
	}

	cudaStatusAllocate = cudaMemcpy(d_LO,arrayLO,nBytesLO,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto labelError;
		exit(1);
	}

	cudaStatusAllocate = cudaMemcpy(d_N,arrayN,nBytesN,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto labelError; 
		exit(1);
	}

	cudaStatusAllocate = cudaMemcpy(d_LN,arrayLN,nBytesLN,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		//goto labelError;
		exit(1);
	}


	//Đồng bộ đồng thời kiểm tra xem đồng bộ có lỗi không
	cudaStatusAllocate= cudaDeviceSynchronize();
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatusAllocate);
		//goto labelError;
		exit(1);
	}
	//xác định grid and block structure
	dim3 block(blocksize);
	dim3 grid((nBytesO + block.x -1)/block.x);
	printf("grid %d; block %d;\n",grid.x,block.x);


	//********kiểm tra đảm bảo dữ liệu ở GPU giống với Host********

	cudaError_t cudaStatus = checkDataBetweenHostAndGPU(d_O,d_LO,d_N,d_LN,sizeOfarrayO,noDeg,arrayO,arrayLO,arrayN,arrayLN,nBytesO,nBytesLO,nBytesN,nBytesLN);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkDataBetweenHostAndGPU failed!");
		return 1;
	}
	printf("\n***********Finished: Database has been copied from host to device. Next: count different label of vertex in all graph in database **********");
	printf("\n***********Press the Enter key to continous**********\n");
	getch();

#pragma endregion

	//don't use this snippet 

	/*
	//********Đếm số đỉnh song song và loại nhỏ những đỉnh nhỏ hơn minsup****
	//Nếu số đỉnh nhỏ hơn minSup thì đánh dấu đỉnh đó là -1 trong mảng O và mảng LO và các cạnh liên quan đến đỉnh đó cũng được đánh dấu là -1

	cudaStatus = markInvalidVertex(d_O,d_LO,sizeOfarrayO,minsup);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "markInvalidVertex failed!");
		return 1;
	}
	*/ 

#pragma region "Get Lv: Distinct vertex labels and Le: Distinct edge labels in graph database"
	//**********Đếm số nhãn đỉnh khác nhau trong CSDL đồ thị************
	//Nhãn đỉnh được lưu trữ trong mảng d_LO. Nhãn không hợp lệ mang giá trị -1
	//1.Cấp phát một mảng số nguyên có kích thước bằng với kích thước mảng d_LO gọi là d_Lv
	//2.Cấp phát |d_LO| threads
	//3.thread thứ i sẽ đọc giá trị nhãn tại vị trí d_LO[i], rồi ghi 1 vào mảng d_Lv[d_LO[i]]
	//4. Reduction mảng d_Lv để thu được các nhãn phân biệt
	unsigned int Lv=0;
	cudaStatus = countNumberOfLabelVetex(d_LO,sizeOfarrayO,Lv);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "countNumberOfLabelVetex failed!");
		return 1;
	}
	printf("\nNumber of different label Lv is: %d ;",Lv);
	printf("\n***********Finished: count different label of vertex in all graph in database. Next: count different label of edge in all graph in database **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

	//*******Đếm các loại cạnh khác nhau trong CSDL đồ thị************
	//Nhãn của tất cả các cạnh được lưu trữ trong mảng d_LN.
	/*
		1. Cấp phát một mảng số nguyên có kích thước băng với kích thước mảng d_LN gọi là d_Ln
		2. Cấp phát |d_Ln| threads
		3. Thread thứ i sẽ đọc giá trị nhãn tại vị trí d_Ln[i], rồi ghi vào mảng d_Ln[d_LN[i]]
		4. Reduction mảng d_Ln để thu được các loại cạnh phân biệt
	*/
	unsigned int Le=0;
	cudaStatus = countNumberOfEdgeLabel(d_LN,sizeOfArrayN,Le);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "countNumberOfLabelVetex failed!");
		return 1;
	}
	printf("\nNumber of different label Le is: %d ;",Le);
	printf("\n***********Finished: count different label of edge in all graph in database. Next: get and store all single edge extension into d_Extension **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

#pragma endregion

	//don't use this snippet

	/*******Thu thập tất cả các pattern P có 1 cạnh và tất cả các embeddings của P ********
	1. Duyệt qua cơ sở dữ liệu và rút trích các cạnh phân biệt dựa vào mảng d_O,d_LO,d_N và d_LN và 
	set giá trị 1 trong mảng d_SinglePattern có kích thước là (Lv.LE)
	tương ứng (lij.Lv+lj).
	2. Reduction d_singlePattern để biết số lượng pattern(numberOfPattern) là bao nhiêu. Sau đó cấp 
	phát numberOfPattern threads để tìm embeddings cho pattern đó.
	   Kết quả được lưu vào mảng cấu trúc d_Ext có số lượng phần tử bằng với d_N. 
	   Các thông tin cần lưu gồm:
		i. DFS Code của pattern theo cấu trúc (vi,vj,li,lij,lj)
	   ii. Lưu trữ các Embeddings của pattern(vig,vjg)
      iii. Row pointer trỏ đến heading của embedding ở cột cuối cùng Qp //
	3. Cấp phát mảng B có kích thước bằng với d_Ext để ghi nhận thông tin boundary của d_Ext
	4. Exclusive scan mảng B để ánh xạ lại graphid
	Tính support của từng pattern P
	5. Cấp phát mảng F có kích thước bằng với số lượng đồ thị trong CSDL
	6. Cấp phát numberOfPattern threads để cập nhật mảng F tương ứng với giá trị của kết quả scan trên B
	7. Reduction F để có độ support của từng pattern.
	*/


	//don't use this snippet

	//int *d_singlePattern=NULL;
	////size_t nBytesd_singlePattern = Lv*Le*sizeof(int);
	////Kích thước của mảng d_singlePattern là =[Lv!/(k!(Lv-k)!+n]*Le = [((Lv-2+1)*(Lv-2+2))/2 + Lv]*Le = [((Lv-1)*Lv)/2 + Lv]*Le
	////Trong trường hợp này k luôn = 2 vì cạnh có 2 đầu nhãn đỉnh lấy từ tập nhãn đỉnh phân biệt. 
	////Le là tập nhãn cạnh phân biệt trong CSDL
	////Vậy chúng ta có công thức cho kích thước của d_singlePattern như sau:
	//unsigned int numberOfElementd_singlePattern=(((Lv-1)*Lv)/2 +Lv)*Le;
	//size_t nBytesd_singlePattern = numberOfElementd_singlePattern*sizeof(int);
	//cudaStatus=cudaMalloc((int**)&d_singlePattern,nBytesd_singlePattern);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"cudaMalloc d_singlePattern failed", cudaStatus);
	//	return 1;
	//}
	//else
	//{
	//	cudaMemset(d_singlePattern,0,nBytesd_singlePattern);
	//}
	//
	//cudaStatus = extractUniqueEdge(d_O,d_LO,sizeOfarrayO,d_N,d_LN,sizeOfArrayN,d_singlePattern,numberOfElementd_singlePattern,Lv,Le);
	//if(cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"call extractUniqueEdge failed",cudaStatus);
	//	return 1;
	//}


	//*******Nhãn truy xuất nhãn LO theo index từ d_N; tên hàm đặt bị nhầm
	/*cudaStatus = access_d_LO_from_idx_of_d_O(d_LO,d_N,sizeOfArrayN);
	if(cudaStatus != cudaSuccess){
	fprintf(stderr, "access_d_LO_from_idx_of_d_N");
	return 1;
	}
	*/



#pragma region "Extract all edge in database and store them into d_Extension"
	//Giải thuật:
	/* //May/04/2017: Trích tất cả các cạnh từ CSDL và lưu vào d_Extension.
	Cách làm:
		1. Tạo một cấu trúc Extension để lưu trữ các mở rộng: DFSCode của cạnh mở rộng (vi,vj,li,lij,lj),global from vertex id(vgi),
		global to vertex id (vgj).
		2. Tạo một mảng có kích thước bằng với kích thước của d_N để lưu trữ các cạnh mở rộng ban đầu, lúc chưa có bất kỳ một cạnh
		phổ biến nào (P=0).
		3. Tạo một kernel có số lượng threads bằng d_O, mỗi thread sẽ xử lý một đỉnh. Nhiệm vụ của thread là đọc các cạnh kề với nó rồi
		lưu trữ thông tin vào mảng Extension tương ứng tại vị trí.
		
	*/
	

	//cấp phát bộ nhớ cho d_Extension
	int numberOfElementd_N=noDeg;
	size_t nBytesOfArrayExtension = numberOfElementd_N*sizeof(Extension);
	Extension *d_Extension;
	cudaStatus= cudaMalloc((Extension**)&d_Extension,nBytesOfArrayExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"CudaMalloc d_Extension fail",cudaStatus);
		exit(1);
	}

	//Trích tất cả các cạnh từ database rồi lưu vào d_Extension
	int numberOfElementd_O=sizeOfarrayO;	
	cudaStatus = getAndStoreExtension(d_Extension,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Le,Lv);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize getAndStoreExtension failed",cudaStatus);
		return 1;
	}

	CHECK(printfExtension(d_Extension,numberOfElementd_N));

	
	printf("\n***********Finished: get and store all single edge extension in to d_Extension. Next: set 1 for all valid single edge extension in V array **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

#pragma endregion

#pragma region "Mark valid edge from d_Extension and set 1 in array V at the corresponding position"
	//Giải thuật:
	/* //05-May-2017: Khởi tạo mảng V với giá trị của các phần tử ban đầu là 0, để lưu trữ những mở rộng hợp lệ.
	1. Mở rộng hợp lệ là mở rộng có Lj<=Li
	2. Mảng V có số lượng phần tử bằng với số lượng phần tử của mảng d_Extension
	3. Tạo kernel với số lượng threads bằng với số lượng phần tử của d_Extension
		Mỗi thread sẽ xử lý một phần tử trong d_Extension. Kiểm tra nếu Lj<=Li thì gán V tại vị trí tương ứng là 1
	*/

	int numberElementd_Extension = numberOfElementd_N;
	int *V;
	size_t nBytesV= numberElementd_Extension*sizeof(int);

	cudaStatus=cudaMalloc((int**)&V,nBytesV);
	if (cudaStatus!= cudaSuccess){
		fprintf(stderr,"cudaMalloc array V failed",cudaStatus);
		exit(1);
	}
	else
	{
		cudaMemset(V,0,nBytesV);
	}

	cudaStatus=validEdge(d_Extension,V,numberElementd_Extension);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize validEdge failed",cudaStatus);
		return 1;
	}

	
	printf("\n***********Finished: set 1 for all valid single edge extension in V array. Next: Extract all valid extension  from d_Extension base on V array **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

#pragma endregion
	
#pragma region "Exclusive scan V array and then store scan result into index array"

	//Giải thuật:
	/* //07-May-2017: Extract unique Edge from d_Extension
	1. Tiếp theo, chúng ta exclusive scan mảng V để thu được index chỉ vị trí của các valid edge trong d_Extension.
	2. Input data: mảng V
	3. Output data: mảng index
	Mảng Index có số lượng phần tử bằng với mảng V
	*/

	//cấp phát bộ nhớ cho mảng index
	int* index;
	cudaStatus=cudaMalloc((int**)&index,numberElementd_Extension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"Cuda Malloc failed",cudaStatus);
		return 1;
	}	
	//Exclusive scan mảng V và lưu kết quả scan vào mảng index
	cudaStatus = scanV(V,numberElementd_Extension,index);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize getAndStoreExtension failed",cudaStatus);
		return 1;
	}
	//Hiển thị nội dung mảng index
	printf("\n Scan Result index: ");
	kernelPrintf<<<grid,block>>>(index,numberElementd_Extension);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelPrintf failed",cudaStatus);
		exit(1);
	}

#pragma endregion

#pragma region "Replying on the index array to extract the valid edges from d_Extension to d_ValidExtension"

/*	//Khởi tạo một mảng d_Unique có kích thước bằng với kích thước của giá trị của phần tử index cuối cùng vừa mới scan được.
	1. Hàm getLastElement sẽ trả về giá trị của phần tử cuối của mảng index
	2. Viết hàm để trích và lưu trữ các mở rộng hợp lệ
		a. khởi tạo mảng có kích thước bằng với kích thước của phần tử cuối của mảng index
		b. Rút trích các mở rộng hợp lệ từ d_Extension tương ứng tại vị trí V=1 vào index tương ứng.
*/
	//1. Hàm getLastElement
	int noElem_d_ValidExtension=0;
	cudaStatus=getLastElement(index,numberElementd_Extension,noElem_d_ValidExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getLastElement failed",cudaStatus);
		return 1;
	}
	noElem_d_ValidExtension++;
	//printf("\n\nnumberElementd_UniqueExtension:%d",noElem_d_ValidExtension);

	/* //08-May-2017: getValidExtension */
	//2.Hàm extractValidExtension: Trích và lưu trữ các mở rộng hợp lệ
	//2.1. Cấp phát bộ nhớ cho d_ValidExtension
	Extension *d_ValidExtension;
	cudaStatus=cudaMalloc((Extension**)&d_ValidExtension,noElem_d_ValidExtension*sizeof(Extension));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudamalloc failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_ValidExtension,0,noElem_d_ValidExtension*sizeof(Extension));
	}

	
	cudaDeviceSynchronize();
	
	//Trích những cạnh hợp lệ từ mảng d_Extension sang d_ValidExtension
	cudaStatus=getValidExtension(d_Extension,V,index,numberElementd_Extension,d_ValidExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getValidExtension failed",cudaStatus);
		return 1;
	}

	printf("\nNumber Element of d_ValidExtension:%d",noElem_d_ValidExtension);
	CHECK(printfExtension(d_ValidExtension,noElem_d_ValidExtension));


	printf("\n***********Finished: Extract all valid extension  from d_Extension base on V array and put result in d_ValidExtension. Next: Extract unique extension from d_ValidExtension base on label of vertex and label of edge **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

#pragma endregion

#pragma region "Extract the unique edges from d_ValidExtension replying on their label. Note: d_allPossibleExtension"

	//Giải thuật
	/* //Hàm getUniqueExtension: Trích tra các cạnh duy nhất dựa vào nhãn Li, Lj và Lij của edge extension
	1. Tạo mảng d_allPossibleExtension có kích thước là noElem_allPossibleExtension=Le*Lv*Lv để lưu trữ 
		tất cả các mở rộng có thể có của tất cả các đỉnh. Các mở rộng có thể có từ 1 đỉnh trên righ most path có kích thước là Le*Lv.
	2. Viết hàm getUniqueExtension để gán giá trị là 1 tại vị trí Li Lj tương ứng của extension.
	*/
	//Tính số lượng tất cả các cạnh có thể có dựa vào nhãn của chúng
	unsigned int noElem_allPossibleExtension=Le*Lv*Lv;
	int *d_allPossibleExtension;

	//cấp phát bộ nhớ cho mảng d_allPossibleExtension
	cudaStatus=cudaMalloc((int**)&d_allPossibleExtension,noElem_allPossibleExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtension failed",cudaStatus);
		return 1;
	}

	/* //09-May-2017 */
	//Trích các cạnh duy nhất dựa vào nhãn cạnh
	cudaStatus=getUniqueExtension(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"getUniqueExtension failed",cudaStatus);
		return 1;
	}
	printf("\n\nLe:%d Lv:%d",Le,Lv);
	printf("\nd_allPossibleExtension: ");
	printInt(d_allPossibleExtension,noElem_allPossibleExtension);
	
	printf("\n***********Finished: Extract unique extension from d_ValidExtension base on label of vertex and label of edge and set 1 as result in d_allPossibleExtension array. Next: Mapping label of vertex and edge into edge and store them in d_UniqueExtension **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();

	/* //Tiếp theo chúng ta exclusive scan mảng d_allPossibleExtension để thu được index phục vụ cho việc
		lưu trữ các unique extension.
	1. Chúng ta khởi tạo một mảng d_allPossibleExtensionScanResult có kích thước bằng với kích thước của d_allPossibleExtension
		đồng thời khởi tạo giá trị cho các phần tử của nó là 0.
	2. Sau khi thu được kết quả scan, chúng ta tạo một mảng Extension* d_UniqueExtension có kích thước bằng với kích thước của giá trị phần tử cuối cùng
		trong mảng d_allPossibleExtensionScanResult cộng với 1.
	3. Dựa vào giá trị index trong mảng d_allPossibleExtensionScanResult để suy ra các nhãn Li, Lj và Lij của Extension và lưu trữ chúng vào d_UniqueExtension
	*/

	//Cấp phát bộ nhớ cho mảng d_allPossibleExtensionScanResult có kích thước bằng với mảng d_allPossibleExtension
	int *d_allPossibleExtensionScanResult;
	cudaStatus=cudaMalloc((int**)&d_allPossibleExtensionScanResult,noElem_allPossibleExtension*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtensionScanResult failed");
		return 1;
	}
	else
	{
		cudaMemset(d_allPossibleExtensionScanResult,0,noElem_allPossibleExtension*sizeof(int));
	}

	//Exclusive scan mảng d_allPossibleExtension và lưu kết quả vào mảng d_allPossibleExtensionScanResult
	cudaStatus=scanV(d_allPossibleExtension,noElem_allPossibleExtension,d_allPossibleExtensionScanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanV function failed",cudaStatus);
		return 1;
	}

	printf("\n\nd_allPossibleExtensionScanResult:\n");
	CHECK(printInt(d_allPossibleExtensionScanResult,noElem_allPossibleExtension));

	/*	Tính kích thước cho mảng d_UniqueExtension
	*	Lấy giá trị của phần tử cuối cùng trong mảng d_allPossibleExtensionScanResult và lưu vào biến noElem_d_UniqueExtension
	*	Nếu phần tử cuối cùng của mảng d_allPossibleExtension có giá trị 1 thì phải tăng biến noElem_d_UniqueExtension lên 1
	*/
	int noElem_d_UniqueExtension=0;
	//Tính kích thước của mảng d_UniqueExtension dựa vào kết quả exclusive scan
	cudaStatus=getSizeBaseOnScanResult(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,noElem_d_UniqueExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getLastElement failed",cudaStatus);
		return 1;
	}
	// printf("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtension);

	//Tạo mảng d_UniqueExtension với kích thước mảng vừa tính được
	UniEdge *d_UniqueExtension;
	cudaStatus=cudaMalloc((void**)&d_UniqueExtension,noElem_d_UniqueExtension*sizeof(UniEdge));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_UniqueExtension failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_UniqueExtension,0,noElem_d_UniqueExtension*sizeof(UniEdge));
	}

	//Ánh xạ ngược lại từ vị trí trong d_allPossibleExtension thành cạnh và lưu kết quả vào d_UniqueExtension
	cudaStatus=calcLabelAndStoreUniqueExtension(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,noElem_d_UniqueExtension,Le,Lv);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n\ncalcLabelAndStoreUniqueExtension function failed",cudaStatus);
		return 1;
	}

	printf("\n\nUnique Extension:");
	printfUniEdge(d_UniqueExtension,noElem_d_UniqueExtension);

	printf("\n***********Finished: Mapping label of vertex and edge into edge and store them in d_UniqueExtension . Next: compute support for valid unique extension in d_UniqueExtension **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();


#pragma endregion


#pragma region "Caculating support for each edge in d_UniqueExtension"
	
	//Giải thuật:
	/* //10-May-2017: Tính độ hỗ trợ
	1. Trước tiên, chúng ta cấp phát một mảng d_B, mảng này có số lượng phần tử bằng với số lượng phần tử của d_ValidExtension
		Mảng d_B dùng để đánh dấu vị trí biên (boundary: nơi tiếp giáp giữa 2 đồ thị)
	2. Exclusive scan mảng d_B và lưu kế quả vào d_scanB_result
	3. Khởi tạo mảng d_F có số lượng phần tử bằng với giá trị của phần tử cuối cùng của mảng d_scanB_Result cộng 1
	4. Tính độ hỗ trợ của từng phần tử trong mảng d_UniqueExtension dựa vào d_ValidExtension và ScanB_Result
	*/
	
	/* Xây dựng Boundary cho mảng d_ValidExtension */
	//1. Cấp phát một mảng d_B và gán các giá trị 0 cho mọi phần tử của d_B
	unsigned int noElement_d_B=noElem_d_ValidExtension;
	int* d_B;
	cudaStatus=cudaMalloc((int**)&d_B,noElement_d_B*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_B failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_B,0,noElement_d_B*sizeof(int));
	}

	//Gián giá trị boundary cho d_B
	cudaStatus=calcBoundary(d_ValidExtension,noElem_d_ValidExtension,d_B,maxOfVer);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"calcBoundary function failed",cudaStatus);
		return 1;
	}

	printf("\n\nd_B:\n");
	printInt(d_B,noElement_d_B);
	printf("\n***********Finished: set Boundary for d_ValidExtension . Next: compute support for valid unique extension in d_UniqueExtension **********");
	printf("\n***********Press the Enter key to continous**********\n");
	//getch();


	//2. Exclusive Scan mảng d_B
	int* d_scanB_Result;
	cudaStatus=cudaMalloc((int**)&d_scanB_Result,noElement_d_B*sizeof(int));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_scanB_Result failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_scanB_Result,0,noElement_d_B*sizeof(int));
	}

	cudaStatus=scanV(d_B,noElement_d_B,d_scanB_Result);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanB function failed",cudaStatus);
		return 1;
	}

	printf("\n\nd_scanB_Result:\n");
	printInt(d_scanB_Result,noElement_d_B);

	//3. Tính độ hỗ trợ cho các mở rộng trong d_UniqueExtension
	//3.1 Tạo mảng d_F có số lượng phần tử bằng với giá trị cuối cùng của mảng d_scanB_Result cộng 1 và gán giá trị 0 cho các phần tử.
	int noElemF=0;
	cudaStatus=getLastElement(d_scanB_Result,noElement_d_B,noElemF);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ngetLastElement function failed",cudaStatus);
		return 1;
	}

	noElemF++;
	printf("\nnoElement_F:%d",noElemF);

	float *d_F;
	cudaStatus=cudaMalloc((int**)&d_F,noElem_d_UniqueExtension*noElemF*sizeof(float));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_F failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_F,0,noElemF*sizeof(float));
	}
		
	
	/* //Gọi hàm calcSupport để tính độ hỗ trợ cho các mở rộng trong mảng d_UniqueExtension đồng thời gọi hàm buildEmbedding để xây dựng embedding cho mở rộng thoả minsup*/
	//Mở rộng nào phổ biến sẽ được ghi nhận lại vào mảng h_frequentEdge (Có số lượng phần tử bằng với d_uniqueExtension)
	//Tại vị trí tương ứng của cạnh lớn hơn bằng minsup sẽ được set là 1.
	/*int numberEle_h_frequentEdge=noElem_d_UniqueExtension; //không dùng h_frequentEdge để hi nhận những mở rộng thoả minSup
	int *h_frequentEdge = (int*) malloc(numberEle_h_frequentEdge*sizeof(int));
	if(h_frequentEdge==NULL){
		printf("\n Malloc array h_frequentEdge failed");
		exit(1);
	}
	else
	{
		memset(h_frequentEdge,0,numberEle_h_frequentEdge*sizeof(int));
	}*/
	//vector<int> h_satisfyEdge;
	//vector<int> h_satisfyEdgeSupport;
	
	//Hàm calcSupport tính độ hỗ trợ của tất cả các cạnh trong d_UniqueExtension
	//Nó  trả về vị trí index của d_UniqueExtension mà tại đó thoả minSup
	//Nó  trả về giá trị minSup
	//cudaStatus=calcSupport(d_UniqueExtension,noElem_d_UniqueExtension,d_ValidExtension,noElem_d_ValidExtension,d_scanB_Result,d_F,noElement_F,minsup,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,maxOfVer,numberOfGraph,noDeg,h_satisfyEdge,h_satisfyEdgeSupport);
	//if (cudaStatus!=cudaSuccess){
	//	fprintf(stderr,"\ncalcSupport function failed",cudaStatus);
	//	return 1;
	//}

	float *h_resultSup=nullptr;
	cudaStatus=computeSupport(d_UniqueExtension,noElem_d_UniqueExtension,d_ValidExtension,noElem_d_ValidExtension,d_scanB_Result,d_F,noElemF,h_resultSup);
	
	//In độ hỗ trợ cho các cạnh tương ứng trong mảng kết quả h_resultSup
	for (int i = 0; i < noElem_d_UniqueExtension; i++)
	{
		printf("\n resultSup[%d]:%.1f",i,h_resultSup[i]);
	}
	
#pragma endregion

#pragma region "count number of edge in each graph in database"

	numberOfGraph;//Có bao nhiêu đồ thị

	//Tạo ra một mảng để lưu trữ số lượng cạnh của các đồ thị trong CSDL
	int noElem_hNumberEdgeInEachGraph=numberOfGraph;
	int *hNumberEdgeInEachGraph; /* mảng này ở bộ nhớ host */
	int *dNumberEdgeInEachGraph=nullptr; /* mảng này ở bộ nhớ trên device */
	cudaStatus = getNumberOfEdgeInGraph(d_O,numberOfElementd_N,maxOfVer,hNumberEdgeInEachGraph,dNumberEdgeInEachGraph,numberOfGraph);
		if(cudaStatus != cudaSuccess){
			fprintf(stderr,"\n getNumberOfEdgeInGraph() in kernel.cu failed",cudaStatus);
			goto Error;
		}

		printf("\n ************ hNumberEdgeInEachGraph **************\n");
		for (int i = 0; i < numberOfGraph; i++)
		{
			printf("\n hNumberEdgeInEachGraph[%d]:%d",i,hNumberEdgeInEachGraph[i]);
		}
#pragma endregion


#pragma region "Build DFS_CODE for the valid edge in d_UniqueExtension && CHECK minDFS_CODE && Create Embedding Column && find Extension from all Embedding"

	/*	Build DFS_Code for the valid Extension********************************************************************
	*	Duyệt qua các cạnh trong mảng d_UniqueExtension và đối chiếu với độ hỗ trợ của cạnh trong mảng h_resultSup
	*	Nếu độ hỗ trợ >= minsup thì sẽ ghi tạo DFS_code cho nó.
	*/
	//Cấp phát bộ nhớ tạm để lấy nhãn cạnh từ device
		UniEdge *h_tempEdge=nullptr;
		h_tempEdge=(UniEdge*)malloc(sizeof(UniEdge));
		if(h_tempEdge==NULL){
			printf("\n malloc h_tempEdge in kernel.cu failed");
			exit(1);
		}
		
	//Duyệt  và kiểm tra xem độ hỗ trợ  của các phần tử trong mảng Unique *d_UniqueExtension có thoả minSup hay không?
	// mảng h_resultSup lưu giá trị support của cạnh tương ứng trong d_UniqueExtension
	for (int i = 0; i < noElem_d_UniqueExtension; i++)
	{
		#pragma region "check minsup statification"

		if(h_resultSup[i]>=minsup){ /*Nếu phần tử nào có độ hỗ trợ lớn hơn minsup thì mới chép nhãn của cạnh đó sang host để kiểm tra minDFS_Code */
			cudaMemcpy(h_tempEdge,&d_UniqueExtension[i],sizeof(UniEdge),cudaMemcpyDeviceToHost); /* chép cạnh trong d_UniqueExtension sang mảng h_tempEdge */
			
			/* Lấy nhãn của cạnh li,lij và lj để xây dựng DFS_CODE */			
			int li,lij,lj;
			li =h_tempEdge[0].li;
			lij = h_tempEdge[0].lij;
			lj=h_tempEdge[0].lj;
			//printf("\n (%d,%d,%d)",h_tempEdge[0].li,h_tempEdge[0].lij,h_tempEdge[0].lj); 
			

			gspan.DFS_CODE.push(0,1,li,lij,lj); /* Xây dựng DFS_CODE ban đầu cho cạnh */

			int minLabel=li; /* lấy minLabel để phục vụ cho quá trình mở rộng cạnh */
			int maxid = 1; /* id lớn nhất của DFS_CODE */
			
			/*	 Kiểm tra xem DFS_CODE có phải là nhỏ nhất hay không. Nếu DFS_CODE là nhỏ nhất thì mới ghi kết quả DFS_CODE vào file result.txt
			 *	 Nếu thoả minDFS_CODE thì quá trình khai thác sẽ được lặp đi lặp lại cho đến khi nào không thể khai thác trên nhánh đó được nữa.
			*/

#pragma region "check graphismin"

			if(gspan.is_min()){
				int *hArrGraphId;
				int noElem_hArrGraphId=0;
				hArrGraphId=(int*)malloc(sizeof(int)*noElemF);	

				/* Trước khi ghi kết quả thì phải biết đồ thị phổ biến đó tồn tại ở những graphId nào. Hàm getGraphIdContainEmbedding dùng để làm việc này
				* 3 tham số đầu tiên của hàm là nhãn cạnh của phần tử d_UniqueExtension đang xét */
				cudaStatus =getGraphIdContainEmbedding(li,lij,lj,d_ValidExtension,noElem_d_ValidExtension,hArrGraphId,noElem_hArrGraphId,maxOfVer);
				if (cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n getGraphIdContainEmbedding in kernel.cu failed",cudaStatus);
					exit(1);
				}
				
				//In nội dung mảng hArrGraphId
				/* 
				printf("\n ************** hArrGraphId ****************\n");
				for (int j = 0; j < noElem_hArrGraphId; j++)
				{
					printf("%d ",hArrGraphId[j]);
				}*/

				/*	Ghi kết quả DFS_CODE vào file result.txt ************************************************************
				 *	Hàm report sẽ chuyển DFS_CODE pattern sang dạng đồ thị, sau đó sẽ ghi đồ thị đó xuống file result.txt
				 *	Hàm report gồm 3 tham số:
				 *	Tham số thứ 1: mảng chứa danh sách các graphID chứa DFS_CODE pattern
				 *	Tham số thứ 2: số lượng mảng
				 *	Tham số thứ 3: độ hỗ trợ của DFS_CODE pattern *******************************************************/
				gspan.report(hArrGraphId,noElem_hArrGraphId,h_resultSup[i]);

				//Giải phóng bộ nhớ hArrGraphId
				free(hArrGraphId);
				

				/* Tạo Embedding cho DFS_CODE **************************************
				 * Mỗi một cột Q được mô tả bởi 3 mảng: dArrPointerEmbedding,dArrSizedQ,dArrPrevQ
				 * Dựa vào d_ValidExtension để xây dựng Embedding cho các DFS_CODE */
				Embedding **dArrPointerEmbedding;
				int noElem_dArrPointerEmbedding=0; //Số lượng cột Q
				int *dArrSizedQ=nullptr; //Số lượng phần tử của từng cột Q
				int noElem_dArrSizedQ=0;
				
				//Không dùng mảng dArrPrevQ nữa
				/*int *dArrPrevQ; //Liên kết với PrevQ nào
				int noElem_dArrPrevQ=0;
				cudaStatus = createEmbeddingRoot(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrSizedQ,noElem_dArrSizedQ,dArrPrevQ,noElem_dArrPrevQ,d_ValidExtension,noElem_d_ValidExtension,li,lij,lj);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n createEmbeddingRoot in kernel.cu failed");
					goto Error;
				}
				*/

				//Tạo embedding root không có prevQ
				cudaStatus = createEmbeddingRoot1(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrSizedQ,noElem_dArrSizedQ,d_ValidExtension,noElem_d_ValidExtension,li,lij,lj);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n createEmbeddingRoot in kernel.cu failed");
					goto Error;
				}

				//In nội dung của Embedding column.
				cudaStatus = printAllEmbeddingColumn(dArrPointerEmbedding,dArrSizedQ,noElem_dArrPointerEmbedding);	
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n printAllEmbeddingColumn() in kernel.cu failed");
					goto Error;
				}
				

				/*Cần ít nhất 2 tham số để truy xuất toàn bộ các đỉnh trên embedding
				*	- Vị trí trong mảng dArrPointerEmbedding (posColumn)
				*	- vị trí trong mảng dArrSizedQ (posRow)
				*/
				/*
				int posColumn =1;
				int posRow = 2;
				cudaStatus = printEmbeddingFromPos(dArrPointerEmbedding,posColumn,posRow);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n printEmbeddingFromPos() in kernel.cu failed");
					goto Error;
				}
				*/
				/* Tạo dRMPath cho Embedding Column. Nó lưu trữ index của dArrPointerEmbedding mà tại đó Q column thuộc right most path 
				 * Làm sao để cập nhật dRMPath khi mở rộng embedding? 
				 * Nếu có embedding và thông tin cột cuối cùng của embedding column 
				 * thì chúng ta có thể lần ngược về các cột phía trước của embedding
				 * và đồng thời cập nhật lại dRMPath
				 * Cần viết một hàm để tính kích thước của dRMPath
				 * Sau đó viết một hàm nữa để cập nhật giá trị cho dRMPath */

				int *dRMPath=nullptr;
				int noElem_dRMPath=0;		
				cudaStatus = createRMPath(dRMPath,noElem_dRMPath);
				if (cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n createRMPath() in kernel.cu failed",cudaStatus);
					goto Error;
				}

				//cudaStatus = printRMPath(dRMPath,noElem_dRMPath);
				//if (cudaStatus!=cudaSuccess){
				//	fprintf(stderr,"\n printRMPath() in kernel.cu failed",cudaStatus);
				//	goto Error;
				//}

				/* Tiếp theo là tìm các mở rộng hợp lệ từ các embedding 
			* Các mở rộng hợp lệ có thể là mở rộng forward hoặc là mở rộng backward

				Backward edge extension:
				Trong đó: 
				- ĐK1: id của đỉnh "to" của cạnh cuối cùng bằng với id của đỉnh from thuộc cạnh right most path đang xét (Hay nói cách khác, đỉnh "to" của cạnh mở rộng phải thuộc right most path).
				- ĐK2: Nhãn cạnh của mở rộng lớn hơn nhãn cạnh của đỉnh from của cạnh thuộc right most path
				- ĐK3: Gồm 3.1 và 3.2
				o ĐK3.1: Nhãn cạnh của mở rộng bằng với nhãn cạnh của right most path
				o ĐK3.2: Nhãn đỉnh to của cạnh cuối cùng (đỉnh from của cạnh mở rộng) của DFS_CODE lớn hơn hoặc bằng nhãn đỉnh to của right most path.

				Forward edge extension:
				1.2.1 Tìm tất cả các mở rộng forward edge từ đỉnh phải cùng của DFS_CODE (get_forward_pure function)
				- Chỉ lấy những mở rộng mà có nhãn đỉnh "to" lớn hơn hoặc bằng nhãn đỉnh minlabel (minlabel là nhãn đỉnh from của cạnh đầu tiên trong DFS_CODE) và đỉnh to của cạnh mở 	rộng chưa thuộc DFS_CODE.
				1.2.2 Tìm tất cả các mở rộng forward edge từ các đỉnh còn lại thuộc right most path (get_forward_rmpath function)
				- Loại bỏ mở rộng khi:
				o id đỉnh to của mở rộng bằng với id đỉnh to của cạnh right most path, vì lúc này cạnh mở rộng trùng với right most path.
				o hoặc nhãn đỉnh to của mở rộng nhỏ hơn nhãn đỉnh minlabel
				o hoặc id của đỉnh to của mở rộng đã thuộc DFS_CODE rồi.
				- Chỉ lấy các mở rộng khi:
				o Nhãn cạnh của right most path nhỏ hơn nhãn cạnh mở rộng
				o hoặc nhãn cạnh của right most path bằng với nhãn cạnh mở rộng và nhãn đỉnh to của righ most path nhỏ hơn hoặc bằng nhãn đỉnh to của mở rộng.
			* Với điều kiện như trên thì dữ liệu đầu vào cần phải có những gì:
			Input:
				- Embedding column: Gồm 3 mảng và các thông tin mô tả cho chúng
				- Right Most Path: là một mảng lưu trữ những Q column thuộc right most path
				- Database: gồm mảng d_O,d_LO,d_N,d_LN và thông tin kích thước của từng mảng
			Output:
				- Extension: mảng d_ForwardExtension và thông tin mô tả kích thước.
			* Mở rộng phải được thực hiện theo trình tự: Backward Extension --> forward Extension (từ đỉnh cuối lần ngược lên đỉnh root của embedding).
			* Mở rộng backward chỉ tồn tại ở đỉnh cuối và right most path có nhiều hơn 1 cạnh
			* Mở rộng forward từ đỉnh cuối:
				 
				 Nhưng từ đỉnh cuối thì làm sao mà lần ra được tất cả các đỉnh thuộc embedding? Nếu từ cột cuối thì chỉ có thể lần ra right most path của embedding mà thôi.
				 Cho nên chúng ta không thể kiểm tra được đỉnh hoặc cạnh đó có thuộc embedding hay không.
				 ==> Chúng ta cần xây dựng GraphHistory, nó cho biết những đỉnh nào và cạnh nào đã thuộc embedding và những đỉnh nào và cạnh nào thuộc right most path.
			Làm sao để xây dựng graphHistory?
			 - Có bao nhiêu Embedding thì xây dựng bấy nhiêu graphHistory. Phải dựa vào kích thước của cột Q cuối cùng để biết số lượng Embedding.
			 - Các graphHistory có số lượng đỉnh bằng nhau (maxOfVer).
			 - Các graphHistory có số lượng cạnh khác nhau
			 - Duyệt qua embedding và gán các giá trị tương ứng cho graphHistory tương ứng là 2(ý muốn nói những đỉnh và cạnh đó thuộc embedding
			 - Mỗi khi có embedding mới, thì ta cần phải cập nhật lại graphHistory tương ứng, chuyển những giá trị 2 thành giá trị 1. Sau đó, duyệt qua embedding
			   mới và gán lại giá trị là 2 tại những vị trí thuộc right most path.
				 
			*/

///*************Tạm gác lại vấn đề khai thác dựa trên graphHistory *******************/
///****************Sẽ quay lại sau, để so sánh hiệu năng *****************************/
			#pragma region "graphHistory is temporary stopping here"
								/*
								////Xây dựng graphHistory
								///*  Mỗi Embedding có một graphHistory gắn liền với nó, mô tả những cạnh và đỉnh đã thuộc embedding
								//*	Khi mở rộng embedding thì phải cập nhật lại graphHistory nếu mở rộng đó là phổ biến
								//*	Ngược lại, nếu không có mở rộng phổ biến nào được phát triển từ embedding đó thì graphHistory của nó cũng phải được giải phóng bộ nhớ.
								//*	==> graphHistory phải được xây dựng bên trong quá trình mở rộng embedding và được cập nhật cho đến khi không còn mở rộng phổ biến nào tồn tại.
								//Nhưng ở đây các mở rộng từ các embedding phải được thực hiện một các song song. Tức là chúng ta phải có một mảng các graphHistory 
								// */


				
								/*
								//graphHistory là sự kết hợp gồm 3 mảng bên dưới. Số lượng phần tử của các mảng này bằng nhau và bằng số lượng embedding.
								int noElem_dArrPointerdHO = 0;
								int **dArrPointerdHO=nullptr; //Lưu trữ pointer của mảng các đỉnh trên device
								int **dArrPointerdHLN=nullptr; //Lưu trữ pointer của mảng các cạnh trên device
								int *dArrNumberEdgeOfEachdHLN=nullptr; //số lượng phần tử của mảng các cạnh trên device
								*/
								/*Tạo graphHistory
								*	Input: Các cột embedding (bộ 6 thành phần), CSDL (d_O,d_N,d_LO,d_LN) và số lượng phần tử của chúng
								*	Output: Bộ 3 thành phần graphHistory
								*/
								/*
								cudaStatus = createGraphHistory(dArrPointerEmbedding,dArrSizedQ,dArrPrevQ,noElem_dArrPointerEmbedding,noElem_dArrSizedQ,noElem_dArrPrevQ,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,maxOfVer,dArrPointerdHO,noElem_dArrPointerdHO,dArrPointerdHLN,dArrNumberEdgeOfEachdHLN,hNumberEdgeInEachGraph,noElem_hNumberEdgeInEachGraph,dNumberEdgeInEachGraph);
								if(cudaStatus!=cudaSuccess){
									fprintf(stderr,"\n createGraphHistory() in kernel.cu failed",cudaStatus);
									goto Error;
								}
								*/
								//In nội dung của graphHistory
								//printf("\n noElem_dArrPointerdHO:%d",noElem_dArrPointerdHO);
 								//printf("\n ********** dArrPointerdHO *****************\n");
								//printDoublePointerInt(dArrPointerdHO,noElem_dArrPointerdHO,maxOfVer);
								//if(cudaStatus!=cudaSuccess){
								//	fprintf(stderr,"\n printDoublePointerInt() in kernel.cu failed",cudaStatus);
								//	goto Error;
								//}
								//printf("\n**************dArrNumberEdgeOfEachdHLN**************\n");
								//printInt(dArrNumberEdgeOfEachdHLN,noElem_dArrPointerdHO);
								//printf("\n**************dArrPointerdHLN**************\n");
								//printDoublePointerInt(dArrPointerdHLN,noElem_dArrPointerdHO,dArrNumberEdgeOfEachdHLN);
				#pragma endregion

//Tìm các mở rộng hợp lệ từ các đỉnh của embedding. 
#pragma region "forward Extension"

				/* Mở rộng từ tất cả các đỉnh của embedding một cách song song.
				* Kết quả là các mở rộng hợp lệ được lưu trữ vào mảng dArrPointerExt
				* dArrPointerExt là một mảng lưu trữ pointer trỏ đến các mảng dExt
				* dArrPointerExt có số lượng phần tử bằng với kích thước của RightMostPath
				*/
				EXT** dArrPointerExt=nullptr;
				 int *dArrNoElemPointerExt = nullptr;
				 

				 int noElem_dArrPointerExt=noElem_dRMPath;
				cudaStatus = forwardExtension(dArrPointerEmbedding,noElem_dArrPointerEmbedding,dArrSizedQ,noElem_dArrSizedQ,dRMPath,noElem_dRMPath,d_O,d_LO,d_N,d_LN,numberOfElementd_O,numberOfElementd_N,maxOfVer,dArrPointerExt,noElem_dArrPointerExt,minLabel,maxid,dArrNoElemPointerExt);
				 if(cudaStatus!=cudaSuccess){
					 fprintf(stderr,"\n forwardExtension() in kernel.cu failed",cudaStatus);
					 goto Error;
				 }					


				cudaStatus = printInt(dArrNoElemPointerExt,noElem_dArrPointerExt);
				 if(cudaStatus!=cudaSuccess){
					 fprintf(stderr,"\n printInt() in kernel.cu failed",cudaStatus);
					 goto Error;
				 }	

				 //In nội dung mảng dArrPointerExt dựa vào mảng kích thước của từng phần tử dArrNoElemPointerExt 
				 printf("\n*******dArrPointerExt******\n");
				cudaStatus = printdArrPointerExt(dArrPointerExt,dArrNoElemPointerExt,noElem_dArrPointerExt);
				if(cudaStatus!=cudaSuccess){
					 fprintf(stderr,"\n printdArrPointerExt() in kernel.cu failed",cudaStatus);
					 goto Error;
				 }	
#pragma endregion

#pragma region "Unique Extension Extraction"				
				
				//Quá trình rút trích này không cần dùng đến Embedding
				UniEdge **dArrPointerUniEdge=nullptr;
				int noElem_dArrPointerUniEdge = noElem_dRMPath;
				int *dArrNoELemPointerUniEdge=nullptr; //Mảng này có số lượng phần tử bằng với dArrPointerUniEdge, nó cho biết số lượng phần tử tương ứng mối phần tử trong dArrPointerUniEdge
				//Duyệt qua các EXTk và trả về pointer UniEdge* dUniEdge

				 int *hArrNoElemPointerExt= (int*)malloc(sizeof(int)*noElem_dArrPointerExt);
				 if (hArrNoElemPointerExt==NULL){
					 printf("\nMalloc hArrNoElemPointerExt in kernel.cu failed");
					 exit(1);
				 }
				 else
				 {
					 cudaMemcpy(hArrNoElemPointerExt,dArrNoElemPointerExt,sizeof(int)*noElem_dArrPointerExt,cudaMemcpyDeviceToHost);
				 }
				 //Gọi hàm trích các mở rộng duy nhất
				cudaStatus = extractUniExtension(dArrPointerExt,noElem_dArrPointerExt,Lv,Le,dArrPointerUniEdge,noElem_dArrPointerUniEdge,dArrNoELemPointerUniEdge,hArrNoElemPointerExt,dArrNoElemPointerExt);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n extractUniExtension() in kernel.cu failed",cudaStatus);
					goto Error;
				}

				printf("\n***********dArrNoELemPointerUniEdge***********\n");
				cudaStatus =printInt(dArrNoELemPointerUniEdge,noElem_dArrPointerUniEdge);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n printInt dArrNoELemPointerUniEdge in kernel.cu failed",cudaStatus);
					goto Error;
				}

				printf("\n***********dArrPointerUniEdge***********\n");
				cudaStatus =printArrPointerUniEdge(dArrPointerUniEdge,dArrNoELemPointerUniEdge,noElem_dArrPointerUniEdge);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n printArrPointerUniEdge dArrPointerUniEdge in kernel.cu failed",cudaStatus);
					goto Error;
				}

#pragma endregion


#pragma region "Compute Support"
				
				/* 
					Tính độ hỗ trợ của các cạnh trong dArrPointerUniEdge
					Giải thuật:
					1. Duyệt qua từng phần tử trong dArrPointerUniEdge. 
					2. 
				*/
				
				int *hArrNoELemPointerUniEdge =(int*)malloc(sizeof(int)*noElem_dArrPointerUniEdge);
				if(hArrNoELemPointerUniEdge==NULL){
					printf("\nmalloc hArrNoElemPointerUniEdge in kernel.cu failed");
					exit(1);
				}
				else
				{
					cudaMemcpy(hArrNoELemPointerUniEdge,dArrNoELemPointerUniEdge,sizeof(int)*noElem_dArrPointerUniEdge,cudaMemcpyDeviceToHost);
				}

				//Các biến sau dùng để lưu trữ kết quả tính độ hỗ trợ của các Unique Edge trong mảng dArrPointerUniEdge.
				unsigned int **hArrPointerSupport=nullptr;
				unsigned int *hArrNoElemPointerSupport=nullptr;
				unsigned int noElem_hArrPointerSupport=noElem_dArrPointerUniEdge;
				//Gọi hàm computeSupportv2 để tính độ hỗ trợ và lưu kết quả vào hArrPointerSupport
				cudaStatus=computeSupportv2(dArrPointerExt,dArrNoElemPointerExt,hArrNoElemPointerExt,noElem_dArrPointerExt,dArrPointerUniEdge,dArrNoELemPointerUniEdge,hArrNoELemPointerUniEdge,noElem_dArrPointerUniEdge,hArrPointerSupport,hArrNoElemPointerSupport,noElem_hArrPointerSupport,maxOfVer);
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n computeSupportv2 in kernel.cu failed",cudaStatus);
					goto Error;
				}
				
				for (int i = 0; i < noElem_hArrPointerSupport; i++)
				{
					int noElem = hArrNoElemPointerSupport[i];
					unsigned int *dArr = hArrPointerSupport[i];
					for (int j = 0; j < noElem; j++)
					{
						printf("\nSupport of i:%d in j:%d:%d",i,j,dArr[j]);
					}
				}

#pragma endregion "Ending of compute support"


				free(hArrNoELemPointerUniEdge);
				//Giải phóng bộ nhớ mảng UniEdge **dArrPointerUniEdge
				cudaStatus = cudaFreeArrPointerUniEdge(dArrPointerUniEdge,dArrNoELemPointerUniEdge,noElem_dArrPointerUniEdge);		
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n cudaFreeArrPointerExt() in kernel.cu failed",cudaStatus);
					goto Error;
				}
				//Giải phóng bộ nhớ mảng dArrPointerExt
				cudaStatus = cudaFreeArrPointerExt(dArrPointerExt,dArrNoElemPointerExt,noElem_dArrPointerExt);		
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n cudaFreeArrPointerExt() in kernel.cu failed",cudaStatus);
					goto Error;
				}

				//Giải phóng bộ nhớ mảng dArrPointerEmbedding
				cudaStatus = cudaFreeArrPointerEmbedding(dArrPointerEmbedding,dArrSizedQ,noElem_dArrPointerEmbedding);		
				if(cudaStatus!=cudaSuccess){
					fprintf(stderr,"\n cudaFreeArrPointerEmbedding() in kernel.cu failed",cudaStatus);
					goto Error;
				}

				//Giải phóng bộ nhớ mảng hArrPointerSupport
				

			} //endif check graphismin
#pragma endregion "checkgraphismin"


			gspan.DFS_CODE.pop();			
		}//endif check minsup statification	
#pragma endregion "check minsup statification"
	} //end for. Kết thúc việc duyệt qua mảng d_UniqueExtension để tính độ hỗ trợ

#pragma endregion
	
	
#pragma region "Tam gac lai van de nay, se quay lai sau"
	///*
	//for (int i = 0; i < h_satisfyEdge.size(); i++)
	//{
	//	printf("\n h_satisfyEdge[%d]:%d",i,h_satisfyEdge[i]);
	//	printf("\n h_satisfyEdgeSupport[%d]:%d",i,h_satisfyEdgeSupport[i]);
	//}



	//getch();

	////Tiếp theo chúng ta dựa vào mảng h_frequentEdge để trích ra những cạnh phổ biến và xây dựng DFS_CODE cho chúng.
	////Sau khi xây dựng DFS_Code, chúng ta sẽ chuyển chúng sang đồ thị và ghi đồ thị đó vào file kết quả result.txt

	///*gspan.DFS_CODE.push(0,1,0,0,1);	
	//bool min = gspan.is_min();
	//printf("\n min:%d",min);*/

	////Gọi P là pattern và EP là các embedding của pattern P
	////Làm sao để có P? ==> Dựa vào h_frequentEdge để lấy cạnh trong d_UniqueExtension xây dựng DFS_Code P
	//for (int i = 0; i < h_satisfyEdge.size(); i++)
	//{		
	//	int li;
	//	int lij;
	//	int lj;
	//	int indexOfSatisfyEdge=h_satisfyEdge[i];
	//	int *d_arr_edgeLabel=nullptr;
	//	cudaStatus = getSatisfyEdge(d_UniqueExtension,noElem_d_UniqueExtension,indexOfSatisfyEdge,li,lij,lj,d_arr_edgeLabel);
	//	if(cudaStatus != cudaSuccess){
	//		fprintf(stderr,"\n getSatisfyEdge failed",cudaStatus);
	//		//goto labelError;
	//		exit(1);
	//	}

	//	int *h_arr_graphIdContainEmbedding=nullptr;
	//	int noElem_h_arr_graphIdContainEmbedding=0;
	//cudaStatus =getGraphIdContainEmbedding(li,lij,lj,d_ValidExtension,noElem_d_ValidExtension,h_arr_graphIdContainEmbedding,noElem_h_arr_graphIdContainEmbedding,maxOfVer); //hàm này được để trong calcSupport file
	//	if(cudaStatus != cudaSuccess){
	//		fprintf(stderr,"\n getGraphIdContainEmbedding failed",cudaStatus);
	//		//goto labelError;
	//		exit(1);
	//	}
	//	
	//	//printf("\n i:%d (li:%d, lij:%d, lj:%d)",i,li,lij,lj);
	//	//printInt(d_arr_edgeLabel,3);			
	//	//1.Xây dựng DFS_CODE, đồng thời ghi nhận lại minLabel và maxtoc của DFS_CODE
	//	gspan.DFS_CODE.push(0,1,li,lij,lj); //Cạnh đầu tiên của DFS_Code luôn có (vi,vj)=(0,1), khi mở rộng DFS_CODE thì 
	//	int minLabel = 0;
	//	int maxtoc = 1; //là id của đỉnh cuối cùng trên rmpath của DFS_CODE_MIN
	//					//tuỳ vào backward hay forward để tính (vi,vj). 
	//	//2. Ở đây các cạnh đã thoả minDFS_CODE, nên không cần xét minDFS_CODE trong trường hợp này.

	//	//3. Chuyển DFS_CODE sang đồ thị và ghi kết quả vào tập tin			
	//	//int graph[3]={0,1,2};
		//gspan.report(h_arr_graphIdContainEmbedding,noElem_h_arr_graphIdContainEmbedding,h_satisfyEdgeSupport[i]);
	//	//4. Tìm các Embedding của DFS_CODE
	//	//xây dựng embedding cho mở rộng thoả minsup (dùng struct_Q *device_arr_Q=NULL; để lưu trữ các cột Q của embeddings)
	//		struct_Q *device_arr_Q=nullptr; //các cột Q và thông tin của nó được lưu trữ trong mảng cấu trúc struct_Q *device_arr_Q;
	//		printf("\n***********support of (%d,%d,%d) >= %d --> create embeddings for DFS_CODE************",li,lij,lj,minsup);
	//		cudaStatus=createForwardEmbedding(d_ValidExtension,noElem_d_ValidExtension,li,lij,lj,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,minsup,maxOfVer,numberOfGraph,noDeg,device_arr_Q);
	//		if (cudaStatus!=cudaSuccess){
	//			fprintf(stderr,"\ncreateForwardEmbedding failed");
	//			exit(1);
	//		}

	//		//Lấy số lượng phần tử của một cột Q bất kỳ trong mảng device_arr_Q
	//		/*
	//		printf("\nPrint information of size of the last element of d_arr_Q:");	
	//		int positionLastElement = 1;
	//		int *dsizeOfLastElement;
	//		int hsizeOfLastElement=0;
	//		
	//		cudaStatus = cudaMalloc((void**)&dsizeOfLastElement,sizeof(int));
	//		if(cudaStatus!=cudaSuccess){
	//			fprintf(stderr,"\n cudaMalloc dsizeOfLastElement failed");
	//			//goto Error;
	//			exit(1);
	//		}
	//		else
	//		{
	//			cudaMemset(dsizeOfLastElement,0,sizeof(int));
	//		}
	//		*/
	//		//Hàm kernelGetInformationLastElement sẽ lấy kích thước của cột Q trong mảng device_arr_Q và lưu kết quả vào biến
	//		/*
	//		kernelGetInformationLastElement<<<1,1>>>(device_arr_Q,positionLastElement,dsizeOfLastElement); 
	//		cudaDeviceSynchronize();
	//		cudaMemcpy(&hsizeOfLastElement,dsizeOfLastElement,sizeof(int),cudaMemcpyDeviceToHost);
	//		printf("\nhsizeOfLastElement:%d",hsizeOfLastElement);
	//		*/
	//		//Truy xuất tất cả các Embeddings khi truyền vào một mảng cấu trúc struct_Q: device_arr_Q
	//		/*
	//		printf("\n\nPrint all embedding from the last element of device_arr_Q");
	//		PrintAllEmbedding<<<1,hsizeOfLastElement>>>(device_arr_Q,1,hsizeOfLastElement);
	//		cudaDeviceSynchronize();
	//		*/
	//		
	//		//11.5.2 Tìm các mở rộng cho các Embedding của DFS_CODE
	//		/* 
	//		- Sau khi đã xây dựng được các Embedding columns để biểu diễn embeddings cho các frequent 1-edge extension.
	//		- Cụ thể các cột embedding ở đây là một mảng device_arr_Q, mỗi phần tử của device_arr_Q là một cột Q, với chỉ số 
	//		được tính bắt đầu từ 0 (Q0, Q1, Q2,...).
	//		- Mảng RMPath: dùng để lưu trữ index của device_arr_Q mà tại đó cột Q thuộc Right Most Path. 
	//		- Biến lastColumn: lữu trữ index của cột cuối cùng (tức là đỉnh phải nhất của Embedding). Từ cột này chúng ta có
	//		thể lần ngược để duyệt qua tất cả các Q thuộc Right Most Path
	//		1. Viết hàm getExtension
	//		- Input: device_arr_Q,lastColumn,RMPath,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,minsup.
	//		- Output: RMPath
	//		*/
	//		int lastColumn=1; //ở đây các embedding chỉ có 1 cạnh, nên Q cuối cùng nằm ở vị trí index=1 trong mảng device_arr_Q,
	//							//Khi mở rộng embedding và bổ sung thêm Q mới vào sau mảng device_arr_Q thì chúng ta phải cập nhật lại lastColumn						
	//		vector<int> RighMostPath(2); //chứa index của mảng device_arr_Q mà tại đó cột Q thuộc right most path
	//		RighMostPath.at(0)=0;							// Tương tự lastColumn, Khi mở rộng embedding và bổ sung thêm Q mới vào sau mảng device_arr_Q thì chúng ta phải cập nhật lại RightMostPath	
	//		RighMostPath.at(1)=1;
	//		cHistory **dH=nullptr; //hàm getExtension sẽ trả về dH và số lượng phần tử của dH (chính bằng số lượng embedding) của DFS_CODE đang xét
	//		int numberElem_dH=0; //Số lượng embeddings của DFS_CODE
	//		cudaStatus = getExtension(device_arr_Q,lastColumn,RighMostPath,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,minsup,maxOfVer,numberOfGraph,noDeg,dH,numberElem_dH); 
	//		if(cudaStatus !=cudaSuccess){
	//			fprintf(stderr,"\n getExtension failed",cudaStatus);
	//			exit(1);
	//		}

	//		//Duyệt qua từng embedding trong device_arr_Q và đánh dấu những đỉnh và cạnh thuộc embedding trong dH là 2
	//		cudaStatus = markEmbedding(dH,device_arr_Q,lastColumn,numberElem_dH,maxOfVer,d_O,d_N);
	//		if (cudaStatus!=cudaSuccess){
	//			fprintf(stderr,"\n markEmbedding function has been failed.",cudaStatus);
	//			/*goto labelError;*/
	//			exit(1);
	//		}			

	//		printf("\n****************Display history of all embedding in dH array***********"); //kiểm tra thử dữ liệu của mảng dH trên device sau khi đã đánh dấu các embedding thuộc right most path
	//		kernelPrintdeviceH<<<1,1>>>(dH,numberElem_dH);
	//		cudaDeviceSynchronize();
	//		
	//		//Duyệt qua từng cột Q thuộc RightMostPath trong device_arr_Q và tìm những mở rộng hợp lệ từ các đỉnh trên cột Q
	//		for (int i = RighMostPath.size()-1; i >=0; i--)
	//		{					
	//			Extension *d_arrE=nullptr;
	//			int numberElement_d_arrE=0;
	//			cudaStatus=getValidExtensionFromEmbeding(d_arrE,numberElement_d_arrE,device_arr_Q,RighMostPath[i],dH,numberElem_dH,maxOfVer,d_O,d_LO,d_N,d_LN,numberOfElementd_O,numberOfElementd_N,lastColumn);
	//			if(cudaStatus!=cudaSuccess){
	//				fprintf(stderr,"\ngetValidExtensionFromEmbedding failed",cudaStatus);
	//				//goto labelError;
	//				exit(1);
	//			}

	//			printfExtension(d_arrE,numberElement_d_arrE);
	//			cudaDeviceSynchronize();
	//		}

	//		//5. Xây dựng hàm để lặp lại quá trình khai thác 
	//		gspan.DFS_CODE.pop();
	//}

	//printf("\n***********Finished: compute support for valid unique extension in d_UniqueExtension **********");
	//printf("\n***********Press the Enter key to continous**********\n");

	//getch();
#pragma endregion

Error:
	//giải phóng vùng nhớ của dữ liệu
	free(h_resultSup);
	free(hNumberEdgeInEachGraph);
	cudaFree(dNumberEdgeInEachGraph);
	cudaFree(d_O);
	cudaFree(d_LO);
	cudaFree(d_N);
	cudaFree(d_LN);	
	//	cudaFree(d_singlePattern);
	cudaFree(d_Extension);
	cudaFree(V);
	cudaFree(index);
	cudaFree(d_ValidExtension);	
	cudaFree(d_allPossibleExtension);
	cudaFree(d_allPossibleExtensionScanResult);
	cudaFree(d_UniqueExtension);
	cudaFree(d_B);
	
	cudaDeviceReset();	

	fout.close();
	//delete[] arrayO;
	free(h_tempEdge);
	delete[] arrayN;
	delete[] arrayLO;
	delete[] arrayLN;

	getch();
	return 0;
}
