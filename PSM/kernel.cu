#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
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
//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>	
using namespace std;

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
		goto labelError;
	}

	cudaStatusAllocate =cudaMalloc((int**) &d_LO,nBytesLO);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto labelError;
	}
	cudaStatusAllocate =cudaMalloc((int**) &d_N,nBytesN);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto labelError;
	}
	cudaStatusAllocate =cudaMalloc((int**) &d_LN,nBytesLN);
	if (cudaStatusAllocate!=cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		goto labelError;
	}


	//chép dữ liệu từ bốn mảng O,LO,N,LN từ Host sang GPU. Đây chính là CSDL đồ thị dùng để khai thác trên GPU
	cudaStatusAllocate = cudaMemcpy(d_O,arrayO,nBytesO,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto labelError;
	}

	cudaStatusAllocate = cudaMemcpy(d_LO,arrayLO,nBytesLO,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto labelError;
	}

	cudaStatusAllocate = cudaMemcpy(d_N,arrayN,nBytesN,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto labelError;
	}

	cudaStatusAllocate = cudaMemcpy(d_LN,arrayLN,nBytesLN,cudaMemcpyHostToDevice);
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto labelError;
	}


	//Đồng bộ đồng thời kiểm tra xem đồng bộ có lỗi không
	cudaStatusAllocate= cudaDeviceSynchronize();
	if (cudaStatusAllocate != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatusAllocate);
		goto labelError;
	}
	//xác định grid and block structure
	dim3 block(512);
	dim3 grid((nBytesO + block.x -1)/block.x);
	printf("grid %d; block %d;\n",grid.x,block.x);


	//********kiểm tra đảm bảo dữ liệu ở GPU giống với Host********

	cudaError_t cudaStatus = checkDataBetweenHostAndGPU(d_O,d_LO,d_N,d_LN,sizeOfarrayO,noDeg,arrayO,arrayLO,arrayN,arrayLN,nBytesO,nBytesLO,nBytesN,nBytesLN);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkDataBetweenHostAndGPU failed!");
		return 1;
	}
	printf("\n***********Press the Enter key to continous**********");
	getch();
	/*
	//********Đếm số đỉnh song song và loại nhỏ những đỉnh nhỏ hơn minsup****
	//Nếu số đỉnh nhỏ hơn minSup thì đánh dấu đỉnh đó là -1 trong mảng O và mảng LO và các cạnh liên quan đến đỉnh đó cũng được đánh dấu là -1

	cudaStatus = markInvalidVertex(d_O,d_LO,sizeOfarrayO,minsup);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "markInvalidVertex failed!");
		return 1;
	}
	*/
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

	/* //May/04/2017: Trích các cạnh từ CSDL và lưu vào d_Extension.
		1. Tạo một cấu trúc Extension để lưu trữ các mở rộng: DFSCode của cạnh mở rộng (vi,vj,li,lij,lj),global from vertex id(vgi),
		global to vertex id (vgj) và pointer trỏ đến header của embedding tương ứng với cạnh mở rộng.
		2. Tạo một mảng có kích thước bằng với kích thước của d_N để lưu trữ các cạnh mở rộng ban đầu, lúc chưa có bất kỳ một cạnh
		phổ biến nào (P=0).
		3. Tạo một kernel có số lượng threads bằng d_O, mỗi thread sẽ xử lý một đỉnh. Nhiệm vụ của thread là đọc các cạnh kề với nó rồi
		lưu trữ thông tin vào mảng Extension tương ứng tại vị trí.
	*/

	int numberOfElementd_N=noDeg;
	size_t nBytesOfArrayExtension = numberOfElementd_N*sizeof(Extension);
	Extension *d_Extension;
	cudaStatus= cudaMalloc((Extension**)&d_Extension,nBytesOfArrayExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"CudaMalloc d_Extension fail",cudaStatus);
		exit(1);
	}


	int numberOfElementd_O=sizeOfarrayO;	
	cudaStatus = getAndStoreExtension(d_Extension,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Le,Lv);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize getAndStoreExtension failed",cudaStatus);
		return 1;
	}

	CHECK(printfExtension(d_Extension,numberOfElementd_N));

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

	/* //07-May-2017: Extract unique Edge from d_Extension
	1. Tiếp theo, chúng ta exclusive scan mảng V để thu được index chỉ vị trí của các valid edge trong d_Extension.
	2. Input data: mảng V
	3. Output data: mảng index
	Mảng Index có số lượng phần tử bằng với mảng V
	*/

	int* index;
	cudaStatus=cudaMalloc((int**)&index,numberElementd_Extension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"Cuda Malloc failed",cudaStatus);
		return 1;
	}	

	cudaStatus = scanV(V,numberElementd_Extension,index);
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize getAndStoreExtension failed",cudaStatus);
		return 1;
	}

	printf("\n Scan Result index: ");
	kernelPrintf<<<grid,block>>>(index,numberElementd_Extension);

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaDeviceSynchronize kernelPrintf failed",cudaStatus);
		exit(1);
	}

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
	
	cudaStatus=getValidExtension(d_Extension,V,index,numberElementd_Extension,d_ValidExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getValidExtension failed",cudaStatus);
		return 1;
	}

	printf("\nNumber Element of d_ValidExtension:%d",noElem_d_ValidExtension);
	CHECK(printfExtension(d_ValidExtension,noElem_d_ValidExtension));

	/* //Hàm getUniqueExtension: Trích tra các cạnh duy nhất dựa vào nhãn Li, Lj và Lij của edge extension
	1. Tạo mảng d_allPossibleExtension có kích thước là noElem_allPossibleExtension=Le*Lv*Lv để lưu trữ 
		tất cả các mở rộng có thể có của tất cả các đỉnh. Các mở rộng có thể có từ 1 đỉnh trên righ most path có kích thước là Le*Lv.
	2. Viết hàm getUniqueExtension để gán giá trị là 1 tại vị trí Li Lj tương ứng của extension.
	*/
	unsigned int noElem_allPossibleExtension=Le*Lv*Lv;
	int *d_allPossibleExtension;

	cudaStatus=cudaMalloc((int**)&d_allPossibleExtension,noElem_allPossibleExtension*sizeof(int));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_allPossibleExtension failed",cudaStatus);
		return 1;
	}

	/* //09-May-2017 */
	
	cudaStatus=getUniqueExtension(d_ValidExtension,noElem_d_ValidExtension,Lv,Le,d_allPossibleExtension);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"getUniqueExtension failed",cudaStatus);
		return 1;
	}
	printf("\n\nLe:%d Lv:%d",Le,Lv);
	printf("\nd_allPossibleExtension: ");
	printInt(d_allPossibleExtension,noElem_allPossibleExtension);
	
	/* //Tiếp theo chúng ta exclusive scan mảng d_allPossibleExtension để thu được index phục vụ cho việc
		lưu trữ các unique extension.
	1. Chúng ta khởi tạo một mảng d_allPossibleExtensionScanResult có kích thước bằng với kích thước của d_allPossibleExtension
		đồng thời khởi tạo giá trị cho các phần tử của nó là 0.
	2. Sau khi thu được kết quả scan, chúng ta tạo một mảng Extension* d_UniqueExtension có kích thước bằng với kích thước của giá trị phần tử cuối cùng
		trong mảng d_allPossibleExtensionScanResult cộng với 1.
	3. Dựa vào giá trị index trong mảng d_allPossibleExtensionScanResult để suy ra các nhãn Li, Lj và Lij của Extension và lưu trữ chúng vào d_UniqueExtension
	*/
	
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

	cudaStatus=scanV(d_allPossibleExtension,noElem_allPossibleExtension,d_allPossibleExtensionScanResult);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\nscanV function failed",cudaStatus);
		return 1;
	}

	printf("\n\nd_allPossibleExtensionScanResult:\n");
	CHECK(printInt(d_allPossibleExtensionScanResult,noElem_allPossibleExtension));

	//Lấy giá trị của phần tử cuối cùng trong mảng d_allPossibleExtensionScanResult
	int noElem_d_UniqueExtension=0;

	cudaStatus=getLastElement(d_allPossibleExtensionScanResult,noElem_allPossibleExtension,noElem_d_UniqueExtension);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"getLastElement failed",cudaStatus);
		return 1;
	}
	//noElem_d_UniqueExtension++;
	//printf("\n\nnoElem_d_UniqueExtension:%d",noElem_d_UniqueExtension);
	Extension *d_UniqueExtension;
	cudaStatus=cudaMalloc((Extension**)&d_UniqueExtension,noElem_d_UniqueExtension*sizeof(Extension));
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"cudaMalloc d_UniqueExtension failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_UniqueExtension,0,noElem_d_UniqueExtension*sizeof(Extension));
	}


	cudaStatus=calcLabelAndStoreUniqueExtension(d_allPossibleExtension,d_allPossibleExtensionScanResult,noElem_allPossibleExtension,d_UniqueExtension,noElem_d_UniqueExtension,Le,Lv);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\n\ncalcLabelAndStoreUniqueExtension function failed",cudaStatus);
		return 1;
	}

	printf("\n\nUnique Extension:");
	printfExtension(d_UniqueExtension,noElem_d_UniqueExtension);

	/* //10-May-2017: Tính độ hỗ trợ
	1. Trước tiên, chúng ta cấp phát một mảng d_B, mảng này có số lượng phần tử bằng với số lượng phần tử của d_ValidExtension
		Mảng d_B dùng để đánh dấu vị trí biên (boundary: nơi tiếp giáp giữa 2 đồ thị)
	2. Exclusive scan mảng d_B và lưu kế quả vào d_scanB_result
	3. Khởi tạo mảng d_F có số lượng phần tử bằng với giá trị của phần tử cuối cùng của mảng d_scanB_Result cộng 1
	4. Tính độ hỗ trợ của từng phần tử trong mảng d_UniqueExtension dựa vào d_ValidExtension và ScanB_Result
	*/
	
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

	cudaStatus=calcBoundary(d_ValidExtension,noElem_d_ValidExtension,d_B,Lv);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"calcBoundary function failed",cudaStatus);
		return 1;
	}

	printf("\n\nd_B:\n");
	printInt(d_B,noElement_d_B);

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
	int noElement_F=0;
	cudaStatus=getLastElement(d_scanB_Result,noElement_d_B,noElement_F);
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ngetLastElement function failed",cudaStatus);
		return 1;
	}

	noElement_F++;
	printf("\nnoElement_F:%d",noElement_F);

	float *d_F;
	cudaStatus=cudaMalloc((int**)&d_F,noElement_F*sizeof(float));
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaMalloc d_F failed",cudaStatus);
		return 1;
	}
	else
	{
		cudaMemset(d_F,0,noElement_F*sizeof(float));
	}
		

	/* //Gọi hàm calcSupport để tính độ hỗ trợ cho các mở rộng trong mảng d_UniqueExtension đồng thời gọi hàm buildEmbedding để xây dựng embedding cho mở rộng thoả minsup*/
	cudaStatus=calcSupport(d_UniqueExtension,noElem_d_UniqueExtension,d_ValidExtension,noElem_d_ValidExtension,d_scanB_Result,d_F,noElement_F,minsup,d_O,d_LO,numberOfElementd_O,d_N,d_LN,numberOfElementd_N,Lv,Le,maxOfVer,numberOfGraph,noDeg);
	if (cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncalcSupport function failed",cudaStatus);
		return 1;
	}



labelError:
	//giải phóng vùng nhớ của dữ liệu
	cudaFree(d_O);
	cudaFree(d_LO);
	cudaFree(d_N);
	cudaFree(d_LN);	
//	cudaFree(d_singlePattern);
	cudaFree(d_Extension);
	cudaFree(V);
	cudaFree(d_ValidExtension);	
	cudaFree(d_UniqueExtension);
	cudaFree(d_B);


	cudaDeviceReset();	

	fout.close();
	//delete[] arrayO;
	delete[] arrayN;
	delete[] arrayLO;
	delete[] arrayLN;

	getch();
	return 0;
}
