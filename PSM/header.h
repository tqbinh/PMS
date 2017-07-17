#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include "kernelPrintf.h"
#include "createForwardEmbedding.h"
#include "device_functions.h"
#include "reduction.h"

#define blocksize 512

//cấu trúc Embedding
struct Embedding
{	
	int idx;
	int vid;	
	Embedding():idx(-1),vid(-1){};
};

//cấu trúc struct_Q
struct Q
{
	int numEle_dArrQColumn; //số lượng phần tử mảng trong Q, vì 3 mảng trong Q đều có số lượng phần tử giống nhau.
	int numEle_dArrPrevQ;
	int numEle_dArrSizeOfQColumn;	
	int *dArrPrevQ; //d_arrPrevQ
	int *dArrSizeOfQColumn; // d_arrSizedQ
	Embedding **dArrQColumn; // is pdQ
	Q():numEle_dArrQColumn(0),numEle_dArrPrevQ(0),numEle_dArrSizeOfQColumn(0),dArrPrevQ(0),dArrSizeOfQColumn(0),dArrQColumn(0){};
};

//Cấu trúc của một EXT
struct EXT
{
	int vi,vj,li,lij,lj; /*DFS code của cạnh mở rộng, Ở đây giá trị của (vi,vj) tuỳ thuộc vào 2 thông tin: Mở rộng là backward hay forward và mở rộng tử đỉnh nào trên RMPath. Nếu là mở rộng forward thì cần phải biết maxid của DFS_CODE hiện tại */
	int vgi,vgj;	/* globalId vertex của cạnh mở rộng*/
	//Hai thông tin bên dưới cho biết cạnh mở rộng từ embedding nào.
	int posColumn; /* vị trí của embedding column trong mảng các embedding Q*/
	int posRow; //vị trí của embedding trong cột Q
	EXT():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0),posColumn(0),posRow(0){};
};





extern inline cudaError_t makeColumnQ(Embedding *dQ,int sizedQ,Embedding **&pdQ,int &sizepdQ,int *&d_arrSizedQ,int &sized_arrSizedQ,int *&d_arrPrevQ,int &sized_arrPrevQ,int iPrevQ,int &first);
extern  inline cudaError_t createEmbeddingElement(Embedding *&dQ,int sizeQ,int &first); //kết quả dQ1 sẽ được trả về cho hàm main nên đối số của nó là một tham chiếu.
extern inline cudaError_t getPointer(Embedding **&pdQ,int &sizepdQ,Embedding *dQ);//pdQ lưu trữ địa chỉ của dQ
extern inline cudaError_t getSizedQ(int *&d_arrSizedQ,int &sized_arrSizedQ,int sizedQ); //trả về h_arrSizedQ;

 extern inline cudaError_t copyDeviceToDeviceInt(int *d_FromIntArray,int *d_ToIntArray,int currentSize);


extern  inline cudaError_t print(Embedding *dQ,int sizedQ);
extern  inline cudaError_t print(Embedding **pdQ,int *d_arrSizedQ,int *d_arrPrevQ,int sizepdQ);
extern inline cudaError_t print(int *dArray,int sizsedArray);
extern inline cudaError_t printEmbedding(Embedding **pdQ,int *d_arrSizedQ,int *d_arrPrevQ,int sizepdQ,int firstEmbedding,int lastColumnQ);


extern inline cudaError_t createEmbeddingRoot(Embedding **&dArrPointerEmbedding,int &noElem_dArrPointerEmbedding,int *&dArrSizedQ,int &noElem_dArrSizedQ,int *&dArrPrevQ,int &noElem_dArrPrevQ,Extension *d_ValidExtension,int noElem_d_ValidExtension,int li,int lij,int lj);

extern inline cudaError_t createRMPath(int *&dRMPath,int &noElem_dRMPath); // Hàm tạo dRMPath
extern inline cudaError_t printRMPath(int *dRMPath,int noElem_dRMPath);


extern inline cudaError_t createGraphHistory(Embedding **dArrPointerEmbedding,int *dArrSizedQ,int *dArrPrevQ,int noElem_dArrPointerEmbedding,int noElem_dArrSizedQ,int noElem_dArrPrevQ,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int maxOfVer,int **&dArrPointerdHO,int &noElem_dArrPointerdHO,int **&dArrPointerdHLN,int *&dArrNumberEdgeOfEachdHLN,int *hNumberEdgeInEachGraph,int noElem_hNumberEdgeInEachGraph,int *dNumberEdgeInEachGraph);
extern inline cudaError_t findNumberOfEmbedding(int *dArrSizedQ,int noElem_dArrSizedQ,int &noElem_dArrPointerdHO);
inline cudaError_t createElementdHO(int *&dHO,int maxOfVer);
extern inline cudaError_t assignPointer(int **&dArrPointerdHO,int pos,int *dHO);
extern inline cudaError_t printDoublePointerInt(int **dArrPointerdHO,int noElem_dArrPointerdHO,unsigned int maxOfVer);
extern inline cudaError_t printDoublePointerInt(int **dArrPointerdHLN,int noElemOfEmbedding,int *dArrNumberEdgeOfEachdHLN);
extern inline cudaError_t createdArrPointerdHO(int **&dArrPointerdHO,int noElem_dArrPointerdHO,unsigned int maxOfVer);
extern inline cudaError_t getNumberOfEdgeInGraph(int *d_O,int numberOfElementd_N,unsigned int maxOfVer,int *&hNumberEdgeInEachGraph,int *&dNumberEdgeInEachGraph,unsigned int numberOfGraph);
extern inline cudaError_t createElementdHLN(int *&dHLN,int noElem_dHLN);
extern  cudaError_t findGraphIdOfAllEmbedding(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *&hArrGraphId,unsigned int maxOfVer,int *&dArrGraphId,int noElemOfEmbedding,int *dArrSizedQ);
extern inline cudaError_t createdArrPointerdHLN(int **&dArrPointerdHLN,int noElem_dArrPointerdHO,int *hNumberEdgeInEachGraph,int *hArrGraphId);
extern inline cudaError_t printdArrPointerdHLN(int **dArrPointerdHLN,int noElem_dArrPointerdHO,int *dNumberEdgeInEachGraph,int *dArrGraphId);
extern inline cudaError_t createdArrNumberEdgeOfEachdHLN(int *&dArrNumberEdgeOfEachdHLN,int noElemOfEmbedding,int *dArrGraphId,int *dNumberEdgeInEachGraph);
extern inline cudaError_t createEmbeddingRoot1(Embedding **&dArrPointerEmbedding,int &noElem_dArrPointerEmbedding,int *&dArrSizedQ,int &noElem_dArrSizedQ,Extension *d_ValidExtension,int noElem_d_ValidExtension,int li,int lij,int lj);
extern inline cudaError_t printAllEmbeddingColumn(Embedding **dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrPointerEmbedding);
extern inline cudaError_t printEmbeddingFromPos(Embedding **dArrPointerEmbedding,int posColumn,int posRow);
extern inline cudaError_t forwardExtension(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrSizedQ,int *dRMPath,int noElem_dRMPath,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,unsigned int maxOfVer,EXT **&dArrPointerExt,int &noElem_dArrPointerExt,int minLabel,int maxid,int *&dArrNoElemPointerExt);
extern cudaError_t forwardExtensionQ(Embedding **dArrPointerEmbedding,int noElem_dArrPointerEmbedding,int *dArrSizedQ,int noElem_dArrSizedQ,int noElem_Embedding,int idxQ,EXT *&dExt,int &noElem_dExt,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,unsigned int maxOfVer,int minLabel,int maxid);
extern inline cudaError_t findMaxDegreeOfVer(Embedding **dArrPointerEmbedding,int idxQ,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N,unsigned int maxOfVer,int &maxDegreeOfVer,float *&dArrDegreeOfVid);
extern inline cudaError_t findDegreeOfVer(Embedding **dArrPointerEmbedding,int idxQ,int *d_O, int numberOfElementd_O,int noElem_Embedding,int numberOfElementd_N, unsigned int maxOfVer,float *&dArrDegreeOfVid);
extern inline cudaError_t printdArrV(struct_V *dArrV,int noElem_dArrV,EXT *dArrExtension);
extern inline cudaError_t extractValidExtensionTodExt(EXT *dArrExtension,struct_V *dArrV,int noElem_dArrV,EXT *&dExt,int &noElem_dExt);
extern inline cudaError_t printdExt(EXT *dExt,int noElem_dExt);
extern inline cudaError_t printdArrPointerExt(EXT **dArrPointerExt,int *dArrNoElemPointerExt,int noElem_dArrPointerExt);
extern inline cudaError_t cudaFreeArrPointerExt(EXT **&dArrPointerExt,int *&dArrNoElemPointerExt,int noElem_dArrPointerExt);
extern inline cudaError_t cudaFreeArrPointerEmbedding(Embedding **&dArrPointerEmbedding,int *&dArrSizedQ,int noElem_dArrPointerEmbedding);
extern inline cudaError_t extractUniExtension(EXT **dArrPointerExt,int noElem_dArrPointerExt,int Lv,int Le,UniEdge **&dArrPointerUniEdge,int noElem_dArrPointerUniEdge,int *&dArrNoELemPointerUniEdge,int *hArrNoElemPointerExt,int *dArrNoElemPointerExt);
extern inline cudaError_t assigndAllPossibleExtension(EXT **dArrPointerExt,int posdArrPointerExt,int Lv,int Le,int *dArrAllPossibleExtension,int noElem_PointerExt);
extern inline cudaError_t printArrPointerUniEdge(UniEdge **dArrPointerUniEdge,int *dArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge);
extern inline cudaError_t assigndArrUniEdge(int *dArrAllPossibleExtension,int *dArrAllPossibleExtensionScanResult,int noElem_dArrAllPossibleExtension,UniEdge *&dArrUniEdge,int Lv,int *dFromLi);
extern inline cudaError_t cudaFreeArrPointerUniEdge(UniEdge **&dArrPointerUniEdge,int *&dArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge);
extern inline cudaError_t computeSupportv2(EXT **dArrPointerExt,int *dArrNoElemPointerExt,int *hArrNoElemPointerExt,int noElem_dArrPointerExt,UniEdge **dArrPointerUniEdge,int *dArrNoELemPointerUniEdge,int *hArrNoELemPointerUniEdge,int noElem_dArrPointerUniEdge,unsigned int **&hArrPointerSupport,unsigned int *&hArrNoElemPointerSupport,unsigned int noElem_hArrPointerSupport,unsigned int maxOfVer);
extern inline cudaError_t findBoundary(EXT **dArrPointerExt,int *hArrNoElemPointerExt,int pos,unsigned int *&dArrBoundary,unsigned int maxOfVer);

