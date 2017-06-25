#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "Embedding.h"
#include "markEmbedding.cuh"
#include "getExtensionFromEmbedding.h"

using namespace std;


class cHistory
{
public:
	int n; //number element of d_arr_HO.
	//int noEle_d_HN;
	int m; //number element of d_arr_HLN.
	int *d_arr_HO;
	//int *d_arr_HN;
	int *d_arr_HLN;
	__device__ __host__ cHistory();	//constructor
	__device__ __host__ cHistory(int,int);
	__device__ __host__ void print();
	__device__ __host__ void printmn();	
};




class ArrHistory{
public:
	int n; //số lượng phần tử của vecA;
	cHistory **vecA;
	__device__ __host__ ArrHistory();
	__device__ __host__ ArrHistory(int);
	__device__ __host__ void print();
};

extern "C" inline __global__ void kernelFindNumberOfEdgeInAGraph(int *d_arr_number_HLN,struct_Q *device_arr_Q,int numberEmbedding,int lastColumn,unsigned int maxOfVer,int *d_O,unsigned int numberOfGraph,unsigned int noDeg);
extern "C" inline __global__ void	kernelGetnoEle_Embedding(struct_Q *device_arr_Q,int lastColumn,int *noEle_Embeddings);
extern "C" inline __global__ void kernelPrintdeviceH(cHistory **device_H,int numberEmbeddings);
extern "C" inline cudaError_t getExtension(struct_Q *device_arr_Q,int lastColumn,vector<int> &RMPath,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int minsup,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg,cHistory **&dH,int &numberElem_dH);
extern "C" inline cudaError_t	findNumberOfEdgeInAGraph(int *d_arr_number_HLN,struct_Q *device_arr_Q,int numberEmbedding,int lastColumn,unsigned int numberOfGraph,int *d_O);
