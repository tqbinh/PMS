#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "ExtensionStructure.h"
#include "kernelPrintf.h"
#include "scanV.h"
#include "getLastElement.h"
#include "Embedding.h"
#include "getExtension.h"
#include <vector>
//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>	
#include <iostream>
using namespace std;

extern "C" inline __global__ void kernelGetInformationLastElement(struct_Q *d_arr_Q,int positionLastElement,int *sizeOfLastElement);
extern "C" inline __global__ void PrintAllEmbedding(struct_Q *device_arr_Q,int position,int noElemOfLastColumn);
extern "C" inline __global__ void kernelCreateForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanResult,int li,int lij,int lj,struct_Embedding *d_Q1,struct_Embedding *d_Q2);
extern "C" inline __global__ void kernelMatchLastElement(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj,bool same);
extern "C" inline __global__ void kernelMarkExtension(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_M,int li,int lij,int lj);
extern "C" inline cudaError_t createForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int minsup,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg,struct_Q *&device_arr_Q);