#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"
#include "reduction.h"
#include "createForwardEmbedding.h"
#include "getLastElement.h"
#include "kernelPrintf.h"
#include "scanV.h"
#include <vector>


#define blocksize 512

extern "C" inline cudaError_t calcSupport(Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,float *d_F,unsigned int noElem_F,unsigned int minsup,int *d_O,int *d_LO,int numberOfElementd_O,int *d_N,int *d_LN,int numberOfElementd_N,unsigned int Lv,unsigned int Le,unsigned int maxOfVer,unsigned int numberOfGraph,unsigned int noDeg,vector<int> &h_satisfyEdge,vector<int> &h_satisfyEdgeSupport);
extern "C" inline cudaError_t getGraphIdContainEmbedding(int *d_arr_edgeLabel,Extension *d_ValidExtension,int noElem_d_ValidExtension,int *&h_graphIdContainEmbedding,int &noElem_h_arr_graphIdContainEmbedding,unsigned int maxOfVer);