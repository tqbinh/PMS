#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "ExtensionStructure.h"
#include "kernelPrintf.h"
#include "scanV.h"
#include "getLastElement.h"
#include "Embedding.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>	
#include <iostream>
using namespace std;

extern "C" inline __global__ void kernelCreateForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanResult,int li,int lij,int lj,struct_Embedding *d_Q1,struct_Embedding *d_Q2);
extern "C" inline __global__ void kernelMatchLastElement(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj,bool same);
extern "C" inline __global__ void kernelMarkExtension(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_M,int li,int lij,int lj);
extern "C" inline cudaError_t createForwardEmbedding(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int li,int lij,int lj);