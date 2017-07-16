#pragma once;
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include "ExtensionStructure.h"
#include "Embedding.h"

extern "C" inline __global__ void kernelPrintEmbedding(struct_Embedding *d_Embedding,int noElem_Embedding);
extern "C" inline cudaError_t printEmbedding(struct_Embedding *d_Embedding,int noElem_Embedding);

extern "C" inline __global__ void kernelPrintf(int*,int sizeO);
extern "C" inline cudaError_t printInt(int*,int sizeO);

extern "C" inline __global__ void kernelPrintFloat(float* array,int numberElementOfArray);
extern "C" inline cudaError_t  printFloat(float* array,int numberElementOfArray);

extern "C" inline __global__ void kernelPrintExtention(Extension *d_Extension,unsigned int n);
extern "C" inline cudaError_t printfExtension(Extension *d_E,unsigned int noElem_d_E);

extern "C" inline __global__ void kernelPrintUniEdge(UniEdge *d_Extension,unsigned int n);
extern "C" inline cudaError_t printfUniEdge(UniEdge *d_UniEdge,unsigned int noElem_d_UniEdge);

extern inline cudaError_t printUnsignedInt(unsigned int* d_array,int noElem_d_Array);