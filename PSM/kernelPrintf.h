#pragma once;
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include "ExtensionStructure.h"

extern "C" inline __global__ void kernelPrintf(int*,int sizeO);
extern "C" inline cudaError_t printInt(int*,int sizeO);

extern "C" inline __global__ void kernelPrintFloat(float* array,int numberElementOfArray);
extern "C" inline cudaError_t  printFloat(float* array,int numberElementOfArray);

extern "C" inline __global__ void kernelPrintExtention(Extension *d_Extension,unsigned int n);
extern "C" inline cudaError_t printfExtension(Extension *d_E,unsigned int noElem_d_E);