#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"

extern "C" inline cudaError_t getLastElement(int* inputArray,unsigned int numberElementOfInputArray,int &outputValue);
extern "C" inline cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);
extern "C" inline cudaError_t getSizeBaseOnScanResult(int *d_allPossibleExtension,int *d_allPossibleExtensionScanResult,unsigned int noElem_allPossibleExtension,int &noElem_d_UniqueExtension);