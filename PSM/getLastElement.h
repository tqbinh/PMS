#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"

extern "C" inline cudaError_t getLastElement(int* inputArray,unsigned int numberElementOfInputArray,int &outputValue);
extern "C" inline cudaError_t getLastElementExtension(Extension* inputArray,unsigned int numberElementOfInputArray,int &outputValue,unsigned int maxOfVer);