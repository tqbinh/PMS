#pragma once;
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_runtime.h"

extern "C" inline __global__ void kernelPrintf(int*,int sizeO);
extern "C" inline __global__ void kernelPrintFloat(float* array,int numberElementOfArray);