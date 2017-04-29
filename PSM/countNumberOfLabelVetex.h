#pragma once;
#include "cuda_runtime.h"
#include <stdio.h>
#include <device_launch_parameters.h>
#include "kernelPrintf.h"
#include "kernelReduce.h"
#include "sumUntilReachZero.h"
extern "C" inline __global__ void kernelCountNumberOfLabelVertex(int*,int*,unsigned int);
extern "C" inline cudaError_t countNumberOfLabelVetex(int*,unsigned int,unsigned int&);
