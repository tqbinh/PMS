#pragma once;
#ifndef __CUDACC__  
    #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include <device_functions.h>

extern "C" inline __global__ void kernelReduce(int*,int*,unsigned int);