#pragma once;
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernelPrintf.h"
#include "kernelExtractUniqueEdge.h"
#include <math.h>
#include "scan_largearray_kernel.h"
#include "castingIntToFloat.h"

extern "C" inline cudaError_t extractUniqueEdge(int *,int *,unsigned int ,int *,int *, unsigned int,int *,unsigned int,unsigned int,unsigned int);
