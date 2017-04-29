#pragma once;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "prescan.h"

extern "C" inline __global__ void kernelExtractUniqueEdge(int*,int*,unsigned int,int*,int*,unsigned int,int*,unsigned int,unsigned int);