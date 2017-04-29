#pragma once;
#include "device_launch_parameters.h"
#include "markInvalidVertex.h"
#include "cuda_runtime.h"
#include "kernelPrintf.h"
#include "kernelCountLabelInGraphDB.h"
#include <stdio.h>

extern "C" inline cudaError_t markInvalidVertex(int *,int *,int,unsigned int);

