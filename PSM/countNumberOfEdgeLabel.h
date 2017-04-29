#pragma once;
#include "cuda_runtime.h"
#include <stdio.h>
#include <device_launch_parameters.h>
#include "kernelPrintf.h"
#include "kernelReduce.h"
#include "sumUntilReachZero.h"
#include "countNumberOfLabelVetex.h"

extern "C" inline cudaError_t countNumberOfEdgeLabel(int*,unsigned int,unsigned int&);