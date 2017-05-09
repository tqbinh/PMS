#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "scan_largearray_kernel.h"
#include "castingIntToFloat.h"
using namespace std;

extern "C" inline cudaError_t scanV(int *inputArray,unsigned int numberElementOfInputArray,int *outputArray);