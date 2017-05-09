#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" inline cudaError_t getLastElement(int* inputArray,unsigned int numberElementOfInputArray,int &outputValue);