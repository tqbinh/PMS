#pragma once;
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernelaccess_d_LO_from_idx_of_d_O.h"


extern "C" inline cudaError_t access_d_LO_from_idx_of_d_O(int*,int*,int);
