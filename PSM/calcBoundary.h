#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"

extern "C" inline cudaError_t calcBoundary(Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_B,unsigned int Lv);