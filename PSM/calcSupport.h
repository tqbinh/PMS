#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"

extern "C" inline cudaError_t calcSupport(Extension *d_UniqueExtension,unsigned int noElem_d_UniqueExtension,Extension *d_ValidExtension,unsigned int noElem_d_ValidExtension,int *d_scanB_Result,int *d_F,unsigned int noElem_F);