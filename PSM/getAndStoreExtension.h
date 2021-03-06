#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"
#include "kernelPrintf.h"

#define blocksize 512



extern "C" inline cudaError_t getAndStoreExtension(Extension*,int*,int*,unsigned int,int*,int*,unsigned int,unsigned int,unsigned int);