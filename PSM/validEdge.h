#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"
using namespace std;

extern "C" inline cudaError_t validEdge(Extension *,int *,unsigned int);