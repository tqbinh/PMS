#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ExtensionStructure.h"
#include <stdio.h>

#define blocksize 512

extern "C" inline cudaError_t getSatisfyEdge(Extension *d_UniqueExtension,int noElem_d_UniqueExtension,int indexOfSatisfyEdge,int &li,int &lij,int &lj,int *&d_arr_edgeLabel);