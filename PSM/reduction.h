#pragma once
//Code is customized from this link http://www.techdarting.com/2014/06/parallel-reduction-in-cuda.html
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 512 // You can change this
#define NUM_OF_ELEMS 4096 // You can change this



extern "C" inline cudaError_t reduction(float *input,int len,float &support);