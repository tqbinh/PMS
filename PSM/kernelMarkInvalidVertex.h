#pragma once;
#include "device_launch_parameters.h"
#include <stdio.h>

inline __global__ void kernelMarkInvalidVertex(int*,int*,unsigned int,int*,unsigned int,unsigned int);
	