#pragma once;
#include "device_launch_parameters.h"
#include <stdio.h>

inline __global__ void kernelCountLabelInGraphDB(int*,int*,unsigned int,unsigned int);