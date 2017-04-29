#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_functions_decls.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
using namespace std;

extern "C" inline __global__ void kernelCastingInt2Float(float*,int*,unsigned int);
extern "C" inline __global__ void kernelCastingFloat2Int(int*,float*,unsigned int);