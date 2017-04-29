#pragma once;
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "checkArray.h"
#include <iostream>

//********kiểm tra đảm bảo dữ liệu ở GPU giống với Host********
extern "C" inline cudaError_t checkDataBetweenHostAndGPU(int *,int *,int *,int *,unsigned int,unsigned int,int*,int*,int*,int*,size_t,size_t,size_t,size_t);