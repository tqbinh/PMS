#pragma once
class cHistory; //khai báo class history rỗng để hệ thống hiểu cHistory. Không hiểu tại sao lại như vậy?
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "Embedding.h"
#include "getExtension.h"
using namespace std;

extern "C"  cudaError_t markEmbedding(cHistory **dH,struct_Q *device_arr_Q,int lastColumn,vector<int> RMPath,int n,unsigned int maxOfVer,int *d_O,int *d_N);