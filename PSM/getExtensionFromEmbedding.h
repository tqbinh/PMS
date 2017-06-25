#pragma once
class cHistory; //khai báo class history rỗng để hệ thống hiểu cHistory. Không hiểu tại sao lại như vậy?
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "Embedding.h"
#include "getExtension.h"
#include "getValidForwardExtensionFromTheLastQ.h"
using namespace std;

extern "C" inline cudaError_t getValidExtensionFromEmbeding(struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,unsigned int maxOfVer,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N,int lastColumn);