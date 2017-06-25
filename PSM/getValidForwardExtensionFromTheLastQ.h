#pragma once
class cHistory; //khai báo class history rỗng để hệ thống hiểu cHistory. Không hiểu tại sao lại như vậy?
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>
#include "Embedding.h"
#include "getExtension.h"
using namespace std;



extern "C" inline __global__ void kernelFindValidForwardFromLastQ(struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,int *d_O,int *d_LO,int *d_N,struct_V *d_arr_V,float *d_arr_degreeOfVerticesInQColumn,int maxOfVer,int m);
extern "C" inline __global__ void find_maximum_kernel(float *array, float *max, int *mutex, unsigned int n);
extern "C" inline __global__ void kernelFindDegreeOfVertex(int *d_O,int *d_N,int numberOfElementd_O,int numberOfElementd_N,struct_Q *device_arr_Q,int indexOfQ,int n,float *d_arr_degreeOfVerticesInQColumn,int maxOfVer);
extern "C" inline cudaError_t getValidForwardExtensionFromTheLastQ(struct_Q *device_arr_Q,int indexOfQ,cHistory **dH,int n,unsigned int maxOfVer,int *d_O,int *d_LO,int *d_N,int *d_LN,int numberOfElementd_O,int numberOfElementd_N);