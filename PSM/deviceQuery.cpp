/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

// Shared Utilities (QA Testing)

// std::system includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <map>
#include "conio.h"
#include <memory>
#include <iostream>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"

int *pArgc = NULL;
char **pArgv = NULL;

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}


inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}

inline bool IsAppBuiltAs64()
{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
   
}
