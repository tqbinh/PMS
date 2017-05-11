#include "reduction.h"
#include "kernelPrintf.h"

#define funcCheck(stmt) {                                            \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
        printf( "Failed to run stmt %d ", __LINE__);                 \
        printf( "Got CUDA error ...  %s ", cudaGetErrorString(err)); \
        return cudaStatus;                                                   \
    }                                                                \
}

__global__  void total(float * input, float * output, int len) 
{
	// Load a segment of the input vector into shared memory
	__shared__ float partialSum[2*BLOCK_SIZE];
	int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockIdx.x*blockDim.x;

	if ((start + t) < len)
	{
		partialSum[t] = input[start + t];      
	}
	else
	{       
		partialSum[t] = 0.0;
	}
	if ((start + blockDim.x + t) < len)
	{   
		partialSum[blockDim.x + t] = input[start + blockDim.x + t];
	}
	else
	{
		partialSum[blockDim.x + t] = 0.0;
	}

	// Traverse reduction tree
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	__syncthreads();

	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId*2) < len)
	{
		output[blockIdx.x] = partialSum[t];
	}
}


cudaError_t reduction(float *deviceInput,int len,float &support){
	cudaError_t cudaStatus;
	
	
    float * deviceOutput;

	int numInputElements = len; // number of elements in the input list
	int numOutputElements; // number of elements in the output list

	numOutputElements = numInputElements / (BLOCK_SIZE<<1);
	if (numInputElements % (BLOCK_SIZE<<1)) 
	{
		numOutputElements++;
	}
		
    funcCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));

	// Initialize the grid and block dimensions here
    dim3 DimGrid( numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // Launch the GPU Kernel here
    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
	
	cudaDeviceSynchronize();
	cudaStatus=cudaGetLastError();
	if(cudaStatus!=cudaSuccess){
		fprintf(stderr,"\ncudaDeviceSynchronize() reduction failed");
		goto Error;
	}
	printf("\n");
	printFloat(deviceOutput,numOutputElements);
	cudaMemcpy(&support,deviceOutput,numOutputElements*sizeof(float),cudaMemcpyDeviceToHost);

Error:
	return cudaStatus;
}
