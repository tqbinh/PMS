#include "prescan.h"

//__global__ void prescan(int *g_odata,int *g_idata,unsigned int n){
//	extern __shared__ int temp[];  // allocated on invocation  
//	int thid = threadIdx.x;  
//	int offset = 1;
//	temp[2*thid] = g_idata[2*thid]; // load input into shared memory  
//	temp[2*thid+1] = g_idata[2*thid+1];  
//	for (int d = n>>1; d > 0; d >>= 1)     // build sum in place up the tree  
//	{   
//		__syncthreads();  
//		if (thid < d)  
//		{  
//			int ai = offset*(2*thid+1)-1;  
//			int bi = offset*(2*thid+2)-1;  
//			temp[bi] += temp[ai];  
//		}  
//		offset *= 2; 
//	}		
//		if (thid == 0) { temp[n - 1] = 0; } // clear the last element  
//	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
//	{  
//		offset >>= 1;  
//		__syncthreads();  
//		if (thid < d)                       
//		{  
//			int ai = offset*(2*thid+1)-1;  
//			int bi = offset*(2*thid+2)-1;  
//			float t = temp[ai];  
//			temp[ai] = temp[bi];  
//			temp[bi] += t;   
//		}  
//	}  
//	__syncthreads(); 
//	g_odata[2*thid] = temp[2*thid]; // write results to device memory  
//	g_odata[2*thid+1] = temp[2*thid+1];  
//
//}

__global__ void scan_bel(int* inputarray,int loop,int* outputarray,int number)
{
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

	int divisor = 2;
	int adder = 1;
	int temp;

	for(int i=0;i<loop;i++)
	{
		if(thIdx%(divisor) == divisor-1)
		{
			outputarray[thIdx] = outputarray[thIdx-adder]+outputarray[thIdx];
		}
		__syncthreads();
		divisor*=2;
		adder*=2;
	}

	divisor = number;
	adder = divisor/2;

	outputarray[number-1] = 0;
	for(int i=0;i<loop;i++)
	{
		if(thIdx%(divisor) == divisor-1)
		{
			temp = outputarray[thIdx];
			outputarray[thIdx] = outputarray[thIdx-adder]+outputarray[thIdx];
			outputarray[thIdx-adder] = temp;
		}
		__syncthreads();
		divisor/=2;
		adder/=2;
	}
}