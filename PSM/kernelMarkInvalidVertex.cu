#include "kernelMarkInvalidVertex.h"

__global__ void kernelMarkInvalidVertex(int *d_O,int *LO,unsigned int sizeLO,int *d_labelAmount,unsigned int sizeLabelAmount,unsigned int minsup){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<sizeLabelAmount){
		if(d_labelAmount[i]<minsup){
			for (int j=0;j<sizeLO;++j){
				if(LO[j]==i) d_O[j]=-1;
			}
		}
	}
}