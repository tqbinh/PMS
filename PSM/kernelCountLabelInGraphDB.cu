#include "kernelCountLabelInGraphDB.h"

__global__ void kernelCountLabelInGraphDB(int *LO,int *result,unsigned int sizeLO,unsigned int sizeResult){
	int i=blockIdx.x*blockDim.x + threadIdx.x;
	if(i<sizeResult){
		for(int j=0;j<sizeLO;++j){
			if(LO[j]==i) ++result[i];
		}

	}

}