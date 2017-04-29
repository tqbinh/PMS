#include "countNumberOfEdgeLabel.h"

cudaError_t countNumberOfEdgeLabel(int *d_LN,unsigned int sizeOfarrayLN,unsigned int &numberOfDifferentEdgeLabel){
	cudaError_t cudaStatus=	countNumberOfLabelVetex(d_LN,sizeOfarrayLN,numberOfDifferentEdgeLabel);
	if (cudaStatus!= cudaSuccess){
		fprintf(stderr,"countNumberOfLabelVetex in countNumberOfEdgeLabel fail",cudaStatus);
		goto Error;
	}
Error:
	return cudaStatus;

}
