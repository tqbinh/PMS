#include "checkArray.h"

bool checkArray(int *hostRef, int *gpuRef, const int N) {
	bool result=true;
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; i++) {
		if ((float)(abs(hostRef[i] - gpuRef[i])) > epsilon) {
			match = 0;
			result=false;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n",
				hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match){
		printf("Arrays match.\n\n");		
	}
	
	return result;
}