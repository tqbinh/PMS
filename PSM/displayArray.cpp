#include "displayArray.h"

void displayArray(int *p, const unsigned int pSize=0)
{
	for(int i=0;i<pSize;i++){
		printf("P[%d]:%d ",i,p[i]);
	}
	printf("\n");
	return;
}

