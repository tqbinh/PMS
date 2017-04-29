#include "sumUntilReachZero.h"

void sumUntilReachZero(int *h_Lv,unsigned int n,int &result){
	for(int i=0;i<n && h_Lv[i]!=0;++i){
		++result;
	}

}
