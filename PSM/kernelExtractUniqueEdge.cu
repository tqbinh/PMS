#include "kernelExtractUniqueEdge.h"

__global__ void kernelExtractUniqueEdge(int *d_O,int *d_LO,unsigned int numberElementOfd_O,int *d_N,int *d_LN,unsigned int numberElementOfd_N,int *d_singlePattern,unsigned int Lv,unsigned int Le){

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i<numberElementOfd_O){
		/*printf("\nThread:%d",i);	*/
		if(d_O[i]!=-1){ 
			int j;
			//printf("\nThread:%d",i);	
			for(j=i+1;j<numberElementOfd_O;++j){					
				if(d_O[j]!=-1) {break;}
				
			}			
			int ek;
			if (j==numberElementOfd_O) {
				ek=numberElementOfd_N;

			}
			else
			{
				ek=d_O[j];
			}

			
			int Li=d_LO[i];				
			int startIndex=((Lv+(Lv-(Li-1)))*(Lv-(Lv-(Li-1))+1)/2)*Le;				
			for (int k=d_O[i];k<ek;++k){
				int Lj, Lij;					
				Lij=d_LN[k];					
				Lj=d_LO[d_N[k]]; 					
				if(Lj<Li) continue;
				startIndex=startIndex+Lij*(Lv-Li) + (Lj-Li);
				d_singlePattern[startIndex]=1;
				//printf("\nThread:%d Li:%d Lj:%d Le:%d  index:%d d_signlePattern:%d\n",i,Li,Lj,Le,startIndex,d_singlePattern[startIndex]);
				startIndex=startIndex-(Lj-Li);
				//printf("index:%d [%d] ",index,d_singlePattern[index]);
			}

		}
	}
}
