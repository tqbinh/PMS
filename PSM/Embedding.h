#pragma once

#include <iostream>
#include <stdlib.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\device_ptr.h>
using namespace std;

//cấu trúc Embedding
struct struct_Embedding
{
	int prevQ;
	int idx;
	int vid;	
	struct_Embedding():idx(-1),vid(-1){};
};

//cấu trúc struct_Q
struct struct_Q
{
	int _prevQ;
	int _size;
	struct_Embedding *_d_arr_Q;
	

};

struct struct_QQ
{
	int size;
	struct_Q *structQ;
};


 // Template structure to pass to kernel
template < typename T >
struct KernelArray
{    
	T*  _array;    
	int _size;
}; 