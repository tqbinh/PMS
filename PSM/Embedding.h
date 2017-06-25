#pragma once

#include <iostream>
#include <stdlib.h>
//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>
//#include <thrust\device_ptr.h>
using namespace std;


//cấu trúc một phần tử của d_arr_V
struct struct_V
{
	int valid;
	int backward;
	struct_V():valid(0),backward(0){};
};

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

struct struct_DFS
{
	int from;
	int to;
	int li;
	int lij;
	int lj;
	struct_DFS():from(-1),to(-1),li(-1),lij(-1),lj(-1){};
};

//struct struct_QQ
//{
//	int size;
//	struct_Q *structQ;
//};
//
//
// // Template structure to pass to kernel
//template < typename T >
//struct KernelArray
//{    
//	T*  _array;    
//	int _size;
//}; 