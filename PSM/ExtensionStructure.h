//@BinhTruong
#pragma once

#include <iostream>
#include <stdlib.h>
#include "Embedding.h"

using namespace std;

struct Extension
{
	int vi,vj,li,lij,lj; //DFS_code của cạnh mở rộng
	int vgi,vgj; //global id của đỉnh
	//struct_Embedding *d_rowpointer;//lưu trữ pointer trỏ đến embedding mà nó mở rộng.
	Extension():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0){};//khởi tạo cấu trúc
};

//Cấu trúc này lưu trữ các cạnh phân biệt dựa vào nhãn của chúng
struct UniEdge
{	
	int li;
	int lij;
	int lj;
	UniEdge():li(-1),lij(-1),lj(-1){};
};
