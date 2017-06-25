//@BinhTruong
#pragma once

#include <iostream>
#include <stdlib.h>

using namespace std;

struct Extension
{
	int vi,vj,li,lij,lj; //DFS_code của cạnh mở rộng
	int vgi,vgj; //global id của đỉnh
	Extension():vi(0),vj(0),li(0),lij(0),lj(0),vgi(0),vgj(0){};//khởi tạo cấu trúc
};
