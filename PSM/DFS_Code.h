#pragma once

#include<iostream>
#include<map>
#include<vector>
#include<set>
#include<algorithm>
#include<fstream>


using namespace std;



struct struct_Edge{ // Cấu trúc của một cạnh 
	int from; //từ đỉnh nào
	int to;	//đến đỉnh nào
	int elabel;	//nhãn cạnh 
	unsigned int id;	//mỗi cạnh đều có id duy nhất
	struct_Edge(): from(0),to(0),elabel(0),id(0) {}; //khởi tạo cho cạnh 
};

class clssVertex
{
public:
	typedef std::vector<struct_Edge>::iterator edge_iterator;	//định nghĩa tên kiểu dữ liệu mới ngắn gọn hơn www.stdio.vn/articles/read/165/typedef-va-enum
	
	int label; // nhãn đỉnh	
	std::vector<struct_Edge> edge;  //mỗi phần tử của vector là một edge, ở đây cấu trúc edge là một thành phần của class Vertex
	
	void push(int from,int to, int elabel) // thêm một cạnh vào vector edge
	{
		edge.resize(edge.size()+1);		//phải tăng kích thước của edge lên 1
		edge[edge.size()-1].from = from;	//thông tin cạnh từ đỉnh nào
		edge[edge.size()-1].to = to;		//đến đỉnh nào
		edge[edge.size()-1].elabel = elabel;	//nhãn cạnh là gì
		return;		
	}
};


class clssGraph:public std::vector<clssVertex>{ //Lớp Graph kế thừa lớp vector<Vertex>, tức khi bạn tạo một thể hiện Graph thì hệ thống cũng tạo ra một vector Graph
private:
	unsigned int edge_size_; //đồ thị có bao nhiêu cạnh	
public:
	
	typedef std::vector<clssVertex>::iterator vertex_iterator; //định nghĩa kiểu dữ liệu iterator cho Vertex
	
	clssGraph(bool _directed)	//Đồ thị có hướng hay vô hướng.
	{
		directed = _directed;
	};
	bool directed;	//chỉ hướng của đồ thị
	
	unsigned int edge_size(){ return edge_size_;} //trả về số cạnh của đồ thị
	unsigned int vertex_size() { return (unsigned int)size(); } //wrapper function, trả về số đỉnh của đồ thị
	
	
	void buildEdge(); //khai báo hàm xây dựng các cạnh cho đồ thị
	int read (char*,int); //khai báo hàm đọc 
	std::ostream& write (std::ostream&); //khai báo hàm chỉ ghi
	std::ofstream& write(std::ofstream&); //khai báo hàm đọc và ghi file
	void check(void); //khai báo hàm kiểm tra
	
	clssGraph():edge_size_(0),directed(false){}; //khởi tạo đồ thị với cạnh và mặc định là vô hướng
};


class clssDFS{	//Lớp DFS, biểu diễn DFS code của một cạnh. Gồm các cả overloading operator dùng để so sánh DFS code của 2 cạnh.
public:
	int from; //từ đỉnh nào
	int to; //đến đỉnh nào
	int fromlabel; //từ nhãn đỉnh nào
	int elabel;	//nhãn cạnh
	int tolabel; //đến nhãn đỉnh nào
	//overloading operator == để so sánh giữa 2 DFS code d1 và d2
	friend bool operator == (const clssDFS& d1,const clssDFS& d2){
		return (d1.from == d2.from && d1.to == d2.to && d1.fromlabel == d2.fromlabel && d1.elabel == d2.elabel && d1.tolabel == d2.tolabel);
	}
	//overloading operator != để so sánh giữa 2 DFS code d1 và d2
	friend bool operator != (const clssDFS& d1,const clssDFS& d2){
		return (!(d1==d2));
	}
	//Khởi tạo DFS code
	clssDFS():from(0),to(0),fromlabel(0),elabel(0),tolabel(0){};
};


typedef std::vector<int> RightMostPath; //định nghĩa một kiểu vector<int> có tên là RMPath dùng để chứa Right Most Path
//cấu trúc của DFS code
struct struct_DFSCode:public std::vector<clssDFS>{ //kế thừa từ vector<DFS>
private:
	RightMostPath rmpath;  //right most path là một vector đã định nghĩa ở trên
public:
	const RightMostPath& buildRMPath(); //xây dựng right most path tức là tìm nhánh phải nhất của DFSCode
	
	bool toGraph(clssGraph&); //*0* không biết để làm gì; có thể là để biết GID của graph
	void fromGraph(clssGraph& g); //*0* không biết để làm gì
	
	unsigned int nodeCount(void); //đếm số node của DFSCode
	void push(int from,int to,int fromlabel,int elabel,int tolabel){ //thêm DFS vào DFSCode
		resize(size()+1); //vì DFSCode kế thừa từ vector<DFS> nên DFSCode có thể dùng phương thức size() của vector<DFS>
		clssDFS& d = (*this)[size()-1]; //DFSCode chẳng qua là vector<DFS>: đây là ý nghĩa của kế thừa
		
		d.from = from;
		d.to = to;
		d.fromlabel = fromlabel;
		d.elabel = elabel;
		d.tolabel = tolabel;
	}
	
	void pop(){ resize (size()-1);} //lấy phần từ cuối ra
	std::ostream& write(std::ostream &); //*0* Hàm ghi nhưng không biết là để ghi cái gì; chắc là ghi cập nhật DFS code mới xuống tập tin.
};
