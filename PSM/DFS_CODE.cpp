#pragma once
#include "DFS_Code.h"


#include <cstring>
#include <string>
#include <iterator>
#include <strstream>
#include <set>
#include <fstream>
using namespace std;
template <typename T,typename I>
void tokenize(const char* str,I iterator)
{
	std::istrstream is (str,std::strlen(str));
	std::copy (std::istream_iterator <T> (is), std::istream_iterator <T> (), iterator);
}

bool struct_DFSCode::toGraph(clssGraph& g) //Convert DFSCode sang đồ thị.
{
	g.clear(); 	
	for(struct_DFSCode::iterator it = begin();it != end(); ++it){ 
		g.resize(std::max (it->from,it->to) +1); 
		
		if(it->fromlabel != -1) g[it->from].label = it->fromlabel; 
		if(it->tolabel != -1)
			g[it->to].label = it->tolabel;
		g[it->from].push (it->from,it->to,it->elabel);
		if(g.directed == false)
			g[it->to].push (it->to,it->from,it->elabel);
	}
	
	g.buildEdge();
	
	return (true);
}


void clssGraph::buildEdge() //Hàm này vừa xây dựng ánh xạ tmp<string,int> (string:from-to-elabel, int:id_edge) và cập nhật id của cạnh và số lượng cạnh của một đồ thị
{
	char buf[512];	//khai báo vùng đệm có kích thước 512 Bytes
	std::map <std::string,unsigned int> tmp;	//tmp là một ánh xạ map <string, unsigned int>. Ở đây mượn ánh xạ tmp để cập nhật edge_id của vector<edge> cho nhanh.
	
	unsigned int id=0;
	for(int from=0;from<(int) size();++from) { //duyệt qua các đỉnh trong đồ thị, tương ứng với mỗi đỉnh, sẽ duyệt qua các cạnh của nó, và set id cho cạnh. Id của cạnh tính từ 0.
		for(clssVertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it) //it là một edge_iterator dùng để duyệt qua các phần tử trong vector<edge>
		{
			if(directed||from<=it->to) //nếu là đồ thị vô hướng hoặc id từ đỉnh đến 'from' nhỏ hơn id của 'to'
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel); //thì bỏ nó vào bộ nhớ đệm buf theo thứ tự from - to - elabel
			else
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel); //ngược lại, nếu là đồ thị có hướng thì bỏ vào buf theo thứ tự 'to'-'from'-'elabel'
			
			if(tmp.find(buf) == tmp.end()){ //kiểm tra xem cạnh của đỉnh đang xét đã tồn tại trong ánh xạ tmp hay chưa? Nếu chưa tồn tại thì thêm vào.
				it->id = id; //cập nhật id cho cạnh
				tmp[buf]=id; //thêm cạnh đó vào ánh xạ.
				++id; //tăng id lên 1
			}else{
				it->id = tmp[buf]; //ngược lại, nếu cạnh đó đã được đưa vào ánh xạ rồi thì cập nhật lại id cho cạnh là đủ. Ở đây buf là key của ánh xạ tmp, nên tmp[buf] sẽ trả về id của tmp.
			}
			
		}
	}
	
	edge_size_=id; //Số lượng cạnh của đồ thị
}


std::ostream& clssGraph::write(std::ostream& os) //ghi kết quả xuống file result.txt
{
	char buf[512];
	std::set <std::string> tmp;
	
	for(int from = 0; from < (int)size();++from){ //duyệt qua các đỉnh của đồ thị g
		os<<"v "<< from << " " << (*this)[from].label << std::endl; //ghi đỉnh đó vào file result.txt
		
		for(clssVertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			if(directed||from <= it->to){ //nếu là đồ thị vô hướng hoặc from<=to
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel);
			}else{
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel);
			}
			tmp.insert(buf); //chèn cạnh đó vào tập hợp (set) tmp
		}
	}
	
	for(std::set<std::string>::iterator it = tmp.begin();it!=tmp.end();++it){ //duyệt qua tất cả các cạnh trong set tmp
		os << "e " << *it << std::endl; //ghi cạnh đó vào tập tin result.txt
	}
	
	return os;
}

std::ofstream& clssGraph::write(std::ofstream& os)
{
	char buf[512];
	std::set <std::string> tmp;
	
	for(int from = 0; from < (int)size();++from){
		os<<"v "<< from << " " << (*this)[from].label << std::endl;
		
		for(clssVertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			if(directed||from <= it->to){
				std::sprintf(buf,"%d %d %d",from,it->to,it->elabel);
			}else{
				std::sprintf(buf,"%d %d %d",it->to,from,it->elabel);
			}
			tmp.insert(buf);
		}
	}
	
	for(std::set<std::string>::iterator it = tmp.begin();it!=tmp.end();++it){
		os << "e " << *it << std::endl;
	}
	
	return os;
}

void clssGraph::check(void)
{
	for(int from = 0; from < (int)size();++from){
		std::cout << "check vertex " << from << ",label " << (*this)[from].label << std::endl;
		for(clssVertex::edge_iterator it = (*this)[from].edge.begin();it!=(*this)[from].edge.end();++it)
		{
			std::cout << "check edge from " << it->from << " to " << it->to << ", label " << it->elabel << std::endl; 
		}
	}
}