#include <iostream>
#include <string>
#include <map>
#include "conio.h"

#include "gspan.h"
//#include <unistd.h>

#define OPT " [-m minsup] [-d] [-e] [-w] "

using namespace std;

void usage(void)
{
	cout<<"usage: gspan "<<OPT<<" [-L maxpat]"<<endl;
}
int main (int argc, char * const  argv[]) {
	//usage();
	/*
	map<string,string>ass_array;
	
	ass_array["fat"]="a";
	ass_array["sodium"]="b";
	ass_array["totalcarb"]="c";
	ass_array["protein"]="d";
	
	ass_array.insert(map<string,string>::value_type("potassium","e"));
	
	cout << ass_array["potassium"] << endl;
	cout << "The map has " << ass_array.size() << " entries.\n";
	
	map<string,string>::iterator loc;
	//string loc;
	loc = ass_array.find("potassium");
	
	if(loc != ass_array.end())
		cout << loc->first <<":" << loc->second << "Sugar found!\n";
	else
		cout << "No Sugar\n";
	*/
	//unsigned int minsup = 34;
	unsigned int minsup = 2;
	unsigned int maxpat = 2;
	//unsigned int maxpat = 0x00000000;
	unsigned int minnodes = 0;
	bool where = true;
	bool enc = false;
	bool directed = false;
	
	int opt;
	/*
	while((opt = getopt(argc,argv,"edws::m:L:Dhn:")) != -1)
	{
		switch(opt){
				case 's':
				case 'm':
					minsup = atoi(optarg);
					break;
				case 'n':
					minsup = atoi(optarg);
					break;
				case 'L':
					maxpat = atoi(optarg);
					break;
				case 'd':
				case 'e':
					enc = true;
					break;
				case 'w':
					where = true;
					break;
				case 'D':
					directed = true;
					break;
				case 'h':
				default:
					usage();
					return -1;
		}
	}
*/
	char* fname;
	fname = "Chemical_340";
	
	gSpan gspan;	
	ofstream fout("result.txt");
    gspan.run(fname,fout,minsup,maxpat,minnodes,enc,where,directed);
	fout.close();
	return 0;
}
