#ifndef __HELPERS__
#define __HELPERS__

#include <iostream>
#include <string>

class Helpers
{
	std::string root;

public:
	Helpers() { root = "PTX_files";}
	std::string getPTXPath( std::string demo, char* cuda_file );
	
};

#endif