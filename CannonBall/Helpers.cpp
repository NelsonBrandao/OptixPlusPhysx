#include "Helpers.h"

std::string Helpers::getPTXPath( std::string demo, char* cuda_file ){
	std::string path;
	path.append(root).append( "/" ).append( demo ).append( "_" ).append( cuda_file ).append( ".ptx" );
	std::cout << path.c_str() << std::endl;
	return path;
}
