#include <iostream>
#include <iomanip>


int main(int argc, char** argv)
try {
	std::cout << "Hello" << std::endl;

	return EXIT_SUCCESS;

}
catch (const char* msg) {
	std::cerr << msg << std::endl;
	return EXIT_FAILURE;
}
catch (...) {
	std::cerr << "Caught unknown exception" << std::endl;
	return EXIT_FAILURE;
}
