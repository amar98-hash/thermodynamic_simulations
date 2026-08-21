#include "helper.h"

#include <exception>
#include <iostream>

int main(int argc, char* argv[])
{
    const std::string fileName =
        (argc > 1) ? argv[1] : "heatpipe-parameters.txt";

    try
    {
        Parameters p;

        readKeyValues(fileName, p);
        printKeyValues(p);
        const std::vector<double> T = readTValues(fileName);
        const std::vector<double> P = readPValues(fileName);

        
        printTValues(T);
        printPValues(P);


        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

}
