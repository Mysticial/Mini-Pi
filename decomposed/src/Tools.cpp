/* Tools.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 */

//  Visual Studio 2010 doesn't have <chrono>.
#if defined(_MSC_VER) && (_MSC_VER <= 1600)
#define USE_CHRONO 0
#else
#define USE_CHRONO 1
#endif

#ifdef _MSC_VER
#pragma warning(disable:4996)   //  fopen() deprecation
#endif

#if USE_CHRONO
#include <chrono>
#else
#include <time.h>
#endif

#include "Tools.h"

namespace Mini_Pi{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
double wall_clock(){
    //  Get the clock in seconds.
#if USE_CHRONO
    auto ratio_object = std::chrono::high_resolution_clock::period();
    double ratio = (double)ratio_object.num / ratio_object.den;
    return std::chrono::high_resolution_clock::now().time_since_epoch().count() * ratio;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
void dump_to_file(const char *path, const std::string &str){
    //  Dump a string to a file.

    FILE *file = fopen(path, "wb");
    if (file == NULL)
        throw "Cannot Create File";

    fwrite(str.c_str(), 1, str.size(), file);
    fclose(file);
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
