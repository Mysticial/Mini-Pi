/* DebugPrinting.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 */

#include <iostream>
#include "DebugPrinting.h"
namespace Mini_Pi{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
uint32_t rand_word(){
    return (uint32_t)(
            (rand() & 0xf) <<  0 |
            (rand() & 0xf) <<  8 |
            (rand() & 0xf) << 16 |
            (rand() & 0xf) << 24
        ) % 1000000000;
}
complex<double> rand_complex(){
    double r = (double)(rand() % 1000);
    double i = (double)(rand() % 1000);
    return complex<double>(r, i);
}
void print_fft(complex<double> *T, int k){
    int length = (size_t)1 << k;
    for (int c = 0; c < length; c++){
        std::cout << T[c].real() << " + " << T[c].imag() << "i" << " , ";
    }
    std::cout << std::endl;
}
void print_word(uint32_t word){
    char str[] = "012345678";
    for (int c = 8; c >= 0; c--){
        str[c] = word % 10 + '0';
        word /= 10;
    }
    std::cout << str;
}
void print_words(uint32_t *T, size_t L){
    while (L-- > 0){
        print_word(T[L]);
    }
    std::cout << std::endl;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
