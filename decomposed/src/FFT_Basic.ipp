/* FFT_Basic.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 *      A basic FFT multiply implementation. No optimizations are done.
 * 
 */

#include <memory>
#include <complex>
#include "FFT.h"

namespace Mini_Pi{
    using std::complex;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
const double PI = 3.14159265358979323846;
void fft_forward(complex<double> *T, int k){
    //  Fast Fourier Transform
    //  This function performs a forward FFT of length 2^k.

    //  This is a Decimation-in-Frequency (DIF) FFT.
    //  The frequency domain output is in bit-reversed order.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    if (k == 0)
        return;

    size_t length = (size_t)1 << k;
    size_t half_length = length / 2;

    double omega = 2 * PI / length;

    //  Perform FFT reduction into two halves.
    for (size_t c = 0; c < half_length; c++){
        //  Generate Twiddle Factor
        double angle = omega * c;
        auto twiddle_factor = complex<double>(cos(angle), sin(angle));

        //  Grab elements
        complex<double> a = T[c];
        complex<double> b = T[c + half_length];

        //  Perform butterfly
        T[c              ]  = a + b;
        T[c + half_length]  = (a - b) * twiddle_factor;
    }

    //  Recursively perform FFT on lower elements.
    fft_forward(T, k - 1);

    //  Recursively perform FFT on upper elements.
    fft_forward(T + half_length, k - 1);
}
void fft_inverse(complex<double> *T, int k){
    //  Fast Fourier Transform
    //  This function performs an inverse FFT of length 2^k.

    //  This is a Decimation-in-Time (DIT) FFT.
    //  The frequency domain input must be in bit-reversed order.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    if (k == 0)
        return;

    size_t length = (size_t)1 << k;
    size_t half_length = length / 2;

    double omega = -2 * PI / length;

    //  Recursively perform FFT on lower elements.
    fft_inverse(T, k - 1);

    //  Recursively perform FFT on upper elements.
    fft_inverse(T + half_length, k - 1);

    //  Perform FFT reduction into two halves.
    for (size_t c = 0; c < half_length; c++){
        //  Generate Twiddle Factor
        double angle = omega * c;
        auto twiddle_factor = complex<double>(cos(angle), sin(angle));

        //  Grab elements
        complex<double> a = T[c];
        complex<double> b = T[c + half_length] * twiddle_factor;

        //  Perform butterfly
        T[c              ]  = a + b;
        T[c + half_length]  = a - b;
    }
}
void fft_pointwise(complex<double> *T, const complex<double> *A, int k){
    //  Performs pointwise multiplications of two FFT arrays.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    size_t length = (size_t)1 << k;
    for (size_t c = 0; c < length; c++){
        T[c] = T[c] * A[c];
    }
}
void int_to_fft(complex<double> *T, int k, const uint32_t *A, size_t AL){
    //  Convert word array into FFT array. Put 3 decimal digits per complex point.

    //Parameters:
    //  -   T   -   FFT array
    //  -   k   -   2^k is the size of the transform
    //  -   A   -   word array
    //  -   AL  -   length of word array

    size_t fft_length = (size_t)1 << k;
    complex<double> *Tstop = T + fft_length;

    //  Since there are 9 digits per word and we want to put 3 digits per
    //  point, the length of the transform must be at least 3 times the word
    //  length of the input.
    if (fft_length < 3*AL)
        throw "FFT length is too small.";

    //  Convert
    for (size_t c = 0; c < AL; c++){
        uint32_t word = A[c];

        *T++ = word % 1000;
        word /= 1000;
        *T++ = word % 1000;
        word /= 1000;
        *T++ = word;
    }

    //  Pad the rest with zeros.
    while (T < Tstop)
        *T++ = complex<double>(0, 0);
}
void fft_to_int(const complex<double> *T, int k, uint32_t *A, size_t AL){
    //  Convert FFT array back to word array. Perform rounding and carryout.

    //Parameters:
    //  -   T   -   FFT array
    //  -   A   -   word array
    //  -   AL  -   length of word array

    //  Compute Scaling Factor
    size_t fft_length = (size_t)1 << k;
    double scale = 1. / fft_length;

    //  Since there are 9 digits per word and we want to put 3 digits per
    //  point, the length of the transform must be at least 3 times the word
    //  length of the input.
    if (fft_length < 3*AL)
        throw "FFT length is too small.";

    //  Round and carry out.
    uint64_t carry = 0;
    for (size_t c = 0; c < AL; c++){
        double   f_point;
        uint64_t i_point;
        uint32_t word;

        f_point = (*T++).real() * scale;        //  Load and scale
        i_point = (uint64_t)(f_point + 0.5);    //  Round
        carry += i_point;                       //  Add to carry
        word = carry % 1000;                    //  Get 3 digits.
        carry /= 1000;

        f_point = (*T++).real() * scale;        //  Load and scale
        i_point = (uint64_t)(f_point + 0.5);    //  Round
        carry += i_point;                       //  Add to carry
        word += (carry % 1000) * 1000;          //  Get 3 digits.
        carry /= 1000;

        f_point = (*T++).real() * scale;        //  Load and scale
        i_point = (uint64_t)(f_point + 0.5);    //  Round
        carry += i_point;                       //  Add to carry
        word += (carry % 1000) * 1000000;       //  Get 3 digits.
        carry /= 1000;

        A[c] = word;
    }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void ensure_FFT_tables(size_t CL){
    //  No-op.
}
void multiply_FFT(uint32_t* C, const uint32_t* A, size_t AL, const uint32_t* B, size_t BL){
    size_t CL = AL + BL;

    //  Determine minimum FFT size.
    int k = 0;
    size_t length = 1;
    while (length < 3*CL){
        length <<= 1;
        k++;
    }

    //  Allocate FFT arrays
    auto Ta = std::unique_ptr<complex<double>[]>(new complex<double>[length]);
    auto Tb = std::unique_ptr<complex<double>[]>(new complex<double>[length]);

    //  Perform a convolution using FFT.
    //  Yeah, this is slow for small sizes, but it's asympotically optimal.

    //  3 digits per point is small enough to not encounter round-off error
    //  until a transform size of 2^30.
    //  A transform length of 2^29 allows for the maximum product size to be
    //  2^29 * 3 = 1, 610, 612, 736 decimal digits.
    if (k > 29)
        throw "FFT size limit exceeded.";

    int_to_fft(Ta.get(), k, A, AL);        //  Convert 1st operand
    int_to_fft(Tb.get(), k, B, BL);        //  Convert 2nd operand
    fft_forward(Ta.get(), k);            //  Transform 1st operand
    fft_forward(Tb.get(), k);            //  Transform 2nd operand
    fft_pointwise(Ta.get(), Tb.get(), k); //  Pointwise multiply
    fft_inverse(Ta.get(), k);            //  Perform inverse transform.
    fft_to_int(Ta.get(), k, C, CL);        //  Convert back to word array.
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
