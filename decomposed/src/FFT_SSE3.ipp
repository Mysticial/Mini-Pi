/* FFT_CachedTwiddles.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 *      FFT multiply with cached twiddle factors and SSE3.
 * 
 */

#include <memory>
#include <vector>
#include <complex>
#include "FFT.h"

namespace Mini_Pi{
    using std::complex;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
const double PI = 3.14159265358979323846;
struct SIMD_delete{
    void operator()(__m128d *p){
        _mm_free(p);
    }
};
struct my_complex{
    double r;
    double i;
    my_complex(double _r, double _i) : r(_r), i(_i) {};
};
std::vector<std::vector<my_complex>> twiddle_table;
void fft_ensure_table(int k){
    //  Makes sure the twiddle factor table is large enough to handle an FFT of
    //  size 2^k.

    int current_k = (int)twiddle_table.size() - 1;
    if (current_k >= k)
        return;

    //  Do one level at a time
    if (k - 1 > current_k){
        fft_ensure_table(k - 1);
    }

    size_t length = (size_t)1 << k;
    double omega = 2 * PI / length;
    length /= 2;

    //  Build the sub-table.
    std::vector<my_complex> sub_table;
    for (size_t c = 0; c < length; c++){
        //  Generate Twiddle Factor
        double angle = omega * c;
        auto twiddle_factor = my_complex(cos(angle), sin(angle));
        sub_table.push_back(twiddle_factor);
    }

    //  Push into main table.
    twiddle_table.push_back(std::move(sub_table));
}
void fft_forward(__m128d *T, int k){
    //  Fast Fourier Transform
    //  This function performs a forward FFT of length 2^k.

    //  This is a Decimation-in-Frequency (DIF) FFT.
    //  The frequency domain output is in bit-reversed order.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    //  End recursion at 2 points.
    if (k == 1){
        __m128d a = T[0];
        __m128d b = T[1];
        T[0] = _mm_add_pd(a, b);
        T[1] = _mm_sub_pd(a, b);
        return;
    }

    size_t length = (size_t)1 << k;
    size_t half_length = length / 2;

    //  Get local twiddle table.
    std::vector<my_complex> &local_table = twiddle_table[k];

    //  Perform FFT reduction into two halves.
    for (size_t c = 0; c < half_length; c++){
        //  Grab Twiddle Factor
        __m128d r0 = _mm_loaddup_pd(&local_table[c].r);
        __m128d i0 = _mm_loaddup_pd(&local_table[c].i);

        //  Grab elements
        __m128d a0 = T[c];
        __m128d b0 = T[c + half_length];

        //  Perform butterfly
        __m128d c0, d0;
        c0 = _mm_add_pd(a0, b0);
        d0 = _mm_sub_pd(a0, b0);

        T[c] = c0;

        //  Multiply by twiddle factor.
        c0 = _mm_mul_pd(d0, r0);
        d0 = _mm_mul_pd(_mm_shuffle_pd(d0, d0, 1), i0);
        c0 = _mm_addsub_pd(c0, d0);

        T[c + half_length] = c0;
    }

    //  Recursively perform FFT on lower elements.
    fft_forward(T, k - 1);

    //  Recursively perform FFT on upper elements.
    fft_forward(T + half_length, k - 1);
}
void fft_inverse(__m128d *T, int k){
    //  Fast Fourier Transform
    //  This function performs an inverse FFT of length 2^k.

    //  This is a Decimation-in-Time (DIT) FFT.
    //  The frequency domain input must be in bit-reversed order.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    //  End recursion at 2 points.
    if (k == 1){
        __m128d a = T[0];
        __m128d b = T[1];
        T[0] = _mm_add_pd(a, b);
        T[1] = _mm_sub_pd(a, b);
        return;
    }

    size_t length = (size_t)1 << k;
    size_t half_length = length / 2;

    //  Recursively perform FFT on lower elements.
    fft_inverse(T, k - 1);

    //  Recursively perform FFT on upper elements.
    fft_inverse(T + half_length, k - 1);

    //  Get local twiddle table.
    std::vector<my_complex> &local_table = twiddle_table[k];

    //  Perform FFT reduction into two halves.
    for (size_t c = 0; c < half_length; c++){
        //  Grab Twiddle Factor
        __m128d r0 = _mm_loaddup_pd(&local_table[c].r);
        __m128d i0 = _mm_loaddup_pd(&local_table[c].i);
        i0 = _mm_xor_pd(i0, _mm_set1_pd(-0.0));

        //  Grab elements
        __m128d a0 = T[c];
        __m128d b0 = T[c + half_length];

        //  Perform butterfly
        __m128d c0, d0;

        //  Multiply by twiddle factor.
        c0 = _mm_mul_pd(b0, r0);
        d0 = _mm_mul_pd(_mm_shuffle_pd(b0, b0, 1), i0);
        c0 = _mm_addsub_pd(c0, d0);

        b0 = _mm_add_pd(a0, c0);
        d0 = _mm_sub_pd(a0, c0);

        T[c] = b0;
        T[c + half_length] = d0;
    }
}
void fft_pointwise(__m128d *T, __m128d *A, int k){
    //  Performs pointwise multiplications of two FFT arrays.

    //Parameters:
    //  -   T           -   Pointer to array.
    //  -   k           -   2^k is the size of the transform

    size_t length = (size_t)1 << k;
    for (size_t c = 0; c < length; c++){
        __m128d a0 = T[c];
        __m128d b0 = A[c];
        __m128d c0, d0;
        c0 = _mm_mul_pd(a0, _mm_unpacklo_pd(b0, b0));
        d0 = _mm_mul_pd(_mm_shuffle_pd(a0, a0, 1), _mm_unpackhi_pd(b0, b0));
        T[c] = _mm_addsub_pd(c0, d0);
    }
}
void int_to_fft(__m128d *T, int k, const uint32_t *A, size_t AL){
    //  Convert word array into FFT array. Put 3 decimal digits per complex point.

    //Parameters:
    //  -   T   -   FFT array
    //  -   k   -   2^k is the size of the transform
    //  -   A   -   word array
    //  -   AL  -   length of word array

    size_t fft_length = (size_t)1 << k;
    __m128d *Tstop = T + fft_length;

    //  Since there are 9 digits per word and we want to put 3 digits per
    //  point, the length of the transform must be at least 3 times the word
    //  length of the input.
    if (fft_length < 3*AL)
        throw "FFT length is too small.";

    //  Convert
    for (size_t c = 0; c < AL; c++){
        uint32_t word = A[c];

        *T++ = _mm_set_sd(word % 1000);
        word /= 1000;
        *T++ = _mm_set_sd(word % 1000);
        word /= 1000;
        *T++ = _mm_set_sd(word);
    }

    //  Pad the rest with zeros.
    while (T < Tstop)
        *T++ = _mm_setzero_pd();
}
void fft_to_int(__m128d *T, int k, uint32_t *A, size_t AL){
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

        f_point = ((double*)T++)[0] * scale;    //  Load and scale
        i_point = (uint64_t)(f_point + 0.5);    //  Round
        carry += i_point;                       //  Add to carry
        word = carry % 1000;                    //  Get 3 digits.
        carry /= 1000;

        f_point = ((double*)T++)[0] * scale;    //  Load and scale
        i_point = (uint64_t)(f_point + 0.5);    //  Round
        carry += i_point;                       //  Add to carry
        word += (carry % 1000) * 1000;          //  Get 3 digits.
        carry /= 1000;

        f_point = ((double*)T++)[0] * scale;    //  Load and scale
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
    int k = 0;
    size_t length = 1;
    while (length < 3*CL){
        length <<= 1;
        k++;
    }
    fft_ensure_table(k);
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
    SIMD_delete deletor;
    auto Ta = std::unique_ptr<__m128d[], SIMD_delete>((__m128d*)_mm_malloc(length * sizeof(__m128d), 16), deletor);
    auto Tb = std::unique_ptr<__m128d[], SIMD_delete>((__m128d*)_mm_malloc(length * sizeof(__m128d), 16), deletor);

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
