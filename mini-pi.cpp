/* Mini Pi
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 07/16/2013
 * 
 *  This is a miniature program that can compute Pi and e to millions of digits
 *  in quasi-linear runtime.
 * 
 *  This program is very slow since it does almost no optimizations. But it uses
 *  asymptotically capable algorithms. So it is capable of computing millions of
 *  digits of Pi - albeit 100x slower than y-cruncher.
 * 
 *  The limit of this program is about 800 million digits. Any higher and the
 *  FFT will encounter malicious round-off error.
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


#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <complex>
#include <algorithm>
#include <memory>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::complex;

#if USE_CHRONO
#include <chrono>
#else
#include <time.h>
#endif

namespace Mini_Pi{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Debug Printing
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
    int length = 1 << k;
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
//  Helpers
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
//  Fast Fourier Transform
#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif
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

    double omega = 2 * M_PI / length;

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

    double omega = -2 * M_PI / length;

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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  BigFloat object
/*  This is the big floating-point object. It represents an arbitrary precision
 *  floating-point number.
 * 
 *  Its numerical value is equal to:
 * 
 *      word = 10^9
 *      word^exp * (T[0] + T[1]*word + T[2]*word^2 + ... + T[L - 1]*word^(L - 1))
 * 
 *  T is an array of 32-bit integers. Each integer stores 9 decimal digits
 *  and must always have a value in the range [0, 999999999].
 * 
 *  T[L - 1] must never be zero. 
 * 
 *  The number is positive when (sign = true) and negative when (sign = false).
 *  Zero is represented as (sign = true) and (L = 0).
 * 
 */
#define YCL_BIGFLOAT_EXTRA_PRECISION    2
class BigFloat{
public:
    BigFloat(BigFloat &&x);
    BigFloat& operator=(BigFloat &&x);

    BigFloat();
    BigFloat(uint32_t x, bool sign = true);

    std::string to_string    (size_t digits = 0) const;
    std::string to_string_sci(size_t digits = 0) const;
    size_t get_precision() const;
    int64_t get_exponent() const;
    uint32_t word_at(int64_t mag) const;

    void negate();
    BigFloat mul(uint32_t x) const;
    BigFloat add(const BigFloat &x, size_t p = 0) const;
    BigFloat sub(const BigFloat &x, size_t p = 0) const;
    BigFloat mul(const BigFloat &x, size_t p = 0) const;
    BigFloat rcp(size_t p) const;
    BigFloat div(const BigFloat &x, size_t p) const;

private:
    bool sign;      //  true = positive or zero, false = negative
    int64_t exp;    //  Exponent
    size_t L;       //  Length
    std::unique_ptr<uint32_t[]> T;

    //  Internal helpers
    int64_t to_string_trimmed(size_t digits, std::string &str) const;
    int ucmp(const BigFloat &x) const;
    BigFloat uadd(const BigFloat &x, size_t p) const;
    BigFloat usub(const BigFloat &x, size_t p) const;

    friend BigFloat invsqrt(uint32_t x, size_t p);
};
BigFloat invsqrt(uint32_t x, size_t p);
////////////////////////////////////////////////////////////////////////////////
//  Move operators
BigFloat::BigFloat(BigFloat &&x)
    : sign(x.sign)
    , exp(x.exp)
    , L(x.L)
    , T(std::move(x.T))
{
    x.sign  = true;
    x.exp   = 0;
    x.L     = 0;
}
BigFloat& BigFloat::operator=(BigFloat &&x){
    sign    = x.sign;
    exp     = x.exp;
    L       = x.L;
    T       = std::move(x.T);

    x.sign  = true;
    x.exp   = 0;
    x.L     = 0;

    return *this;
}
////////////////////////////////////////////////////////////////////////////////
//  Constructors
BigFloat::BigFloat()
    : sign(true)
    , exp(0)
    , L(0)
{}
BigFloat::BigFloat(uint32_t x, bool sign_)
    : sign(true)
    , exp(0)
    , L(1)
{
    //  Construct a BigFloat with a value of x and the specified sign.

    if (x == 0){
        L = 0;
        return;
    }
    sign = sign_;

    T = std::unique_ptr<uint32_t[]>(new uint32_t[1]);
    T[0] = x;
}
////////////////////////////////////////////////////////////////////////////////
//  String Conversion
int64_t BigFloat::to_string_trimmed(size_t digits, std::string &str) const{
    //  Converts this object to a string with "digits" significant figures.

    //  After calling this function, the following expression is equal to the
    //  numeric value of this object. (after truncation of precision)
    //      str + " * 10^" + (return value)

    if (L == 0){
        str = "0";
        return 0;
    }

    //  Collect operands
    int64_t exponent = exp;
    size_t length = L;
    uint32_t *ptr = T.get();

    if (digits == 0){
        //  Use all digits.
        digits = length * 9;
    }else{
        //  Truncate precision
        size_t words = (digits + 17) / 9;
        if (words < length){
            size_t chop = length - words;
            exponent += chop;
            length = words;
            ptr += chop;
        }
    }
    exponent *= 9;

    //  Build string
    char buffer[] = "012345678";
    str.clear();
    size_t c = length;
    while (c-- > 0){
        uint32_t word = ptr[c];
        for (int i = 8; i >= 0; i--){
            buffer[i] = word % 10 + '0';
            word /= 10;
        }
        str += buffer;
    }

    //  Count leading zeros
    size_t leading_zeros = 0;
    while (str[leading_zeros] == '0')
        leading_zeros++;
    digits += leading_zeros;

    //  Truncate
    if (digits < str.size()){
        exponent += str.size() - digits;
        str.resize(digits);
    }

    return exponent;
}
std::string BigFloat::to_string(size_t digits) const{
    //  Convert this number to a string. Auto-select format type.
    if (L == 0)
        return "0.";

    int64_t mag = exp + L;

    //  Use scientific notation if out of range.
    if (mag > 1 || mag < 0)
        return to_string_sci();

    //  Convert
    std::string str;
    int64_t exponent = to_string_trimmed(digits, str);

    //  Less than 1
    if (mag == 0){
        if (sign)
            return std::string("0.") + str;
        else
            return std::string("-0.") + str;
    }

    //  Get a string with the digits before the decimal place.
    std::string before_decimal = std::to_string((long long)T[L - 1]);

    //  Nothing after the decimal place.
    if (exponent >= 0){
        if (sign){
            return before_decimal + ".";
        }else{
            return std::string("-") + before_decimal + ".";
        }
    }

    //  Get digits after the decimal place.
    std::string after_decimal = str.substr((size_t)(str.size() + exponent), (size_t)-exponent);

    if (sign){
        return before_decimal + "." + after_decimal;
    }else{
        return std::string("-") + before_decimal + "." + after_decimal;
    }
}
std::string BigFloat::to_string_sci(size_t digits) const{
    //  Convert to string in scientific notation.
    if (L == 0)
        return "0.";

    //  Convert
    std::string str;
    int64_t exponent = to_string_trimmed(digits, str);

    //  Strip leading zeros.
    {
        size_t leading_zeros = 0;
        while (str[leading_zeros] == '0')
            leading_zeros++;
        str = &str[leading_zeros];
    }

    //  Insert decimal place
    exponent += str.size() - 1;
    str = str.substr(0, 1) + "." + &str[1];

    //  Add exponent
    if (exponent != 0){
        str += " * 10^";
        str += std::to_string(exponent);
    }

    //  Add sign
    if (!sign)
        str = std::string("-") + str;

    return str;
}
////////////////////////////////////////////////////////////////////////////////
//  Getters
size_t BigFloat::get_precision() const{
    //  Returns the precision of the number in words.
    //  Note that each word is 9 decimal digits.
    return L;
}
int64_t BigFloat::get_exponent() const{
    //  Returns the exponent of the number in words.
    //  Note that each word is 9 decimal digits.
    return exp;
}
uint32_t BigFloat::word_at(int64_t mag) const{
    //  Returns the word at the mag'th digit place.
    //  This is useful for additions where you need to access a specific "digit place"
    //  of the operand without having to worry if it's out-of-bounds.

    //  This function is mathematically equal to:
    //      (return value) = floor(this * (10^9)^-mag) % 10^9

    if (mag < exp)
        return 0;
    if (mag >= exp + (int64_t)L)
        return 0;
    return T[(size_t)(mag - exp)];
}
int BigFloat::ucmp(const BigFloat &x) const{
    //  Compare function that ignores the sign.
    //  This is needed to determine which direction subtractions will go.

    //  Magnitude
    int64_t magA = exp + L;
    int64_t magB = x.exp + x.L;
    if (magA > magB)
        return 1;
    if (magA < magB)
        return -1;

    //  Compare
    int64_t mag = magA;
    while (mag >= exp || mag >= x.exp){
        uint32_t wordA = word_at(mag);
        uint32_t wordB = x.word_at(mag);
        if (wordA < wordB)
            return -1;
        if (wordA > wordB)
            return 1;
        mag--;
    }
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
//  Arithmetic
void BigFloat::negate(){
    //  Negate this number.
    if (L == 0)
        return;
    sign = !sign;
}
BigFloat BigFloat::mul(uint32_t x) const{
    //  Multiply by a 32-bit unsigned integer.

    if (L == 0 || x == 0)
        return BigFloat();

    //  Compute basic fields.
    BigFloat z;
    z.sign = sign;
    z.exp  = exp;
    z.L    = L;

    //  Allocate mantissa
    z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L + 1]);

    uint64_t carry = 0;
    for (size_t c = 0; c < L; c++){
        carry += (uint64_t)T[c] * x;                //  Multiply and add to carry
        z.T[c] = (uint32_t)(carry % 1000000000);    //  Store bottom 9 digits
        carry /= 1000000000;                        //  Shift down the carry
    }

    //  Carry out
    if (carry != 0)
        z.T[z.L++] = (uint32_t)carry;

    return z;
}
BigFloat BigFloat::uadd(const BigFloat &x, size_t p) const{
    //  Perform addition ignoring the sign of the two operands.

    //  Magnitude
    int64_t magA = exp + L;
    int64_t magB = x.exp + x.L;
    int64_t top = std::max(magA, magB);
    int64_t bot = std::min(exp, x.exp);

    //  Target length
    int64_t TL = top - bot;

    if (p == 0){
        //  Default value. No trunction.
        p = (size_t)TL;
    }else{
        //  Increase precision
        p += YCL_BIGFLOAT_EXTRA_PRECISION;
    }

    //  Perform precision truncation.
    if (TL > (int64_t)p){
        bot = top - p;
        TL = p;
    }

    //  Compute basic fields.
    BigFloat z;
    z.sign  = sign;
    z.exp   = bot;
    z.L     = (uint32_t)TL;

    //  Allocate mantissa
    z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L + 1]);

    //  Add
    uint32_t carry = 0;
    for (size_t c = 0; bot < top; bot++, c++){
        uint32_t word = word_at(bot) + x.word_at(bot) + carry;
        carry = 0;
        if (word >= 1000000000){
            word -= 1000000000;
            carry = 1;
        }
        z.T[c] = word;
    }

    //  Carry out
    if (carry != 0){
        z.T[z.L++] = 1;
    }

    return z;
}
BigFloat BigFloat::usub(const BigFloat &x, size_t p) const{
    //  Perform subtraction ignoring the sign of the two operands.

    //  "this" must be greater than or equal to x. Otherwise, the behavior
    //  is undefined.

    //  Magnitude
    int64_t magA = exp + L;
    int64_t magB = x.exp + x.L;
    int64_t top = std::max(magA, magB);
    int64_t bot = std::min(exp, x.exp);

    //  Truncate precision
    int64_t TL = top - bot;

    if (p == 0){
        //  Default value. No trunction.
        p = (size_t)TL;
    }else{
        //  Increase precision
        p += YCL_BIGFLOAT_EXTRA_PRECISION;
    }

    if (TL > (int64_t)p){
        bot = top - p;
        TL = p;
    }

    //  Compute basic fields.
    BigFloat z;
    z.sign  = sign;
    z.exp   = bot;
    z.L     = (uint32_t)TL;

    //  Allocate mantissa
    z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L]);

    //  Subtract
    int32_t carry = 0;
    for (size_t c = 0; bot < top; bot++, c++){
        int32_t word = (int32_t)word_at(bot) - (int32_t)x.word_at(bot) - carry;
        carry = 0;
        if (word < 0){
            word += 1000000000;
            carry = 1;
        }
        z.T[c] = word;
    }

    //  Strip leading zeros
    while (z.L > 0 && z.T[z.L - 1] == 0)
        z.L--;
    if (z.L == 0){
        z.exp = 0;
        z.sign = true;
        z.T.reset();
    }

    return z;
}
BigFloat BigFloat::add(const BigFloat &x, size_t p) const{
    //  Addition

    //  The target precision is p.
    //  If (p = 0), then no truncation is done. The entire operation is done
    //  at maximum precision with no data loss.

    //  Same sign. Add.
    if (sign == x.sign)
        return uadd(x, p);

    //  this > x
    if (ucmp(x) > 0)
        return usub(x, p);

    //  this < x
    return x.usub(*this, p);
}
BigFloat BigFloat::sub(const BigFloat &x, size_t p) const{
    //  Subtraction

    //  The target precision is p.
    //  If (p = 0), then no truncation is done. The entire operation is done
    //  at maximum precision with no data loss.

    //  Different sign. Add.
    if (sign != x.sign)
        return uadd(x, p);

    //  this > x
    if (ucmp(x) > 0)
        return usub(x, p);

    //  this < x
    BigFloat z = x.usub(*this, p);
    z.negate();
    return z;
}
BigFloat BigFloat::mul(const BigFloat &x, size_t p) const{
    //  Multiplication

    //  The target precision is p.
    //  If (p = 0), then no truncation is done. The entire operation is done
    //  at maximum precision with no data loss.

    //  Either operand is zero.
    if (L == 0 || x.L == 0)
        return BigFloat();

    if (p == 0){
        //  Default value. No trunction.
        p = L + x.L;
    }else{
        //  Increase precision
        p += YCL_BIGFLOAT_EXTRA_PRECISION;
    }

    //  Collect operands.
    int64_t Aexp = exp;
    int64_t Bexp = x.exp;
    size_t AL = L;
    size_t BL = x.L;
    uint32_t *AT = T.get();
    uint32_t *BT = x.T.get();

    //  Perform precision truncation.
    if (AL > p){
        size_t chop = AL - p;
        AL = p;
        Aexp += chop;
        AT += chop;
    }
    if (BL > p){
        size_t chop = BL - p;
        BL = p;
        Bexp += chop;
        BT += chop;
    }

    //  Compute basic fields.
    BigFloat z;
    z.sign = sign == x.sign;    //  Sign is positive if signs are equal.
    z.exp  = Aexp + Bexp;       //  Add the exponents.
    z.L    = AL + BL;           //  Add the lenghts for now. May need to correct later.

    //  Allocate mantissa
    z.T = std::unique_ptr<uint32_t[]>(new uint32_t[z.L]);

    //  Perform multiplication.

    //  Determine minimum FFT size.
    int k = 0;
    size_t length = 1;
    while (length < 3*z.L){
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
    //  2^29 * 3 = 1,610,612,736 decimal digits.
    if (k > 29)
        throw "FFT size limit exceeded.";

    int_to_fft(Ta.get(), k, AT, AL);           //  Convert 1st operand
    int_to_fft(Tb.get(), k, BT, BL);           //  Convert 2nd operand
    fft_forward(Ta.get(), k);                //  Transform 1st operand
    fft_forward(Tb.get(), k);                //  Transform 2nd operand
    fft_pointwise(Ta.get(), Tb.get(), k);     //  Pointwise multiply
    fft_inverse(Ta.get(), k);                //  Perform inverse transform.
    fft_to_int(Ta.get(), k, z.T.get(), z.L);   //  Convert back to word array.

    //  Check top word and correct length.
    if (z.T[z.L - 1] == 0)
        z.L--;

    return z;
}
BigFloat BigFloat::rcp(size_t p) const{
    //  Compute reciprocal using Newton's Method.

    //  r1 = r0 - (r0 * x - 1) * r0

    if (L == 0)
        throw "Divide by Zero";

    //  Collect operand
    int64_t Aexp = exp;
    size_t AL = L;
    uint32_t *AT = T.get();

    //  End of recursion. Generate starting point.
    if (p == 0){
        //  Truncate precision to 3.
        p = 3;
        if (AL > p){
            size_t chop = AL - p;
            AL = p;
            Aexp += chop;
            AT += chop;
        }

        //  Convert number to floating-point.
        double val = AT[0];
        if (AL >= 2)
            val += AT[1] * 1000000000.;
        if (AL >= 3)
            val += AT[2] * 1000000000000000000.;

        //  Compute reciprocal.
        val = 1. / val;
        Aexp = -Aexp;

        //  Scale
        while (val < 1000000000.){
            val *= 1000000000.;
            Aexp--;
        }

        //  Rebuild a BigFloat.
        uint64_t val64 = (uint64_t)val;

        BigFloat out;
        out.sign = sign;

        out.T = std::unique_ptr<uint32_t[]>(new uint32_t[2]);
        out.T[0] = (uint32_t)(val64 % 1000000000);
        out.T[1] = (uint32_t)(val64 / 1000000000);
        out.L = 2;
        out.exp = Aexp;

        return out;
    }

    //  Half the precision
    size_t s = p / 2 + 1;
    if (p == 1) s = 0;
    if (p == 2) s = 1;

    //  Recurse at half the precision
    BigFloat T = rcp(s);

    //  r1 = r0 - (r0 * x - 1) * r0
    return T.sub(this->mul(T, p).sub(BigFloat(1), p).mul(T, p), p);
}
BigFloat BigFloat::div(const BigFloat &x, size_t p) const{
    //  Division
    return this->mul(x.rcp(p), p);
}
BigFloat invsqrt(uint32_t x, size_t p){
    //  Compute inverse square root using Newton's Method.

    //            (  r0^2 * x - 1  )
    //  r1 = r0 - (----------------) * r0
    //            (       2        )

    if (x == 0)
        throw "Divide by Zero";

    //  End of recursion. Generate starting point.
    if (p == 0){
        double val = 1. / sqrt((double)x);

        int64_t exponent = 0;

        //  Scale
        while (val < 1000000000.){
            val *= 1000000000.;
            exponent--;
        }

        //  Rebuild a BigFloat.
        uint64_t val64 = (uint64_t)val;

        BigFloat out;
        out.sign = true;

        out.T = std::unique_ptr<uint32_t[]>(new uint32_t[2]);
        out.T[0] = (uint32_t)(val64 % 1000000000);
        out.T[1] = (uint32_t)(val64 / 1000000000);
        out.L = 2;
        out.exp = exponent;

        return out;
    }

    //  Half the precision
    size_t s = p / 2 + 1;
    if (p == 1) s = 0;
    if (p == 2) s = 1;

    //  Recurse at half the precision
    BigFloat T = invsqrt(x, s);

    BigFloat temp = T.mul(T, p);         //  r0^2
    temp = temp.mul(x);                 //  r0^2 * x
    temp = temp.sub(BigFloat(1), p);     //  r0^2 * x - 1
    temp = temp.mul(500000000);         //  (r0^2 * x - 1) / 2
    temp.exp--;
    temp = temp.mul(T, p);               //  (r0^2 * x - 1) / 2 * r0
    return T.sub(temp, p);               //  r0 - (r0^2 * x - 1) / 2 * r0
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  e
double logf_approx(double x){
    //  Returns a very good approximation to log(x!).
    //  log(x!) ~ (x + 1/2) * (log(x) - 1) + (log(2*pi) + 1) / 2
    //  This approximation gets better as x is larger.
    if (x <= 1) return 0;
    return (x + .5) * (log(x) - 1) + (1.4189385332046727417803297364056176398613974736378);
}
size_t e_terms(size_t p){
    //  Returns the # of terms needed to reach a precision of p.

    //  The taylor series converges to log(x!) / log(10) decimal digits after
    //  x terms. So to find the number of terms needed to reach a precision of p
    //  we need to solve this question for x:
    //      p = log(x!) / log(1000000000)

    //  This function solves this equation via binary search.

    double sizeL = (double)p * 20.723265836946411156161923092159277868409913397659 + 1;

    size_t a = 0;
    size_t b = 1;

    //  Double up
    while (logf_approx((double)b) < sizeL)
        b <<= 1;

    //  Binary search
    while (b - a > 1){
        size_t m = (a + b) >> 1;

        if (logf_approx((double)m) < sizeL)
            a = m;
        else
            b = m;
    }

    return b + 2;
}
void e_BSR(BigFloat &P, BigFloat &Q, uint32_t a, uint32_t b){
    //  Binary Splitting recursion for exp(1).

    if (b - a == 1){
        P = BigFloat(1);
        Q = BigFloat(b);
        return;
    }

    uint32_t m = (a + b) / 2;

    BigFloat P0, Q0, P1, Q1;
    e_BSR(P0, Q0, a, m);
    e_BSR(P1, Q1, m, b);

    P = P0.mul(Q1).add(P1);
    Q = Q0.mul(Q1);
}
void e(size_t digits){
    //  The leading 2 doesn't count.
    digits++;

    size_t p = (digits + 8) / 9;
    size_t terms = e_terms(p);

    //  Limit Exceeded
    if ((uint32_t)terms != terms)
        throw "Limit Exceeded";

    cout << "Computing e..." << endl;
    cout << "Algorithm: Taylor Series of exp(1)" << endl << endl;

    double time0 = wall_clock();

    cout << "Summing Series... " << terms << " terms" << endl;
    BigFloat P, Q;
    e_BSR(P, Q, 0, (uint32_t)terms);
    double time1 = wall_clock();
    cout << "Time: " << time1 - time0 << endl;

    cout << "Division... " << endl;
    P = P.div(Q, p).add(BigFloat(1), p);
    double time2 = wall_clock();
    cout << "Time: " << time2 - time1 << endl;

    cout << "Total Time = " << time2 - time0 << endl << endl;

    dump_to_file("e.txt", P.to_string(digits));
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Pi
void Pi_BSR(BigFloat &P, BigFloat &Q, BigFloat &R, uint32_t a, uint32_t b, size_t p){
    //  Binary Splitting recursion for the Chudnovsky Formula.

    if (b - a == 1){
        //  P = (13591409 + 545140134 b)(2b-1)(6b-5)(6b-1) (-1)^b
        P = BigFloat(b).mul(545140134);
        P = P.add(BigFloat(13591409));
        P = P.mul(2*b - 1);
        P = P.mul(6*b - 5);
        P = P.mul(6*b - 1);
        if (b % 2 == 1)
            P.negate();

        //  Q = 10939058860032000 * b^3
        Q = BigFloat(b);
        Q = Q.mul(Q).mul(Q).mul(26726400).mul(409297880);

        //  R = (2b-1)(6b-5)(6b-1)
        R = BigFloat(2*b - 1);
        R = R.mul(6*b - 5);
        R = R.mul(6*b - 1);

        return;
    }

    uint32_t m = (a + b) / 2;

    BigFloat P0, Q0, R0, P1, Q1, R1;
    Pi_BSR(P0, Q0, R0, a, m, p);
    Pi_BSR(P1, Q1, R1, m, b, p);

    P = P0.mul(Q1, p).add(P1.mul(R0, p), p);
    Q = Q0.mul(Q1, p);
    R = R0.mul(R1, p);
}
void Pi(size_t digits){
    //  The leading 3 doesn't count.
    digits++;

    size_t p = (digits + 8) / 9;
    size_t terms = (size_t)(p * 0.6346230241342037371474889163921741077188431452678) + 1;

    //  Limit Exceeded
    if ((uint32_t)terms != terms)
        throw "Limit Exceeded";

    cout << "Computing Pi..." << endl;
    cout << "Algorithm: Chudnovsky Formula" << endl << endl;

    double time0 = wall_clock();

    cout << "Summing Series... " << terms << " terms" << endl;
    BigFloat P, Q, R;
    Pi_BSR(P, Q, R, 0, (uint32_t)terms, p);
    P = Q.mul(13591409).add(P, p);
    Q = Q.mul(4270934400);
    double time1 = wall_clock();
    cout << "Time: " << time1 - time0 << endl;

    cout << "Division... " << endl;
    P = Q.div(P, p);
    double time2 = wall_clock();
    cout << "Time: " << time2 - time1 << endl;

    cout << "InvSqrt... " << endl;
    Q = invsqrt(10005, p);
    double time3 = wall_clock();
    cout << "Time: " << time3 - time2 << endl;

    cout << "Final Multiply... " << endl;
    P = P.mul(Q, p);
    double time4 = wall_clock();
    cout << "Time: " << time4 - time3 << endl;

    cout << "Total Time = " << time4 - time0 << endl << endl;

    dump_to_file("pi.txt", P.to_string(digits));
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}   //  Namespace: Mini_Pi
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int string_to_double(const char *a, double *r, const double min, const double max) {
  char *p;
  double d = strtod(a, &p);
  if ((p == a) || (*p != 0) || errno == ERANGE || (d < min ) || (d > max ) ) {
    cerr << "ERROR when parsing \"" << a << "\" as double value. Expecting number in range " << min << " - " << max << " in double notation, see \"man strtod\" for details." << endl;
    return 1;
  }
  *r = d;
  return 0;
}

int main(int argc, char** argv){

    size_t digits = 1000000;
    bool parsing_passed = false;

    if ( argc == 2 ) {
      double x;
      if ( string_to_double(argv[1], &x, 1000, 800e6) == 0 ) {
        parsing_passed = true;
        digits = x;
      }
    }

    if (argc >2 || ! parsing_passed) {
      cerr << "Unsupported number of arguments" << endl;
      cerr << "Usage: " << argv[0] << " [digits] (max 800 millions). Input in double notation (800e6)" << endl;
      cerr << "Default value: " << digits << " digits." << endl;
      return EXIT_FAILURE;
    }

    Mini_Pi::e (digits);
    Mini_Pi::Pi(digits);

#ifdef _WIN32
    system("pause");
#endif
}
