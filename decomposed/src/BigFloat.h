/* BigFloat.h
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 *  This is the big floating-point object. It represents an arbitrary precision
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

#ifndef _miniPi_BigFloat_H
#define _miniPi_BigFloat_H

#include <stdint.h>
#include <string>
#include <memory>
namespace Mini_Pi{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Object
class BigFloat{
    static const size_t EXTRA_PRECISION = 2;

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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  External Functions
BigFloat invsqrt(uint32_t x, size_t p);
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
#endif
