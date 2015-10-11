/* e.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 *      Compute e using Taylor Series of exp(1).
 * 
 */

#include <iostream>

#include "Tools.h"
#include "FFT.h"
#include "BigFloat.h"
#include "Constants.h"

namespace Mini_Pi{
    using std::cout;
    using std::endl;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
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

    ensure_FFT_tables(2*p);

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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
