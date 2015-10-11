/* Pi.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 *      Compute Pi using the Chudnovsky Formula.
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

    ensure_FFT_tables(2*p);

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
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
