y-cruncher-Mini
===============

    Author           : Alexander J. Yee
    Date Created     : 07/09/2013
    Last Modified    : 07/13/2013

A miniature program that can compute Pi to millions of digits.


This is one of my not-so-serious projects hacked together in a very short amount
of time - for no other purpose than fun and to practice my C++11 skills.

The goal of this project is to construct a (minimal) program that can compute Pi
to millions of digits in a quasi-linear runtime.

To do this, the program implements the following algorithms:
 - FFT-based Multiplication
 - Newton's Method
 - Binary Splitting

The focus of this project is conciseness and readability, not optimization.
So this program is very slow - more than 100x slower than y-cruncher itself.
Nevertheless, it runs in quasi-linear time. So it can feasibly compute Pi to
hundreds of millions of digits if you're willing to wait.


Feel free to branch the project and try out your own optimizations. I've found
it amusingly fun to toy around with. There's tons of room for optimizations
and it's easy to speed up the baseline by a factor of 4 with minimal effort.


For what it's worth, here's a short list of optimizations to try out:
 - Cache the twiddle factors.
 - Exploit the fact that FFT of real input leads to complex conjugate output.
   (In other words, half the FFT work is redundant.)
 - Use more than 3 digits per FFT point when it's safe to do so.
 - Hard code larger FFT base sizes.
 - Parallelize the computation.
 - Vectorization and SIMD
 - Implement the Basecase and Karatsuba multiplication algorithms to better
   handle small multiplications.
 - Small number optimization to reduce heap allocation calls.
 - Binary Splitting always splits the series in half. But half isn't optimal.


Files:
 - y-cruncher_mini.cpp             - This is the baseline. No optimizations.
 - y-cruncher_mini_optimized_1.cpp - This applies two very basic optimizations
                                     for a 2-4x speedup.
