Mini Pi
===============

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
So it is very slow - more than 100x slower than [y-cruncher](http://www.numberworld.org/y-cruncher/).
Nevertheless, it runs in quasi-linear time. So it can feasibly compute Pi to
hundreds of millions of digits if you're willing to wait.

Mini Pi will compute N digits of e in `O(N * Log(N)^2)` time and N digits of Pi
in `O(N * Log(N)^3)` time.

These complexities are exact since the FFT uses a fixed # of digits per point.
However, the program will fail above 800 million digits due to round off error.
(It might actually work up to 1.6 billion digits, but it gets risky...)

-----

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
 - Implement the Basecase and Karatsuba algorithms to better handle small multiplications.
 - Small number optimization to reduce heap allocation calls.
 - Binary Splitting always splits the series in half. But half isn't optimal.
 - Micro-optimizations such as manual loop-unrolling.


Files:
 - `mini-pi.cpp`<br>
   This is the baseline. No optimizations.

 - `mini-pi_optimized_1_cached_twiddles.cpp`<br>
   This version caches the twiddle factors and stops the FFT recursions one step earlier.

 - `mini-pi_optimized_2_SSE3.cpp`<br>
   This version vectorizes the FFT using SSE3 instructions.

 - `mini-pi_optimized_3_OpenMP.cpp`<br>
   This version parallelizes the computation using OpenMP.

Note that each of these builds upon the previous version. So the last one (OpenMP)
incorporates all the optimizations listed here.

-----

While there's a lot of room to optimize the program, it's unlikely it can be made as
fast as y-cruncher without a complete rewrite using a different design.

Some of the things that y-cruncher does are:

Overall:
 - Binary arithmetic in base `2^32` or base `2^64`.
 - Tight precision control to avoid unnecessary computation.
 - Aggressive Parallelization: All algorithms are carefully designed for parallelization.
 - `O(1)` calls to `malloc()` - no memory allocation overhead
 - A very elaborate mechanism for selecting optimal Binary Splitting split points.
 - Basecase and Karatsuba multiplication for small products.
 - Other FFT-based algorithms for massive sized products.
 - Internal Error-Detection and Correction for dealing with hardware failures.
 - "Swap Mode" that will use disk for computations that are too big to fit in ram.

FFT:
 - Support for transforms sizes of: `2^k`, `3 * 2^k`, and `5 * 2^k`.
 - Cached twiddle factors.
 - Real-to-Complex transforms
 - Split-Radix FFT
 - Variable bits per point. (typically 11 - 19 bits)
 - Basecase FFTs hard-coded up to 64 points.
 - Instruction Sets: SSE3, SSE4.1, AVX. (XOP and AVX2 coming sometime after v0.6.3)
 - Micro-Optimizations: Loop Unrolling, Function Inlining
 - Reuse of redundant FFT transforms.
