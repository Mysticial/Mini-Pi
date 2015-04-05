/* FFT.cpp
 *
 * Author           : Alexander J. Yee
 * Date Created     : 07/09/2013
 * Last Modified    : 03/22/2015
 * 
 */

//  Pick your optimizations.
//#define MINI_PI_CACHED_TWIDDLES
#define MINI_PI_SSE3    //  Includes caching of twiddle factors.

#if 0
#elif defined MINI_PI_SSE3
#include "FFT_SSE3.ipp"
#elif defined MINI_PI_CACHED_TWIDDLES
#include "FFT_CachedTwiddles.ipp"
#else
#include "FFT_Basic.ipp"
#endif
