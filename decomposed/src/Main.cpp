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

#include "DebugPrinting.h"
#include "Tools.h"
#include "FFT.h"
#include "BigFloat.h"
#include "Constants.h"

int main(){
    size_t digits = 1000000;

    Mini_Pi::e (digits);
    Mini_Pi::Pi(digits);

#ifdef _WIN32
    system("pause");
#endif
}
