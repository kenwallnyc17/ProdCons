#ifndef MKTDATASYSTEMNOTES_H_INCLUDED
#define MKTDATASYSTEMNOTES_H_INCLUDED

/**
    Order Priority

        * order modification (price, size, OID) can be provided in different msg types from the exch:
                Replace, Mod, Can are the usual names
        * whichever ones are provided, it looks for the most part like this:

            always is a priority change:
                price change - obvious
                OID change (ultimately a cancel and replace - maybe combined into one msg)

            always not a priority change: (this needs to be verified by looking at the priority flags sent)
                size reduced

        * NASD, NYSE, MEMX,LTSE don't allow mod up on size
        * CBOE, IEX, MIAX allow mod up on size and provide a flag indicating priority change

        * msgs named Modify often have priority change flags because it's not straight forward as to whether or
            not priority changes

        * msgs named Cancel are always size reducing and hence never have a change priority flag
*/

/**
    intrinsics:

    __m128i zero = _mm_setzero_si128();

    _mm_loadu_si128((__m128i const*)p);

    _mm_storeu_si128((__m128i *)d, x);

    Aligning dynamically allocated memory
        Memory allocated with new or malloc is aligned by 8 or 16 depending on the platform. This
        is a problem with vector operations when alignment by 16 or more is required. The C++ 17
        standard gives you the required alignment automatically when using operator new:
        // Example 12.9
        // Aligned memory allocation in C++17
        int arraysize = 100;
        __m512 * pp = new __m512[arraysize];
        // pp will be aligned by alignof(__m512) if using C++17
*/


#endif // MKTDATASYSTEMNOTES_H_INCLUDED
