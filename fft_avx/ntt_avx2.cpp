#include <vector>
#include <complex>
#include <iostream>
#include <cassert>
#include <cstring>
#include <ctime>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <string>
#include <array>
#include <fstream>
#include <type_traits>
#include <immintrin.h>
#pragma GCC optimize("inline")
#pragma GCC target("avx2")

namespace hint
{
    using Float32 = float;
    using Float64 = double;
    using Complex32 = std::complex<Float32>;
    using Complex64 = std::complex<Float64>;
    constexpr size_t L1_BYTE = size_t(1) << 15; // L1 cache size, change this if you know your cache size.
    constexpr size_t L2_BYTE = size_t(1) << 20; // L2 cache size, change this if you know your cache size.

    constexpr Float64 HINT_PI = 3.141592653589793238462643;
    constexpr Float64 HINT_2PI = HINT_PI * 2;
    // bits of 1, equals to 2^bits - 1
    template <typename T>
    constexpr T all_one(int bits)
    {
        T temp = T(1) << (bits - 1);
        return temp - 1 + temp;
    }

    // Leading zeros
    template <typename IntTy>
    constexpr int hint_clz(IntTy x)
    {
        constexpr uint32_t MASK32 = uint32_t(0xFFFF) << 16;
        int res = sizeof(IntTy) * CHAR_BIT;
        if (x & MASK32)
        {
            res -= 16;
            x >>= 16;
        }
        if (x & (MASK32 >> 8))
        {
            res -= 8;
            x >>= 8;
        }
        if (x & (MASK32 >> 12))
        {
            res -= 4;
            x >>= 4;
        }
        if (x & (MASK32 >> 14))
        {
            res -= 2;
            x >>= 2;
        }
        if (x & (MASK32 >> 15))
        {
            res -= 1;
            x >>= 1;
        }
        return res - x;
    }
    // Leading zeros
    constexpr int hint_clz(uint64_t x)
    {
        if (x & (uint64_t(0xFFFFFFFF) << 32))
        {
            return hint_clz(uint32_t(x >> 32));
        }
        return hint_clz(uint32_t(x)) + 32;
    }

    // Integer bit length
    template <typename IntTy>
    constexpr int hint_bit_length(IntTy x)
    {
        if (x == 0)
        {
            return 0;
        }
        return sizeof(IntTy) * CHAR_BIT - hint_clz(x);
    }

    // Integer log2
    template <typename IntTy>
    constexpr int hint_log2(IntTy x)
    {
        return (sizeof(IntTy) * CHAR_BIT - 1) - hint_clz(x);
    }

    constexpr int hint_ctz(uint32_t x)
    {
        int r = 31;
        x &= (-x);
        if (x & 0x0000FFFF)
        {
            r -= 16;
        }
        if (x & 0x00FF00FF)
        {
            r -= 8;
        }
        if (x & 0x0F0F0F0F)
        {
            r -= 4;
        }
        if (x & 0x33333333)
        {
            r -= 2;
        }
        if (x & 0x55555555)
        {
            r -= 1;
        }
        return r;
    }

    constexpr int hint_ctz(uint64_t x)
    {
        if (x & 0xFFFFFFFF)
        {
            return hint_ctz(uint32_t(x));
        }
        return hint_ctz(uint32_t(x >> 32)) + 32;
    }

    // Fast power
    template <typename T, typename T1>
    constexpr T qpow(T m, T1 n)
    {
        T result = 1;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                result *= m;
            }
            m *= m;
            n >>= 1;
        }
        return result;
    }

    // Fast power with mod
    template <typename T, typename T1>
    constexpr T qpow(T m, T1 n, T mod)
    {
        T result = 1;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                result *= m;
                result %= mod;
            }
            m *= m;
            m %= mod;
            n >>= 1;
        }
        return result;
    }

    // Get cloest power of 2 that not larger than n
    template <typename T>
    constexpr T int_floor2(T n)
    {
        constexpr int bits = sizeof(n) * CHAR_BIT;
        for (int i = 1; i < bits; i *= 2)
        {
            n |= (n >> i);
        }
        return (n >> 1) + 1;
    }

    // Get cloest power of 2 that not smaller than n
    template <typename T>
    constexpr T int_ceil2(T n)
    {
        constexpr int bits = sizeof(n) * CHAR_BIT;
        n--;
        for (int i = 1; i < bits; i *= 2)
        {
            n |= (n >> i);
        }
        return n + 1;
    }

    // x + y = sum with carry
    template <typename UintTy>
    constexpr UintTy add_half(UintTy x, UintTy y, bool &cf)
    {
        x = x + y;
        cf = (x < y);
        return x;
    }

    // x - y = diff with borrow
    template <typename UintTy>
    constexpr UintTy sub_half(UintTy x, UintTy y, bool &bf)
    {
        y = x - y;
        bf = (y > x);
        return y;
    }

    // x + y + cf = sum with carry
    template <typename UintTy>
    constexpr UintTy add_carry(UintTy x, UintTy y, bool &cf)
    {
        UintTy sum = x + cf;
        cf = (sum < x);
        sum += y;             // carry
        cf = cf || (sum < y); // carry
        return sum;
    }

    // x - y - bf = diff with borrow
    template <typename UintTy>
    constexpr UintTy sub_borrow(UintTy x, UintTy y, bool &bf)
    {
        UintTy diff = x - bf;
        bf = (diff > x);
        y = diff - y;          // borrow
        bf = bf || (y > diff); // borrow
        return y;
    }

    // a * x + b * y = gcd(a,b)
    template <typename IntTy>
    constexpr IntTy exgcd(IntTy a, IntTy b, IntTy &x, IntTy &y)
    {
        if (b == 0)
        {
            x = 1;
            y = 0;
            return a;
        }
        IntTy k = a / b;
        IntTy g = exgcd(b, a - k * b, y, x);
        y -= k * x;
        return g;
    }

    // return n^-1 mod mod
    template <typename IntTy>
    constexpr IntTy mod_inv(IntTy n, IntTy mod)
    {
        n %= mod;
        IntTy x = 0, y = 0;
        exgcd(n, mod, x, y);
        if (x < 0)
        {
            x += mod;
        }
        else if (x >= mod)
        {
            x -= mod;
        }
        return x;
    }

    // return n^-1 mod 2^pow, Newton iteration
    constexpr uint64_t inv_mod2pow(uint64_t n, int pow)
    {
        const uint64_t mask = all_one<uint64_t>(pow);
        uint64_t xn = 1, t = n & mask;
        while (t != 1)
        {
            xn = (xn * (2 - t));
            t = (xn * n) & mask;
        }
        return xn & mask;
    }

    namespace modint
    {
        //  Montgomery for mod < 2^30
        //  default R = 2^32
        template <uint32_t MOD>
        struct MontInt32Lazy
        {
            static_assert(hint_log2(MOD) < 30, "MOD can't be larger than 30 bits");
            uint32_t data;
            using IntType = uint32_t;

            constexpr MontInt32Lazy() : data(0) {}
            constexpr MontInt32Lazy(uint32_t n) : data(toMont(n)) {}

            constexpr MontInt32Lazy operator+(MontInt32Lazy rhs) const
            {
                rhs.data = data + rhs.data;
                return rhs.largeNorm();
            }
            constexpr MontInt32Lazy operator-(MontInt32Lazy rhs) const
            {
                rhs.data = data - rhs.data;
                rhs.data = rhs.data > data ? rhs.data + mod2() : rhs.data;
                return rhs;
            }
            constexpr MontInt32Lazy operator*(MontInt32Lazy rhs) const
            {
                rhs.data = redcLazy(uint64_t(data) * rhs.data);
                return rhs;
            }
            constexpr MontInt32Lazy &operator+=(const MontInt32Lazy &rhs)
            {
                return *this = *this + rhs;
            }
            constexpr MontInt32Lazy &operator-=(const MontInt32Lazy &rhs)
            {
                return *this = *this - rhs;
            }
            constexpr MontInt32Lazy &operator*=(const MontInt32Lazy &rhs)
            {
                data = redc(uint64_t(data) * rhs.data);
                return *this;
            }
            constexpr MontInt32Lazy norm() const
            {
                MontInt32Lazy res;
                res.data = data >= mod() ? data - mod() : data;
                return res;
            }
            constexpr MontInt32Lazy largeNorm() const
            {
                MontInt32Lazy res;
                res.data = data >= mod2() ? data - mod2() : data;
                return res;
            }
            constexpr MontInt32Lazy add(MontInt32Lazy rhs) const
            {
                rhs.data = data + rhs.data;
                return rhs;
            }
            constexpr MontInt32Lazy sub(MontInt32Lazy rhs) const
            {
                rhs.data = data - rhs.data + mod2();
                return rhs;
            }
            constexpr operator uint32_t() const
            {
                return toInt(data);
            }
            constexpr MontInt32Lazy inv() const
            {
                return qpow(*this, mod() - 2);
            }
            static constexpr uint32_t mod()
            {
                return MOD;
            }
            static constexpr uint32_t mod2()
            {
                return MOD * 2;
            }
            static constexpr uint32_t modInv()
            {
                constexpr uint32_t mod_inv = uint32_t(inv_mod2pow(mod(), 32));
                return mod_inv;
            }
            static constexpr uint32_t modNegInv()
            {
                constexpr uint32_t mod_neg_inv = uint32_t(0 - modInv());
                return mod_neg_inv;
            }
            static_assert((mod() * modInv()) == 1, "mod_inv not correct");

            static constexpr uint32_t toMont(uint32_t n)
            {
                return (uint64_t(n) << 32) % MOD;
            }
            static constexpr uint32_t toInt(uint32_t n)
            {
                return redc(n);
            }

            static constexpr uint32_t redcLazy(uint64_t n)
            {
                uint32_t prod = uint32_t(n) * modNegInv();
                return (uint64_t(prod) * mod() + n) >> 32;
            }
            static constexpr uint32_t redc(uint64_t n)
            {
                uint32_t res = redcLazy(n);
                return res < mod() ? res : res - mod();
            }
            constexpr MontInt32Lazy divR() const
            {
                MontInt32Lazy res;
                res.data = redc(data);
                return res;
            }
            constexpr MontInt32Lazy mulR() const
            {
                MontInt32Lazy res;
                res.data = toMont(data);
                return res;
            }
        };
    }

    namespace simd
    {
        template <typename MontInt32Type>
        struct MontInt32X8
        {
            using MontInt = MontInt32Type;
            using Int32X8 = __m256i;
            __m256i data;

            MontInt32X8() : data(_mm256_setzero_si256()) {}
            MontInt32X8(MontInt x) : data(_mm256_set1_epi32(x.data)) {}
            MontInt32X8(Int32X8 n) : data(toMont(n)) {}
            template <typename T>
            MontInt32X8(const T *p)
            {
                loadu(p);
            }

            MontInt32X8 operator+(MontInt32X8 rhs) const
            {
                rhs.data = _mm256_add_epi32(data, rhs.data);
                return rhs.largeNorm();
            }
            MontInt32X8 operator-(MontInt32X8 rhs) const
            {
                rhs.data = _mm256_sub_epi32(data, rhs.data);
                return rhs.smallNorm();
            }
            MontInt32X8 operator*(MontInt32X8 rhs) const
            {
                rhs.data = mulMontLazy(data, rhs.data);
                return rhs;
            }
            MontInt32X8 &operator+=(const MontInt32X8 &rhs)
            {
                return *this = *this + rhs;
            }
            MontInt32X8 &operator-=(const MontInt32X8 &rhs)
            {
                return *this = *this - rhs;
            }
            MontInt32X8 &operator*=(const MontInt32X8 &rhs)
            {
                return *this = *this * rhs;
            }

            MontInt32X8 add(MontInt32X8 rhs) const
            {
                rhs.data = _mm256_add_epi32(data, rhs.data);
                return rhs;
            }
            MontInt32X8 sub(MontInt32X8 rhs) const
            {
                rhs.data = _mm256_sub_epi32(data, rhs.data);
                rhs.data = _mm256_add_epi32(mod2X8(), rhs.data);
                return rhs;
            }

            static Int32X8 montRedcLazy(Int32X8 even64, Int32X8 odd64)
            {
                Int32X8 prod0 = mul64(even64, modNX8());
                Int32X8 prod1 = mul64(odd64, modNX8());
                prod0 = mul64(prod0, modX8());
                prod1 = mul64(prod1, modX8());
                prod0 = rawAdd64(prod0, even64);
                prod1 = rawAdd64(prod1, odd64);
                prod0 = rShift64<32>(prod0);
                return blend<0b10101010>(prod0, prod1);
            }
            static Int32X8 montRedc(Int32X8 even64, Int32X8 odd64)
            {
                MontInt32X8 res;
                res.data = montRedcLazy(even64, odd64);
                return res.norm().data;
            }

            static Int32X8 mulMont(Int32X8 lhs, Int32X8 rhs)
            {
                mul32X32To64(lhs, rhs, lhs, rhs);
                return montRedc(lhs, rhs);
            }
            static Int32X8 mulMontLazy(Int32X8 lhs, Int32X8 rhs)
            {
                mul32X32To64(lhs, rhs, lhs, rhs);
                return montRedcLazy(lhs, rhs);
            }

            static void mul32X32To64(Int32X8 lhs, Int32X8 rhs, Int32X8 &low, Int32X8 &high)
            {
                low = mul64(lhs, rhs);
                high = mul64(rShift64<32>(lhs), rShift64<32>(rhs));
            }

            MontInt32X8 norm() const
            {
                MontInt32X8 dif;
                dif.data = rawSub(data, modX8());
                dif.data = minU32(data, dif.data);
                return dif;
            }
            MontInt32X8 largeNorm() const
            {
                MontInt32X8 dif;
                dif.data = rawSub(data, mod2X8());
                dif.data = minU32(data, dif.data);
                return dif;
            }
            MontInt32X8 smallNorm() const
            {
                MontInt32X8 sum;
                sum.data = rawAdd(data, mod2X8());
                sum.data = minU32(data, sum.data);
                return sum;
            }
            // [a,b]->[0,a]
            MontInt32X8 lshift32In64() const
            {
                MontInt32X8 res;
                res.data = lShift64<32>(data);
                return res;
            }
            // [a,b]->[b,0]
            MontInt32X8 rshift32In64() const
            {
                MontInt32X8 res;
                res.data = rShift64<32>(data);
                return res;
            }
            template <int N>
            static MontInt32X8 blend(MontInt32X8 a, MontInt32X8 b)
            {
                a.data = _mm256_blend_epi32(a.data, b.data, N);
                return a;
            }
            template <int N>
            static MontInt32X8 permute2X128(MontInt32X8 a, MontInt32X8 b)
            {
                a.data = _mm256_permute2x128_si256(a.data, b.data, N);
                return a;
            }
            template <int N>
            MontInt32X8 lShiftByte128() const
            {
                MontInt32X8 res;
                res.data = _mm256_bslli_epi128(data, N);
                return res;
            }
            template <int N>
            MontInt32X8 rShiftByte128() const
            {
                MontInt32X8 res;
                res.data = _mm256_bsrli_epi128(data, N);
                return res;
            }
            MontInt32X8 lshift64In128() const
            {
                return lShiftByte128<8>();
            }
            MontInt32X8 rshift64In128() const
            {
                return rShiftByte128<8>();
            }
            // even[a,b],odd[c,d]->[a,d]
            static MontInt32X8 cross32(const MontInt32X8 &even, const MontInt32X8 &odd)
            {
                return blend<0b10101010>(even, odd);
            }
            // even[a,b,c,d],odd[e,f,g,h]->[a,b,g,h]
            static MontInt32X8 cross64(const MontInt32X8 &even, const MontInt32X8 &odd)
            {
                return blend<0b11001100>(even, odd);
            }
            // lo[a,b],hi[c,d]->[a,c]
            static MontInt32X8 packLo128(const MontInt32X8 &lo, const MontInt32X8 &hi)
            {
                return permute2X128<0x20>(lo, hi);
            }
            // lo[a,b],hi[c,d]->[b,d]
            static MontInt32X8 packHi128(const MontInt32X8 &lo, const MontInt32X8 &hi)
            {
                return permute2X128<0x31>(lo, hi);
            }
            static constexpr uint32_t mod()
            {
                return MontInt::mod();
            }
            static constexpr uint32_t modNegInv()
            {
                return MontInt::modNegInv();
            }
            static Int32X8 zeroX8()
            {
                return _mm256_setzero_si256();
            }
            static Int32X8 modX8()
            {
                return _mm256_set1_epi32(mod());
            }
            static Int32X8 mod2X8()
            {
                constexpr uint32_t MOD2 = mod() * 2;
                return _mm256_set1_epi32(MOD2);
            }
            static Int32X8 modNX8()
            {
                constexpr uint32_t MOD_INV_NEG = modNegInv();
                return _mm256_set1_epi32(MOD_INV_NEG);
            }
            static Int32X8 r2X8()
            {
                constexpr uint32_t R = (uint64_t(1) << 32) % mod(), R2 = uint64_t(R) * R % mod();
                return _mm256_set1_epi32(R2);
            }

            static Int32X8 mul64(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_mul_epu32(lhs, rhs);
            }
            template <int N>
            static Int32X8 lShift64(const Int32X8 &n)
            {
                return _mm256_slli_epi64(n, N);
            }
            template <int N>
            static Int32X8 rShift64(const Int32X8 &n)
            {
                return _mm256_srli_epi64(n, N);
            }

            static Int32X8 rawAdd(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_add_epi32(lhs, rhs);
            }
            static Int32X8 rawSub(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_sub_epi32(lhs, rhs);
            }
            static Int32X8 rawAdd64(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_add_epi64(lhs, rhs);
            }
            static Int32X8 rawSub64(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_sub_epi64(lhs, rhs);
            }

            static Int32X8 maxU32(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_max_epu32(lhs, rhs);
            }
            static Int32X8 minU32(const Int32X8 &lhs, const Int32X8 &rhs)
            {
                return _mm256_min_epu32(lhs, rhs);
            }

            template <int N>
            static Int32X8 blend(const Int32X8 &a, const Int32X8 &b)
            {
                return _mm256_blend_epi32(a, b, N);
            }

            static Int32X8 toMont(const Int32X8 &n)
            {
                return mulMont(r2X8(), n);
            }
            static Int32X8 toInt(const Int32X8 &n)
            {
                Int32X8 e = evenElements(n);
                Int32X8 o = rShift64<32>(n);
                return montRedc(e, o);
            }
            Int32X8 toInt() const
            {
                return toInt(data);
            }
            // a,b,c,d -> a,0,b,0
            static Int32X8 evenElements(const Int32X8 &n)
            {
                return blend<0b10101010>(n, zeroX8());
            }
            // a,b,c,d -> 0,b,0,d
            static Int32X8 oddElements(const Int32X8 &n)
            {
                return blend<0b01010101>(n, zeroX8());
            }

            void set1(int32_t n)
            {
                data = _mm256_set1_epi32(n);
            }
            template <typename T>
            void loadu(const T *p)
            {
                data = _mm256_loadu_si256((const Int32X8 *)p);
            }
            template <typename T>
            void load(const T *p)
            {
                data = _mm256_load_si256((const Int32X8 *)p);
            }
            template <typename T>
            void loadMask(const T *p, const Int32X8 &mask)
            {
                data = _mm256_maskload_epi32((const int *)p, mask);
            }
            template <typename T>
            void loadN(const T *p, int n)
            {
                constexpr uint32_t m = UINT32_MAX;
                constexpr uint32_t mask_arr[16]{m, m, m, m, m, m, m, m};
                Int32X8 mask = _mm256_loadu_si256((const Int32X8 *)(mask_arr + 8 - n));
                loadMask(p, mask);
            }
            template <typename T>
            void storeu(T *p) const
            {
                _mm256_storeu_si256((__m256i *)p, data);
            }
            template <typename T>
            void store(T *p) const
            {
                _mm256_store_si256((__m256i *)p, data);
            }
            template <typename T>
            void storeMask(T *p, const __m256i &mask) const
            {
                _mm256_maskstore_epi32((int *)p, mask, data);
            }
            template <typename T>
            void storeN(T *p, int n) const
            {
                constexpr uint32_t m = UINT32_MAX;
                constexpr uint32_t mask_arr[16]{m, m, m, m, m, m, m, m};
                Int32X8 mask = _mm256_loadu_si256((const Int32X8 *)(mask_arr + 8 - n));
                storeMask(p, mask);
            }
            uint32_t nthU32(size_t i) const
            {
                return _mm256_extract_epi32(data, i);
            }
            uint64_t nthU64(size_t i) const
            {
                return _mm256_extract_epi64(data, i);
            }
            void printU32(bool norm = false) const
            {
                if (!norm)
                {
                    std::cout << "[" << nthU32(0) << "," << nthU32(1)
                              << "," << nthU32(2) << "," << nthU32(3)
                              << "," << nthU32(4) << "," << nthU32(5)
                              << "," << nthU32(6) << "," << nthU32(7) << "]" << std::endl;
                    return;
                }
                largeNorm().norm().printU32();
            }
            void printU64() const
            {
                std::cout << "[" << nthU64(0) << "," << nthU64(1)
                          << "," << nthU64(2) << "," << nthU64(3) << "]" << std::endl;
            }
            void printU32Int() const
            {
                MontInt32X8 res;
                res.data = toInt();
                res.printU32();
            }
        };

        template <typename YMM>
        inline void transpose8x8(YMM &row0, YMM &row1, YMM &row2, YMM &row3, YMM &row4, YMM &row5, YMM &row6, YMM &row7)
        {
            using Type = YMM;
            static_assert(sizeof(Type) == 32);
            __m256 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
            temp0 = _mm256_unpacklo_ps(__m256(row0), __m256(row1));
            temp1 = _mm256_unpackhi_ps(__m256(row0), __m256(row1));
            temp2 = _mm256_unpacklo_ps(__m256(row2), __m256(row3));
            temp3 = _mm256_unpackhi_ps(__m256(row2), __m256(row3));
            temp4 = _mm256_unpacklo_ps(__m256(row4), __m256(row5));
            temp5 = _mm256_unpackhi_ps(__m256(row4), __m256(row5));
            temp6 = _mm256_unpacklo_ps(__m256(row6), __m256(row7));
            temp7 = _mm256_unpackhi_ps(__m256(row6), __m256(row7));
            row0 = Type(_mm256_shuffle_ps(temp0, temp2, _MM_SHUFFLE(1, 0, 1, 0)));
            row1 = Type(_mm256_shuffle_ps(temp0, temp2, _MM_SHUFFLE(3, 2, 3, 2)));
            row2 = Type(_mm256_shuffle_ps(temp1, temp3, _MM_SHUFFLE(1, 0, 1, 0)));
            row3 = Type(_mm256_shuffle_ps(temp1, temp3, _MM_SHUFFLE(3, 2, 3, 2)));
            row4 = Type(_mm256_shuffle_ps(temp4, temp6, _MM_SHUFFLE(1, 0, 1, 0)));
            row5 = Type(_mm256_shuffle_ps(temp4, temp6, _MM_SHUFFLE(3, 2, 3, 2)));
            row6 = Type(_mm256_shuffle_ps(temp5, temp7, _MM_SHUFFLE(1, 0, 1, 0)));
            row7 = Type(_mm256_shuffle_ps(temp5, temp7, _MM_SHUFFLE(3, 2, 3, 2)));
            temp0 = _mm256_permute2f128_ps(__m256(row0), __m256(row4), 0x20);
            temp1 = _mm256_permute2f128_ps(__m256(row1), __m256(row5), 0x20);
            temp2 = _mm256_permute2f128_ps(__m256(row2), __m256(row6), 0x20);
            temp3 = _mm256_permute2f128_ps(__m256(row3), __m256(row7), 0x20);
            temp4 = _mm256_permute2f128_ps(__m256(row0), __m256(row4), 0x31);
            temp5 = _mm256_permute2f128_ps(__m256(row1), __m256(row5), 0x31);
            temp6 = _mm256_permute2f128_ps(__m256(row2), __m256(row6), 0x31);
            temp7 = _mm256_permute2f128_ps(__m256(row3), __m256(row7), 0x31);
            row0 = Type(temp0), row1 = Type(temp1), row2 = Type(temp2), row3 = Type(temp3);
            row4 = Type(temp4), row5 = Type(temp5), row6 = Type(temp6), row7 = Type(temp7);
        }
    }

    namespace transform
    {
        template <typename T>
        constexpr void transform2(T &sum, T &diff)
        {
            T temp0 = sum, temp1 = diff;
            sum = temp0 + temp1;
            diff = temp0 - temp1;
        }

        // 多模式，自动类型，自检查快速数论变换
        namespace ntt
        {
            using namespace modint;
            using namespace simd;
            template <typename IntType>
            constexpr bool check_inv(uint64_t n, uint64_t n_inv, uint64_t mod)
            {
                n %= mod;
                n_inv %= mod;
                IntType m(n);
                m *= IntType(n_inv);
                m %= IntType(mod);
                return m == IntType(1);
            }

            namespace radix2_avx
            {
                template <uint32_t ROOT, typename ModIntType, typename T>
                inline T mul_w41(const T &n)
                {
                    constexpr ModIntType W_4_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 4);
                    return n * T(W_4_1);
                }
                template <uint32_t ROOT, typename ModIntType, typename T>
                inline T mul_w81(const T &n)
                {
                    constexpr ModIntType W_8_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                    return n * T(W_8_1);
                }
                template <uint32_t ROOT, typename ModIntType, typename T>
                inline T mul_w83(const T &n)
                {
                    constexpr ModIntType W_8_3 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8 * 3);
                    return n * T(W_8_3);
                }

                // in: in_out0<4p, in_ou1<4p
                // out: in_out0<4p, in_ou1<4p
                template <typename ModIntType>
                inline void dit_butterfly2(ModIntType &in_out0, ModIntType &in_out1, ModIntType omega)
                {
                    auto x = in_out0.largeNorm();
                    auto y = in_out1 * omega;
                    in_out0 = x.add(y);
                    in_out1 = x.sub(y);
                }

                // in: in_out0<2p, in_ou1<4p
                // out: in_out0<4p, in_ou1<4p
                template <typename ModIntType>
                inline void dit_butterfly2_i24(ModIntType &in_out0, ModIntType &in_out1, ModIntType omega, std::false_type)
                {
                    auto x = in_out0;
                    auto y = in_out1 * omega;
                    in_out0 = x.add(y);
                    in_out1 = x.sub(y);
                }

                // in: in_out0<2p, in_ou1<4p
                // out: in_out0<2p, in_ou1<2p
                template <typename ModIntType>
                inline void dit_butterfly2_i24(ModIntType &in_out0, ModIntType &in_out1, ModIntType omega, std::true_type)
                {
                    auto x = in_out0;
                    auto y = in_out1 * omega;
                    in_out0 = x + y;
                    in_out1 = x - y;
                }

                // in: in_out0<2p, in_ou1<4p, in_out2<2p, in_ou3<4p
                // out: in_out0<2p or 4p, in_ou1<2p or 4p, in_out2<2p or 4p, in_ou3<2p or 4p
                template <bool OUT2P, typename ModIntType>
                inline void dit_butterfly2_i2424_2layer(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                        ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    dit_butterfly2_i24(in_out0, in_out1, omega_last, std::true_type{});
                    dit_butterfly2_i24(in_out2, in_out3, omega_last, std::false_type{});
                    dit_butterfly2_i24(in_out0, in_out2, omega0, std::integral_constant<bool, OUT2P>{});
                    dit_butterfly2_i24(in_out1, in_out3, omega1, std::integral_constant<bool, OUT2P>{});
                }

                // in: in_out0<2p, in_ou1<4p, in_out2<2p, in_ou3<4p
                // out: in_out0<2p, in_ou1<2p , in_out2<2p , in_ou3<2p
                template <typename ModIntType>
                inline void dit_butterfly2_2layer_out3(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                       ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    dit_butterfly2_i24(in_out0, in_out1, omega_last, std::true_type{});
                    dit_butterfly2_i24(in_out2, in_out3, omega_last, std::false_type{});
                    dit_butterfly2_i24(in_out0, in_out2, omega0, std::true_type{});
                    in_out1 = in_out1 + in_out3 * omega1;
                }

                // in: in_out0<2p, in_ou1<4p, in_out2<2p, in_ou3<4p
                // out: in_out0<2p, in_ou1<2p , in_out2<2p , in_ou3<2p
                template <typename ModIntType>
                inline void dit_butterfly2_2layer_out2(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                       ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    dit_butterfly2_i24(in_out0, in_out1, omega_last, std::true_type{});
                    dit_butterfly2_i24(in_out2, in_out3, omega_last, std::false_type{});
                    in_out0 = in_out0 + in_out2 * omega0;
                    in_out1 = in_out1 + in_out3 * omega1;
                }

                // in: in_out0<2p, in_ou1<2p
                // out: in_out0<2p, in_ou1<2p
                template <typename ModIntType>
                inline void dif_butterfly2(ModIntType &in_out0, ModIntType &in_out1, ModIntType omega)
                {
                    auto x = in_out0.add(in_out1);
                    auto y = in_out0.sub(in_out1);
                    in_out0 = x.largeNorm();
                    in_out1 = y * omega;
                }

                // in: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                // out: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                template <typename ModIntType>
                inline void dif_butterfly2_2layer(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                  ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    dif_butterfly2(in_out0, in_out2, omega0);
                    dif_butterfly2(in_out1, in_out3, omega1);
                    dif_butterfly2(in_out0, in_out1, omega_last);
                    dif_butterfly2(in_out2, in_out3, omega_last);
                }

                // in: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                // out: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                template <typename ModIntType>
                inline void dif_butterfly2_2layer_in2(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                      ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    in_out2 = in_out0 * omega0;
                    in_out3 = in_out1 * omega1;
                    dif_butterfly2(in_out0, in_out1, omega_last);
                    dif_butterfly2(in_out2, in_out3, omega_last);
                }

                // in: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                // out: in_out0<2p, in_ou1<2p, in_out2<2p, in_out3<2p
                template <typename ModIntType>
                inline void dif_butterfly2_2layer_in1(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3,
                                                      ModIntType omega0, ModIntType omega1, ModIntType omega_last)
                {
                    in_out2 = in_out0 * omega0;
                    in_out1 = in_out0 * omega_last;
                    in_out3 = in_out2 * omega_last;
                }

                template <typename ModIntType, uint32_t ROOT>
                static auto omegax8(size_t ntt_len, int factor, size_t begin = 0, bool inv = false)
                {
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    alignas(32) ModIntType w_arr[8]{};
                    ModIntType unit(qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / ntt_len * factor));
                    if (inv)
                    {
                        unit = unit.inv();
                    }
                    ModIntType w(qpow(unit, begin));
                    for (auto &&i : w_arr)
                    {
                        i = w;
                        w = w * unit;
                    }
                    return ModIntX8(w_arr);
                }

                // LEN <= MAX_LEN
                template <size_t MAX_LEN, uint32_t ROOT, typename ModIntTy>
                struct NTTShort
                {
                    static constexpr size_t NTT_LEN = MAX_LEN;
                    static constexpr int LOG_LEN = hint_log2(NTT_LEN);

                    static constexpr uint32_t mod()
                    {
                        return ModIntTy::mod();
                    }
                    static constexpr uint32_t root()
                    {
                        return ROOT;
                    }
                    static constexpr uint32_t rootInv()
                    {
                        constexpr uint32_t IROOT = mod_inv<int64_t>(ROOT, mod());
                        return IROOT;
                    }

                    using ModIntType = ModIntTy;
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    using INTT = NTTShort<MAX_LEN, rootInv(), ModIntTy>;

                    struct TableType
                    {
                        alignas(64) std::array<ModIntType, NTT_LEN> omega_table;
                        // Compute in compile time if need.
                        /*constexpr*/ TableType()
                        {
                            for (int omega_log_len = 4; omega_log_len <= LOG_LEN; omega_log_len++)
                            {
                                size_t omega_len = size_t(1) << omega_log_len, omega_count = omega_len / 2;
                                auto it = &omega_table[omega_len / 2];
                                ModIntType root = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / omega_len);
                                ModIntType omega(1);
                                size_t i = 0;
                                for (; i < 8; i++)
                                {
                                    it[i] = omega;
                                    omega *= root;
                                }
                                ModIntX8 omegaX8, rootX8 = qpow(root, 8);
                                omegaX8.loadu(&it[0]);
                                for (; i < omega_count; i += 8)
                                {
                                    omegaX8 *= rootX8;
                                    omegaX8.storeu(&it[i]);
                                }
                            }
                        }
                        constexpr ModIntType &operator[](size_t i)
                        {
                            return omega_table[i];
                        }
                        constexpr const ModIntType &operator[](size_t i) const
                        {
                            return omega_table[i];
                        }
                        constexpr const ModIntType *getOmegaIt(size_t len) const
                        {
                            return &omega_table[len / 2];
                        }
                    };

                    static const TableType table;

                    static constexpr ModIntType W_8_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                    static constexpr ModIntType W_8_2 = qpow(W_8_1, 2);
                    static constexpr ModIntType W_8_3 = qpow(W_8_1, 3);
                    static constexpr const ModIntType *W16_IT = table.getOmegaIt(16);
                    static constexpr const ModIntType *W32_IT = table.getOmegaIt(32);
                    static constexpr const ModIntType *W64_IT = table.getOmegaIt(64);

                    template <bool OUT2P = false>
                    static void ditLayer(ModIntType in_out[], size_t rank)
                    {
                        const size_t gap = rank / 4;
                        auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                        auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                        for (size_t i = 0; i < gap; i += 16)
                        {
                            // In: 2p, 4p, 2p, 4p
                            ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, omega0, omega1, omega_last;
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);
                            temp4.load(&it0[8 + i]), temp5.load(&it1[8 + i]), temp6.load(&it2[8 + i]), temp7.load(&it3[8 + i]);

                            omega0.load(&omega_it[i]), omega1.load(&omega_it[gap + i]), omega_last.load(&last_omega_it[i]);
                            dit_butterfly2_i2424_2layer<OUT2P>(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);
                            omega0.load(&omega_it[8 + i]), omega1.load(&omega_it[8 + gap + i]), omega_last.load(&last_omega_it[8 + i]);
                            dit_butterfly2_i2424_2layer<OUT2P>(temp4, temp5, temp6, temp7, omega0, omega1, omega_last);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                            temp4.store(&it0[8 + i]), temp5.store(&it1[8 + i]), temp6.store(&it2[8 + i]), temp7.store(&it3[8 + i]);
                        }
                    }

                    static void difLayer(ModIntType in_out[], size_t rank)
                    {
                        const size_t gap = rank / 4;
                        auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                        auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                        for (size_t i = 0; i < gap; i += 16)
                        {
                            ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, omega0, omega1, omega_last;
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);
                            temp4.load(&it0[8 + i]), temp5.load(&it1[8 + i]), temp6.load(&it2[8 + i]), temp7.load(&it3[8 + i]);

                            omega0.load(&omega_it[i]), omega1.load(&omega_it[gap + i]), omega_last.load(&last_omega_it[i]);
                            dif_butterfly2_2layer(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);
                            omega0.load(&omega_it[8 + i]), omega1.load(&omega_it[8 + gap + i]), omega_last.load(&last_omega_it[8 + i]);
                            dif_butterfly2_2layer(temp4, temp5, temp6, temp7, omega0, omega1, omega_last);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                            temp4.store(&it0[8 + i]), temp5.store(&it1[8 + i]), temp6.store(&it2[8 + i]), temp7.store(&it3[8 + i]);
                        }
                    }
                    static void difLayerX2(ModIntType in_out1[], ModIntType in_out2[], size_t rank)
                    {
                        const size_t gap = rank / 4;
                        auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                        auto it0 = in_out1, it1 = in_out1 + gap, it2 = in_out1 + gap * 2, it3 = in_out1 + gap * 3;
                        auto it4 = in_out2, it5 = in_out2 + gap, it6 = in_out2 + gap * 2, it7 = in_out2 + gap * 3;
                        for (size_t i = 0; i < gap; i += 8)
                        {
                            ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, omega0, omega1, omega_last;
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);
                            temp4.load(&it4[i]), temp5.load(&it5[i]), temp6.load(&it6[i]), temp7.load(&it7[i]);
                            omega0.load(&omega_it[i]), omega1.load(&omega_it[gap + i]), omega_last.load(&last_omega_it[i]);

                            dif_butterfly2_2layer(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);
                            dif_butterfly2_2layer(temp4, temp5, temp6, temp7, omega0, omega1, omega_last);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                            temp4.store(&it4[i]), temp5.store(&it5[i]), temp6.store(&it6[i]), temp7.store(&it7[i]);
                        }
                    }
                    template <bool OUT2P>
                    static void dit(ModIntType in_out[], size_t len, size_t rank)
                    {
                        len = std::min(NTT_LEN, len);
                        for (; rank < len; rank *= 4)
                        {
                            const size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            for (size_t j = 0; j < len; j += rank * 2)
                            {
                                ditLayer<true>(in_out + j, rank);
                                ditLayer<false>(in_out + j + rank, rank);
                            }
                        }
                        assert(rank == len);
                        ditLayer<OUT2P>(in_out, len);
                    }
                    static void dif(ModIntType in_out[], size_t len, size_t &rank, size_t rank_end = 128)
                    {
                        len = std::min(NTT_LEN, len);
                        for (rank = len; rank >= rank_end; rank /= 4)
                        {
                            for (size_t j = 0; j < len; j += rank)
                            {
                                difLayer(in_out + j, rank);
                            }
                        }
                    }
                    static void dif(ModIntType in_out1[], ModIntType in_out2[], size_t len, size_t &rank, size_t rank_end = 128)
                    {
                        len = std::min(NTT_LEN, len);
                        for (rank = len; rank >= rank_end; rank /= 4)
                        {
                            const size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            for (size_t j = 0; j < len; j += rank)
                            {
                                difLayerX2(in_out1 + j, in_out2 + j, rank);
                            }
                        }
                    }
                    template <bool OUT2P = false>
                    static void convolution(ModIntType in_out[], ModIntType in[], size_t len, ModIntType ntt_len_inv_r)
                    {
                        constexpr size_t L1_THRESHOLD = L1_BYTE / (2 * sizeof(ModIntType));
                        if (len <= L1_THRESHOLD)
                        {
                            convolutionL1<OUT2P>(in_out, in, len, ntt_len_inv_r);
                            return;
                        }
                        difLayerX2(in_out, in, len);
                        const size_t len_4 = len / 4;
                        convolution<true>(in_out, in, len_4, ntt_len_inv_r);
                        convolution<false>(in_out + len_4, in + len_4, len_4, ntt_len_inv_r);
                        convolution<true>(in_out + len_4 * 2, in + len_4 * 2, len_4, ntt_len_inv_r);
                        convolution<false>(in_out + len_4 * 3, in + len_4 * 3, len_4, ntt_len_inv_r);
                        INTT::template ditLayer<OUT2P>(in_out, len);
                    }
                    template <bool OUT2P>
                    static void convolutionL1(ModIntType in_out[], ModIntType in[], size_t len, ModIntType ntt_len_inv_r)
                    {
                        if (len <= 64)
                        {
                            convolutionTiny(in_out, in, len, ntt_len_inv_r);
                            if (OUT2P)
                            {
                                size_t i = 0;
                                for (const size_t rem_len = len - len % 8; i < rem_len; i += 8)
                                {
                                    ModIntX8 temp;
                                    temp.load(&in_out[i]);
                                    temp.largeNorm().store(&in_out[i]);
                                }
                                for (; i < len; ++i)
                                {
                                    in_out[i] = in_out[i].largeNorm();
                                }
                            }
                            return;
                        }
                        // len >= 128
                        size_t rank = 0;
                        dif(in_out, in, len, rank);
                        if (rank == 32)
                        {
                            for (size_t i = 0; i < len; i += 64)
                            {
                                convolution32<true>(in_out + i, in + i, ntt_len_inv_r);
                                convolution32<false>(in_out + 32 + i, in + 32 + i, ntt_len_inv_r);
                            }
                        }
                        else if (rank == 64)
                        {
                            for (size_t i = 0; i < len; i += 128)
                            {
                                convolution64<true>(in_out + i, in + i, ntt_len_inv_r);
                                convolution64<false>(in_out + 64 + i, in + 64 + i, ntt_len_inv_r);
                            }
                        }
                        else
                        {
                            assert(0);
                        }
                        INTT::template dit<OUT2P>(in_out, len, rank * 4);
                    }

                    static void convolutionTiny(ModIntType in_out[], ModIntType in[], size_t len, ModIntType ntt_len_inv_r)
                    {
                        switch (len)
                        {
                        case 1:
                            in_out[0] *= in[0] * ntt_len_inv_r;
                            break;
                        case 2:
                            convolution2(in_out, in, ntt_len_inv_r);
                            break;
                        case 4:
                            convolution4(in_out, in, ntt_len_inv_r);
                            break;
                        case 8:
                            convolution8(in_out, in, ntt_len_inv_r);
                            break;
                        case 16:
                            convolution16(in_out, in, ntt_len_inv_r);
                            break;
                        case 32:
                            convolution32<false>(in_out, in, ntt_len_inv_r);
                            break;
                        case 64:
                            convolution64<false>(in_out, in, ntt_len_inv_r);
                            break;
                        default:
                            assert(0);
                        }
                    }

                    static void convolution2(ModIntType in_out[], ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        ModIntType temp0 = in_out[0], temp1 = in_out[1];
                        ModIntType temp2 = in[0], temp3 = in[1];
                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        temp0 = temp0 * temp2 * ntt_len_inv_r;
                        temp1 = temp1 * temp3 * ntt_len_inv_r;
                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                    }

                    static void convolution4(ModIntType in_out[], ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        dif4(in_out);
                        dif4(in);
                        in_out[0] = in_out[0] * in[0] * ntt_len_inv_r;
                        in_out[1] = in_out[1] * in[1] * ntt_len_inv_r;
                        in_out[2] = in_out[2] * in[2] * ntt_len_inv_r;
                        in_out[3] = in_out[3] * in[3] * ntt_len_inv_r;
                        INTT::dit4(in_out);
                    }

                    static void convolution8(ModIntType in_out[], ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        ModIntX8 temp0, temp1, empty;
                        temp0.load(in_out), temp1.load(in);
                        dif8X2(temp0, temp1);
                        temp0 *= temp1 * ntt_len_inv_r;
                        INTT::dit8X2(temp0, empty);
                        temp0.store(in_out);
                    }

                    static void convolution16(ModIntType in_out[], ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        ModIntX8 temp0, temp1, temp2, temp3, empty;
                        dif16(in_out);
                        dif16(in);
                        temp0.load(in_out), temp1.load(in_out + 8);
                        temp2.load(in), temp3.load(in + 8);
                        temp0 *= temp2 * ntt_len_inv_r;
                        temp1 *= temp3 * ntt_len_inv_r;
                        temp0.store(in_out), temp1.store(in_out + 8);
                        INTT::dit16(in_out);
                    }

                    static void dit2(ModIntType &in_out0, ModIntType &in_out1)
                    {
                        auto x = in_out0.largeNorm();
                        auto y = in_out1.largeNorm();
                        in_out0 = x.add(y);
                        in_out1 = x.sub(y);
                    }
                    static void dif2(ModIntType &in_out0, ModIntType &in_out1)
                    {
                        transform2(in_out0, in_out1);
                    }
                    static void dit4(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0].largeNorm();
                        auto temp1 = in_out[1].largeNorm();
                        auto temp2 = in_out[2].largeNorm();
                        auto temp3 = in_out[3].largeNorm();

                        transform2(temp0, temp1);
                        auto sum = temp2.add(temp3);
                        auto dif = temp2.sub(temp3);
                        temp2 = sum.largeNorm();
                        temp3 = mul_w41<ROOT, ModIntType>(dif);

                        in_out[0] = temp0.add(temp2);
                        in_out[1] = temp1.add(temp3);
                        in_out[2] = temp0.sub(temp2);
                        in_out[3] = temp1.sub(temp3);
                    }
                    static void dif4(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp2);
                        auto sum = temp1.add(temp3);
                        auto dif = temp1.sub(temp3);
                        temp1 = sum.largeNorm();
                        temp3 = mul_w41<ROOT, ModIntType>(dif);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                    }

                    static void dit8(ModIntType in_out[])
                    {
                        ModIntX8 A, B;
                        A.load(in_out);
                        dit8X2(A, B);
                        A.store(in_out);
                    }
                    static void dif8(ModIntType in_out[])
                    {
                        ModIntX8 A, B;
                        A.load(in_out);
                        dif8X2(A, B);
                        A.store(in_out);
                    }
                    static void dit16(ModIntType in_out[])
                    {
                        ModIntX8 temp0, temp1, omega;
                        temp0.load(&in_out[0]);
                        temp1.load(&in_out[8]);
                        omega.load(W16_IT);
                        dit8X2(temp0, temp1);
                        temp0 = temp0.largeNorm();
                        temp1 = temp1 * omega;

                        temp0.add(temp1).store(&in_out[0]);
                        temp0.sub(temp1).store(&in_out[8]);
                    }
                    static void dif16(ModIntType in_out[])
                    {
                        ModIntX8 temp0, temp1, sum, dif, omega;
                        temp0.load(&in_out[0]);
                        temp1.load(&in_out[8]);
                        omega.load(W16_IT);
                        sum = temp0.add(temp1);
                        dif = temp0.sub(temp1);
                        temp0 = sum.largeNorm();
                        temp1 = dif * omega;
                        dif8X2(temp0, temp1);

                        temp0.store(&in_out[0]);
                        temp1.store(&in_out[8]);
                    }

                    static ModIntX8 dit2X4(ModIntX8 in)
                    {
                        // in = in.largeNorm();
                        ModIntX8 lo = in.lshift32In64(); // 0, a
                        ModIntX8 hi = in.rshift32In64(); // b, 0
                        lo = lo.sub(in);                 // X, a - b + mod2
                        hi = hi.add(in);                 // a + b ,X
                        return ModIntX8::cross32(hi, lo);
                    }

                    static ModIntX8 dif2X4(ModIntX8 in)
                    {
                        ModIntX8 lo = in.lshift32In64(); // 0, a
                        ModIntX8 hi = in.rshift32In64(); // b, 0
                        lo = lo.sub(in);                 // X, a - b + mod2
                        hi = hi.add(in);                 // a + b ,X
                        return ModIntX8::cross32(hi, lo).largeNorm();
                    }

                    static void dit4X4(ModIntX8 &A, ModIntX8 &B)
                    {
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_2, ModIntType(1), W_8_2, ModIntType(1), W_8_2, ModIntType(1), W_8_2};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp0 = dit2X4(A); // A0,A1,A2,A3,A4,A5,A6,A7
                        temp1 = dit2X4(B); // B0,B1,B2,B3,B4,B5,B6,B7

                        omega.load(w_arr);
                        temp2 = temp0.rshift64In128(); // A2,A3,X,X,A6,A7,X,X
                        temp3 = temp1.lshift64In128(); // X,X,B0,B1,X,X,B4,B5

                        temp0 = ModIntX8::cross64(temp0, temp3); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp1 = ModIntX8::cross64(temp2, temp1); // A2,A3,B2,B3,A6,A7,B6,B7

                        temp0 = temp0.largeNorm();
                        temp1 = temp1 * omega; // (A2,A3,B2,B3,A6,A7,B6,B7)*w

                        temp2 = temp0.add(temp1); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp3 = temp0.sub(temp1); // A2,A3,B2,B3,A6,A7,B6,B7

                        temp0 = temp2.rshift64In128(); // B0,B1,X,X,B4,B5,X,X
                        temp1 = temp3.lshift64In128(); // X,X,A2,A3,X,X,A6,A7

                        A = ModIntX8::cross64(temp2, temp1); // A0,A1,A2,A3,A4,A5,A6,A7
                        B = ModIntX8::cross64(temp0, temp3); // B0,B1,B2,B3,B4,B5,B6,B7
                    }
                    static void dif4X4(ModIntX8 &A, ModIntX8 &B)
                    {

                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_2, ModIntType(1), W_8_2, ModIntType(1), W_8_2, ModIntType(1), W_8_2};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp2 = A.rshift64In128(); // A2,A3,X,X,A6,A7,X,X
                        temp3 = B.lshift64In128(); // X,X,B0,B1,X,X,B4,B5

                        omega.load(w_arr);
                        temp0 = ModIntX8::cross64(A, temp3); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp1 = ModIntX8::cross64(temp2, B); // A2,A3,B2,B3,A6,A7,B6,B7

                        temp2 = temp0.add(temp1); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp3 = temp0.sub(temp1);
                        temp2 = temp2.largeNorm();
                        temp3 = temp3 * omega; // (A2,A3,B2,B3,A6,A7,B6,B7)*w

                        temp0 = temp2.rshift64In128(); // B0,B1,X,X,B4,B5,X,X
                        temp1 = temp3.lshift64In128(); // X,X,A2,A3,X,X,A6,A7

                        temp2 = ModIntX8::cross64(temp2, temp1); // A0,A1,A2,A3,A4,A5,A6,A7
                        temp3 = ModIntX8::cross64(temp0, temp3); // B0,B1,B2,B3,B4,B5,B6,B7

                        A = dif2X4(temp2); // A
                        B = dif2X4(temp3); // B
                    }

                    static void dit8X2(ModIntX8 &A, ModIntX8 &B)
                    {
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_1, W_8_2, W_8_3, ModIntType(1), W_8_1, W_8_2, W_8_3};

                        dit4X4(A, B); // A0,A1,A2,A3,A4,A5,A6,A7; B0,B1,B2,B3,B4,B5,B6,B7
                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        omega.load(w_arr);
                        temp0 = ModIntX8::packLo128(A, B); // A0,A1,A2,A3,B0,B1,B2,B3
                        temp1 = ModIntX8::packHi128(A, B); // A4,A5,A6,A7,B4,B5,B6,B7
                        temp0 = temp0.largeNorm();
                        temp1 = temp1 * omega;                 // (A4,A5,A6,A7,B4,B5,B6,B7)*w
                        temp2 = temp0.add(temp1);              // A0,A1,A2,A3,B0,B1,B2,B3
                        temp3 = temp0.sub(temp1);              // A4,A5,A6,A7,B4,B5,B6,B7
                        A = ModIntX8::packLo128(temp2, temp3); // A0,A1,A2,A3,A4,A5,A6,A7
                        B = ModIntX8::packHi128(temp2, temp3); // B0,B1,B2,B3,B4,B5,B6,B7
                    }
                    static void dif8X2(ModIntX8 &A, ModIntX8 &B)
                    {
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_1, W_8_2, W_8_3, ModIntType(1), W_8_1, W_8_2, W_8_3};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp0 = ModIntX8::packLo128(A, B); // A0,A1,A2,A3,B0,B1,B2,B3
                        temp1 = ModIntX8::packHi128(A, B); // A4,A5,A6,A7,B4,B5,B6,B7
                        omega.load(w_arr);
                        temp2 = temp0.add(temp1); // A0,A1,A2,A3,B0,B1,B2,B3
                        temp3 = temp0.sub(temp1); // A4,A5,A6,A7,B4,B5,B6,B7
                        temp2 = temp2.largeNorm();
                        temp3 = temp3 * omega;                 //(A4,A5,A6,A7,B4,B5,B6,B7)*w
                        A = ModIntX8::packLo128(temp2, temp3); // A0,A1,A2,A3,A4,A5,A6,A7
                        B = ModIntX8::packHi128(temp2, temp3); // B0,B1,B2,B3,B4,B5,B6,B7
                        dif4X4(A, B);
                    }
                    static void dit8X2(ModIntType in_out[])
                    {
                        ModIntX8 temp0, temp1;
                        temp0.load(in_out), temp1.load(in_out + 8);
                        dit8X2(temp0, temp1);
                        temp0.store(in_out), temp1.store(in_out + 8);
                    }
                    static void dif8X2(ModIntType in_out[])
                    {
                        ModIntX8 temp0, temp1;
                        temp0.load(in_out), temp1.load(in_out + 8);
                        dif8X2(temp0, temp1);
                        temp0.store(in_out), temp1.store(in_out + 8);
                    }

                    template <bool OUT2P>
                    static void convolution32(ModIntType in_out[], const ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, omega;
                        temp0.load(&in_out[0]), temp1.load(&in_out[8]), temp2.load(&in_out[16]), temp3.load(&in_out[24]);
                        temp4.load(&in[0]), temp5.load(&in[8]), temp6.load(&in[16]), temp7.load(&in[24]);
                        // DIF32X2
                        omega.load(W32_IT);
                        dif_butterfly2(temp0, temp2, omega);
                        dif_butterfly2(temp4, temp6, omega);
                        omega.load(W32_IT + 8);
                        dif_butterfly2(temp1, temp3, omega);
                        dif_butterfly2(temp5, temp7, omega);
                        // DIF16X4
                        omega.load(W16_IT);
                        dif_butterfly2(temp0, temp1, omega);
                        dif_butterfly2(temp2, temp3, omega);
                        dif_butterfly2(temp4, temp5, omega);
                        dif_butterfly2(temp6, temp7, omega);
                        // DIF8X8
                        transpose8x8(temp0.data, temp1.data, temp2.data, temp3.data, temp4.data, temp5.data, temp6.data, temp7.data);

                        omega = ModIntX8(W_8_2);
                        transform2(temp0, temp4);
                        dif_butterfly2(temp1, temp5, ModIntX8(W_8_1));
                        dif_butterfly2(temp2, temp6, omega);
                        dif_butterfly2(temp3, temp7, ModIntX8(W_8_3));

                        transform2(temp0, temp2);
                        dif_butterfly2(temp1, temp3, omega);
                        transform2(temp4, temp6);
                        dif_butterfly2(temp5, temp7, omega);

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        transform2(temp4, temp5);
                        transform2(temp6, temp7);

                        transpose8x8(temp0.data, temp1.data, temp2.data, temp3.data, temp4.data, temp5.data, temp6.data, temp7.data);
                        // DOT MUL
                        omega = ModIntX8(ntt_len_inv_r);
                        temp0 *= (temp4 * omega), temp1 *= (temp5 * omega);
                        temp2 *= (temp6 * omega), temp3 *= (temp7 * omega);
                        // DIT8X4
                        INTT::dit8X2(temp0, temp1);
                        INTT::dit8X2(temp2, temp3);
                        // DIT16X2
                        omega.load(INTT::W16_IT);
                        temp0 = temp0.largeNorm();
                        dit_butterfly2_i24(temp0, temp1, omega, std::true_type{});
                        temp2 = temp2.largeNorm();
                        dit_butterfly2_i24(temp2, temp3, omega, std::false_type{});
                        // DIT32
                        using Out22Ty = std::integral_constant<bool, OUT2P>;
                        omega.load(INTT::W32_IT);
                        dit_butterfly2_i24(temp0, temp2, omega, Out22Ty{});
                        omega.load(INTT::W32_IT + 8);
                        dit_butterfly2_i24(temp1, temp3, omega, Out22Ty{});

                        temp0.store(&in_out[0]), temp1.store(&in_out[8]), temp2.store(&in_out[16]), temp3.store(&in_out[24]);
                    }
                    template <bool OUT2P>
                    static void convolution64(ModIntType in_out[], ModIntType in[], ModIntType ntt_len_inv_r)
                    {
                        ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, omega;
                        temp0.load(&in_out[0]), temp1.load(&in_out[8]), temp2.load(&in_out[32]), temp3.load(&in_out[40]);
                        temp4.load(&in[0]), temp5.load(&in[8]), temp6.load(&in[32]), temp7.load(&in[40]);
                        omega.load(W64_IT);
                        dif_butterfly2(temp0, temp2, omega);
                        dif_butterfly2(temp4, temp6, omega);
                        omega.load(W64_IT + 8);
                        dif_butterfly2(temp1, temp3, omega);
                        dif_butterfly2(temp5, temp7, omega);
                        temp0.store(&in_out[0]), temp1.store(&in_out[8]), temp2.store(&in_out[32]), temp3.store(&in_out[40]);
                        temp4.store(&in[0]), temp5.store(&in[8]), temp6.store(&in[32]), temp7.store(&in[40]);
                        temp0.load(&in_out[16]), temp1.load(&in_out[24]), temp2.load(&in_out[48]), temp3.load(&in_out[56]);
                        temp4.load(&in[16]), temp5.load(&in[24]), temp6.load(&in[48]), temp7.load(&in[56]);
                        omega.load(W64_IT + 16);
                        dif_butterfly2(temp0, temp2, omega);
                        dif_butterfly2(temp4, temp6, omega);
                        omega.load(W64_IT + 24);
                        dif_butterfly2(temp1, temp3, omega);
                        dif_butterfly2(temp5, temp7, omega);
                        temp0.store(&in_out[16]), temp1.store(&in_out[24]), temp2.store(&in_out[48]), temp3.store(&in_out[56]);
                        temp4.store(&in[16]), temp5.store(&in[24]), temp6.store(&in[48]), temp7.store(&in[56]);
                        convolution32<true>(in_out, in, ntt_len_inv_r);
                        convolution32<false>(in_out + 32, in + 32, ntt_len_inv_r);
                        temp0.load(&in_out[0]), temp1.load(&in_out[8]), temp2.load(&in_out[16]), temp3.load(&in_out[24]);
                        temp4.load(&in_out[32]), temp5.load(&in_out[40]), temp6.load(&in_out[48]), temp7.load(&in_out[56]);
                        using Out22Ty = std::integral_constant<bool, OUT2P>;
                        omega.load(INTT::W64_IT);
                        dit_butterfly2_i24(temp0, temp4, omega, Out22Ty{});
                        omega.load(INTT::W64_IT + 8);
                        dit_butterfly2_i24(temp1, temp5, omega, Out22Ty{});
                        omega.load(INTT::W64_IT + 16);
                        dit_butterfly2_i24(temp2, temp6, omega, Out22Ty{});
                        omega.load(INTT::W64_IT + 24);
                        dit_butterfly2_i24(temp3, temp7, omega, Out22Ty{});
                        temp0.store(&in_out[0]), temp1.store(&in_out[8]), temp2.store(&in_out[16]), temp3.store(&in_out[24]);
                        temp4.store(&in_out[32]), temp5.store(&in_out[40]), temp6.store(&in_out[48]), temp7.store(&in_out[56]);
                    }
                };
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                const typename NTTShort<LEN, ROOT, ModIntType>::TableType NTTShort<LEN, ROOT, ModIntType>::table;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr size_t NTTShort<LEN, ROOT, ModIntType>::NTT_LEN;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr int NTTShort<LEN, ROOT, ModIntType>::LOG_LEN;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr ModIntType NTTShort<LEN, ROOT, ModIntType>::W_8_1;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr ModIntType NTTShort<LEN, ROOT, ModIntType>::W_8_2;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr ModIntType NTTShort<LEN, ROOT, ModIntType>::W_8_3;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr const ModIntType *NTTShort<LEN, ROOT, ModIntType>::W16_IT;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr const ModIntType *NTTShort<LEN, ROOT, ModIntType>::W32_IT;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr const ModIntType *NTTShort<LEN, ROOT, ModIntType>::W64_IT;

                template <uint32_t MOD, uint32_t ROOT>
                struct NTT
                {
                    static constexpr uint32_t mod()
                    {
                        return MOD;
                    }
                    static constexpr uint32_t root()
                    {
                        return ROOT;
                    }
                    static constexpr uint32_t rootInv()
                    {
                        constexpr uint32_t IROOT = mod_inv<int64_t>(ROOT, MOD);
                        return IROOT;
                    }

                    static_assert(root() < mod(), "ROOT must be smaller than MOD");
                    static_assert(check_inv<uint64_t>(root(), rootInv(), mod()), "IROOT * ROOT % MOD must be 1");
                    static constexpr int MOD_BITS = hint_log2(mod()) + 1;
                    static constexpr int MAX_LOG_LEN = hint_ctz(mod() - 1);

                    static constexpr size_t getMaxLen()
                    {
                        if (MAX_LOG_LEN < sizeof(size_t) * CHAR_BIT)
                        {
                            return size_t(1) << MAX_LOG_LEN;
                        }
                        return size_t(1) << (sizeof(size_t) * CHAR_BIT - 1);
                    }
                    static constexpr size_t NTT_MAX_LEN = getMaxLen();

                    using INTT = NTT<mod(), rootInv()>;
                    using ModIntType = MontInt32Lazy<MOD>;
                    using ModIntX8 = MontInt32X8<ModIntType>;

                    static constexpr size_t LONG_THRESHOLD = std::min(L2_BYTE / (2 * sizeof(ModIntType)), NTT_MAX_LEN);
                    using NTTTemplate = NTTShort<LONG_THRESHOLD, root(), ModIntType>;

                    static ModIntX8 unitx8(size_t ntt_len, int factor, uint32_t root_in = root())
                    {
                        return ModIntX8(qpow(ModIntType(root_in), (mod() - 1) / ntt_len * factor * 8));
                    }
                    static ModIntX8 omegax8(size_t ntt_len, int factor, uint32_t root_in = root())
                    {
                        alignas(32) ModIntType w_arr[8]{};
                        ModIntType w(1), unit(qpow(ModIntType(root_in), (mod() - 1) / ntt_len * factor));
                        for (auto &&i : w_arr)
                        {
                            i = w;
                            w = w * unit;
                        }
                        return ModIntX8(w_arr);
                    }

                    static void ditOutLayer(uint32_t out[], size_t len, const ModIntType in[], size_t ntt_len)
                    {
                        assert(len > ntt_len / 2);
                        ModIntX8 omega0 = omegax8(ntt_len, 1, root());
                        ModIntX8 omega1 = mul_w41<root(), ModIntType>(omega0);
                        ModIntX8 omega_last = omega0 * omega0;
                        ModIntX8 unit1X8 = unitx8(ntt_len, 1, root()), unit_last = unit1X8 * unit1X8;
                        ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                        size_t len_4 = ntt_len / 4;
                        auto it0 = in, it1 = in + len_4, it2 = in + len_4 * 2, it3 = in + len_4 * 3;
                        auto it4 = out, it5 = out + len_4, it6 = out + len_4 * 2, it7 = out + len_4 * 3;
                        if (len > len_4 * 3)
                        {
                            const size_t len1 = len - len_4 * 3, rem_len = len1 - len1 % 8;
                            size_t i = 0;
                            for (; i < rem_len; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_i2424_2layer<true>(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm(), temp3 = temp3.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]), temp2.storeu(&it6[i]), temp3.storeu(&it7[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                            if (len1 % 8 > 0)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_i2424_2layer<true>(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm(), temp3 = temp3.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]), temp2.storeu(&it6[i]), temp3.storeN(&it7[i], len1 % 8);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                                i += 8;
                            }
                            for (; i < len_4; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_2layer_out3(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]), temp2.storeu(&it6[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                        }
                        else
                        {
                            const size_t len1 = len - len_4 * 2, rem_len = len1 - len1 % 8;
                            size_t i = 0;
                            for (; i < rem_len; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_2layer_out3(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]), temp2.storeu(&it6[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                            if (len1 % 8 > 0)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_2layer_out3(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]), temp2.storeN(&it6[i], len1 % 8);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                                i += 8;
                            }
                            for (; i < len_4; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_2layer_out2(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0 = temp0.norm(), temp1 = temp1.norm();
                                temp0.storeu(&it4[i]), temp1.storeu(&it5[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                        }
                    }
                    static void difInLayer(const uint32_t in[], size_t len, ModIntType out[], size_t ntt_len)
                    {
                        assert(ntt_len / 4 < len && len <= ntt_len / 2);
                        ModIntX8 omega0 = omegax8(ntt_len, 1, root());
                        ModIntX8 omega1 = mul_w41<root(), ModIntType>(omega0);
                        ModIntX8 omega_last = omega0 * omega0;
                        ModIntX8 unit1X8 = unitx8(ntt_len, 1, root()), unit_last = unit1X8 * unit1X8;
                        ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                        size_t len_4 = ntt_len / 4;
                        auto it0 = in, it1 = in + len_4, it2 = in + len_4 * 2, it3 = in + len_4 * 3;
                        auto it4 = out, it5 = out + len_4, it6 = out + len_4 * 2, it7 = out + len_4 * 3;
                        const size_t len1 = len - len_4, rem_len = len1 - len1 % 8;
                        size_t i = 0;
                        for (; i < rem_len; i += 8)
                        {
                            temp0.loadu(&it0[i]), temp1.loadu(&it1[i]);

                            dif_butterfly2_2layer_in2(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                            temp0.store(&it4[i]), temp1.store(&it5[i]), temp2.store(&it6[i]), temp3.store(&it7[i]);
                            omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                        }
                        if (len1 % 8 > 0)
                        {
                            temp0.loadu(&it0[i]), temp1.loadN(&it1[i], len1 % 8);

                            dif_butterfly2_2layer_in2(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                            temp0.store(&it4[i]), temp1.store(&it5[i]), temp2.store(&it6[i]), temp3.store(&it7[i]);
                            omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            i += 8;
                        }
                        for (; i < len_4; i += 8)
                        {
                            temp0.loadu(&it0[i]);

                            dif_butterfly2_2layer_in1(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                            temp0.store(&it4[i]), temp1.store(&it5[i]), temp2.store(&it6[i]), temp3.store(&it7[i]);
                            omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                        }
                    }

                    // Inner recursion
                    template <bool OUT2P = true>
                    static void convolutionRecursion(ModIntType in_out1[], ModIntType in_out2[], size_t ntt_len, ModIntType len_inv_r, bool norm = false)
                    {
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::template convolution<OUT2P>(in_out1, in_out2, ntt_len, len_inv_r);
                            if (norm)
                            {
                                for (size_t i = 0; i < ntt_len; i++)
                                {
                                    in_out1[i] = in_out1[i].norm();
                                }
                            }
                            return;
                        }
                        ModIntX8 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                        ModIntX8 omega0 = omegax8(ntt_len, 1, root());
                        ModIntX8 omega1 = mul_w41<root(), ModIntType>(omega0);
                        ModIntX8 omega_last = omega0 * omega0;
                        ModIntX8 unit1X8 = unitx8(ntt_len, 1, root()), unit_last = unit1X8 * unit1X8;
                        size_t len_4 = ntt_len / 4;
                        auto it0 = in_out1, it1 = in_out1 + len_4, it2 = in_out1 + len_4 * 2, it3 = in_out1 + len_4 * 3;
                        auto it4 = in_out2, it5 = in_out2 + len_4, it6 = in_out2 + len_4 * 2, it7 = in_out2 + len_4 * 3;
                        for (size_t i = 0; i < len_4; i += 8)
                        {
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);
                            temp4.load(&it4[i]), temp5.load(&it5[i]), temp6.load(&it6[i]), temp7.load(&it7[i]);

                            dif_butterfly2_2layer(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);
                            dif_butterfly2_2layer(temp4, temp5, temp6, temp7, omega0, omega1, omega_last);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                            temp4.store(&it4[i]), temp5.store(&it5[i]), temp6.store(&it6[i]), temp7.store(&it7[i]);
                            omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                        }
                        convolutionRecursion<true>(it0, it4, len_4, len_inv_r);
                        convolutionRecursion<false>(it1, it5, len_4, len_inv_r);
                        convolutionRecursion<true>(it2, it6, len_4, len_inv_r);
                        convolutionRecursion<false>(it3, it7, len_4, len_inv_r);
                        omega0 = omegax8(ntt_len, 1, rootInv());
                        omega1 = mul_w41<rootInv(), ModIntType>(omega0);
                        omega_last = omega0 * omega0;
                        unit1X8 = unitx8(ntt_len, 1, rootInv());
                        unit_last = unit1X8 * unit1X8;
                        if (norm)
                        {
                            for (size_t i = 0; i < len_4; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_i2424_2layer<true>(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);
                                temp0 = temp0.norm(), temp1 = temp1.norm(), temp2 = temp2.norm(), temp3 = temp3.norm();

                                temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < len_4; i += 8)
                            {
                                temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                                dit_butterfly2_i2424_2layer<OUT2P>(temp0, temp1, temp2, temp3, omega0, omega1, omega_last);

                                temp0.store(&it0[i]), temp1.store(&it1[i]), temp2.store(&it2[i]), temp3.store(&it3[i]);
                                omega0 *= unit1X8, omega1 *= unit1X8, omega_last *= unit_last;
                            }
                        }
                    }

                    // Outer recursion
                    static void convolution(const uint32_t in1[], size_t len1, const uint32_t in2[], size_t len2, uint32_t out[])
                    {
                        size_t conv_len = len1 + len2 - 1, ntt_len = int_ceil2(conv_len);
                        const ModIntType len_inv_r = ModIntType(ntt_len).inv().mulR();
                        auto ntt_p1 = reinterpret_cast<ModIntType *>(_mm_malloc(sizeof(ModIntType) * ntt_len, 32));
                        auto ntt_p2 = reinterpret_cast<ModIntType *>(_mm_malloc(sizeof(ModIntType) * ntt_len, 32));
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            std::memcpy(ntt_p1, in1, sizeof(ModIntType) * len1);
                            std::memcpy(ntt_p2, in2, sizeof(ModIntType) * len2);
                            std::memset(ntt_p1 + len1, 0, sizeof(ModIntType) * (ntt_len - len1));
                            std::memset(ntt_p2 + len2, 0, sizeof(ModIntType) * (ntt_len - len2));
                            convolutionRecursion<true>(ntt_p1, ntt_p2, ntt_len, len_inv_r);
                            size_t i = 0;
                            for (const size_t rem_len = conv_len - conv_len % 8; i < rem_len; i += 8)
                            {
                                ModIntX8 temp0;
                                temp0.load(&ntt_p1[i]);
                                temp0.norm().storeu(&out[i]);
                            }
                            for (; i < conv_len; i++)
                            {
                                ModIntType temp = ntt_p1[i];
                                out[i] = temp.norm().data;
                            }
                            _mm_free(ntt_p1);
                            _mm_free(ntt_p2);
                            return;
                        }
                        difInLayer(in1, len1, ntt_p1, ntt_len);
                        difInLayer(in2, len2, ntt_p2, ntt_len);
                        size_t len_4 = ntt_len / 4;
                        convolutionRecursion<true>(ntt_p1, ntt_p2, len_4, len_inv_r);
                        convolutionRecursion<false>(ntt_p1 + len_4, ntt_p2 + len_4, len_4, len_inv_r);
                        convolutionRecursion<true>(ntt_p1 + len_4 * 2, ntt_p2 + len_4 * 2, len_4, len_inv_r);
                        convolutionRecursion<false>(ntt_p1 + len_4 * 3, ntt_p2 + len_4 * 3, len_4, len_inv_r);
                        INTT::ditOutLayer(out, conv_len, ntt_p1, ntt_len);
                        _mm_free(ntt_p1);
                        _mm_free(ntt_p2);
                    }
                };
                template <uint32_t MOD, uint32_t ROOT>
                constexpr int NTT<MOD, ROOT>::MOD_BITS;
                template <uint32_t MOD, uint32_t ROOT>
                constexpr int NTT<MOD, ROOT>::MAX_LOG_LEN;
                template <uint32_t MOD, uint32_t ROOT>
                constexpr size_t NTT<MOD, ROOT>::NTT_MAX_LEN;
                template <uint32_t MOD, uint32_t ROOT>
                constexpr size_t NTT<MOD, ROOT>::LONG_THRESHOLD;
            }
        }
    }
}

void poly_multiply(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    using NTT = hint::transform::ntt::radix2_avx::NTT<998244353, 3>;
    NTT::convolution(a, n + 1, b, m + 1, c);
}

void poly_multiply2(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    using namespace hint;
    using NTT = hint::transform::ntt::radix2_avx::NTT<998244353, 3>;
    using ModInt = NTT::ModIntType;
    size_t len1 = n + 1, len2 = m + 1;
    size_t conv_len = len1 + len2 - 1, ntt_len = int_ceil2(conv_len);
    const ModInt len_inv_r = ModInt(ntt_len).inv().mulR();
    auto ntt_p1 = reinterpret_cast<ModInt *>(_mm_malloc(sizeof(ModInt) * ntt_len, 32));
    auto ntt_p2 = reinterpret_cast<ModInt *>(_mm_malloc(sizeof(ModInt) * ntt_len, 32));
    std::memcpy(ntt_p1, a, sizeof(ModInt) * len1);
    std::memcpy(ntt_p2, b, sizeof(ModInt) * len2);
    std::memset(ntt_p1 + len1, 0, sizeof(ModInt) * (ntt_len - len1));
    std::memset(ntt_p2 + len2, 0, sizeof(ModInt) * (ntt_len - len2));
    NTT::convolutionRecursion<true>(ntt_p1, ntt_p2, ntt_len, len_inv_r, true);
    std::memcpy(c, ntt_p1, sizeof(ModInt) * conv_len);
    _mm_free(ntt_p1);
    _mm_free(ntt_p2);
}

template <typename ModIntType, typename ModIntX8>
void arrToInt(const ModIntType in[], size_t n, unsigned out[])
{
    static_assert(sizeof(ModIntType) == 4, "ModIntType must be 4 bytes");
    static_assert(sizeof(unsigned) == 4, "unsigned must be 4 bytes");
    size_t i = 0, rem_len = n - n % 16;
    for (; i < rem_len; i += 16)
    {
        ModIntX8 temp0, temp1;
        temp0.load(&in[i]), temp1.load(&in[8 + i]);
        temp0.data = temp0.toInt();
        temp1.data = temp1.toInt();
        temp0.storeu(&out[i]), temp1.storeu(&out[8 + i]);
    }
    for (; i < n; i++)
    {
        out[i] = uint32_t(in[i]);
    }
}

template <typename ModIntType, typename ModIntX8>
void arrToMont(const unsigned in[], size_t n, ModIntType out[])
{
    static_assert(sizeof(ModIntType) == 4, "ModIntType must be 4 bytes");
    static_assert(sizeof(unsigned) == 4, "unsigned must be 4 bytes");
    size_t i = 0, rem_len = n - n % 16;
    for (; i < rem_len; i += 16)
    {
        ModIntX8 temp0, temp1;
        temp0.loadu(&in[i]), temp1.loadu(&in[8 + i]);
        temp0 = ModIntX8(temp0.data);
        temp1 = ModIntX8(temp1.data);
        temp0.store(&out[i]), temp1.store(&out[8 + i]);
    }
    for (; i < n; i++)
    {
        out[i] = ModIntType(in[i]);
    }
}

#include <chrono>

std::fstream fout("datab1.txt", std::ios::out);
template <uint32_t ROOT, typename ModInt>
void ntt_dit(ModInt in_out[], size_t ntt_len)
{
    for (size_t rank = 2; rank <= ntt_len; rank *= 2)
    {
        ModInt unit_omega = hint::qpow(ModInt(ROOT), (ModInt::mod() - 1) / rank);
        size_t dis = rank / 2;
        for (auto begin = in_out; begin < in_out + ntt_len; begin += rank)
        {
            ModInt omega = 1;
            for (auto p = begin; p < begin + dis; p++)
            {
                auto temp0 = p[0], temp1 = p[dis] * omega;
                p[0] = temp0 + temp1;
                p[dis] = temp0 - temp1;
                omega = omega * unit_omega;
            }
        }
        // if (rank == ntt_len / 4)
        // {
        //     for (size_t i = ntt_len / 4; i < ntt_len / 2; i++)
        //     {
        //         fout << in_out[i].norm().data << ' ';
        //         if (i % 8 == 7)
        //         {
        //             fout << '\n';
        //         }
        //     }
        //     fout << std::endl;
        // }
    }
}
template <uint32_t ROOT, typename ModInt>
void ntt_dif(ModInt in_out[], size_t ntt_len)
{
    for (size_t rank = ntt_len; rank >= 2; rank /= 2)
    {
        ModInt unit_omega = hint::qpow(ModInt(ROOT), (ModInt::mod() - 1) / rank);
        size_t dis = rank / 2;
        // if (rank == ntt_len / 4)
        // {
        //     for (size_t i = ntt_len / 4; i < ntt_len / 2; i++)
        //     {
        //         fout << in_out[i].norm().data << ' ';
        //         if (i % 8 == 7)
        //         {
        //             fout << '\n';
        //         }
        //     }
        //     fout << std::endl;
        // }
        for (auto begin = in_out; begin < in_out + ntt_len; begin += rank)
        {
            ModInt omega = 1;
            for (auto p = begin; p < begin + dis; p++)
            {
                auto temp0 = p[0], temp1 = p[dis];
                p[0] = temp0 + temp1;
                p[dis] = (temp0 - temp1) * omega;
                omega = omega * unit_omega;
            }
        }
    }
}

void poly_multiply1(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    using ModInt = hint::modint::MontInt32Lazy<998244353>;
    using namespace hint;
    size_t conv_len = n + m + 1, ntt_len = int_ceil2(conv_len);
    auto p1 = new ModInt[ntt_len];
    auto p2 = new ModInt[ntt_len];
    std::cout << ntt_len << std::endl;
    memcpy(p1, a, sizeof(ModInt) * (n + 1));
    memcpy(p2, b, sizeof(ModInt) * (m + 1));
    memset(p1 + n + 1, 0, (ntt_len - n - 1) * sizeof(ModInt));
    memset(p2 + m + 1, 0, (ntt_len - m - 1) * sizeof(ModInt));
    ntt_dif<3, ModInt>(p1, ntt_len);
    ntt_dif<3, ModInt>(p2, ntt_len);
    ModInt inv_r = ModInt(ntt_len).inv().mulR();
    for (size_t i = 0; i < ntt_len; ++i)
    {
        p1[i] = p1[i] * p2[i] * inv_r;
    }
    ntt_dit<hint::mod_inv<int64_t>(3, 998244353), ModInt>(p1, ntt_len);
    for (size_t i = 0; i < conv_len; i++)
    {
        c[i] = p1[i].norm().data;
    }
    delete[] p1;
    delete[] p2;
}

void ntt_check(int s = 24)
{
    using namespace hint;
    using namespace transform;
    using namespace ntt;
    constexpr uint64_t mod1 = 1945555039024054273, root1 = 5;
    constexpr uint64_t mod2 = 4179340454199820289, root2 = 3;
    constexpr uint64_t mod = 998244353, root = 3;

    using ModInt = MontInt32Lazy<mod>;
    using ModIntX8 = MontInt32X8<ModInt>;
    using ntt = radix2_avx::NTT<mod, root>;

    std::cin >> s;
    size_t len = 1 << s;
    size_t times = 1000; // std::max<size_t>(1, (1 << 25) / len);
    alignas(32) static std::array<ModInt, 1 << 23> a;
    alignas(32) static std::array<ModInt, 1 << 23> b;

    std::vector<uint32_t> c(len);
    for (size_t i = 0; i < len; i++)
    {
        a[i] = uint64_t(i);
        b[i] = uint64_t(i);
        c[i] = uint64_t(i);
    }

    auto t1 = std::chrono::steady_clock::now();
    // for (size_t i = 0; i < times; i++)
    {
        ntt_dif<root>(a.data(), len);
        ntt_dif<root>(a.data(), len);
        ntt_dit<root>(a.data(), len);
    }
    auto t2 = std::chrono::steady_clock::now();
    // for (size_t i = 0; i < times; i++)
    {
        // ntt::dif244(b.data(), len);
        // ntt::dif244(b.data(), len);
        // ntt::dit244(b.data(), len);
    }
    auto t3 = std::chrono::steady_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    auto time2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
    for (size_t i = 0; i < len; i++)
    {
        if (uint64_t(a[i]) != uint64_t(b[i]))
        {
            std::cout << i << ":\t" << uint64_t(a[i]) << "\t" << uint64_t(b[i]) << "\n";
            return;
        }
    }
    std::cout << len << "\n";
    std::cout << time1 << "\t" << time2 << "\t" << time1 / time2 << "\n";
}

template <typename T>
void result_test(const std::vector<T> &res, uint64_t ele)
{
    size_t len = res.size();
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << (i + 1) * ele * ele << "\t" << y << "\n";
            return;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    std::cout << "success\n";
}
void test_ntt()
{
    int n = 18;
    std::cin >> n;
    size_t len = 1 << n; // 变换长度
    uint64_t ele = 5;
    std::vector<uint32_t> in1(len / 2, ele);
    std::vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    auto t1 = std::chrono::steady_clock::now();
    std::vector<uint32_t> res; //= poly_multiply(in1, in2);
    auto t2 = std::chrono::steady_clock::now();
    result_test<uint32_t>(res, ele); // 结果校验
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}

void test_DITX8()
{
    using NTT = hint::transform::ntt::radix2_avx::NTT<998244353, 3>;
    using ModInt = NTT::ModIntType;
    alignas(32) ModInt in[64] = {0};
    for (uint32_t i = 0; i < 64; ++i)
    {
        in[i].data = i;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < 100000000; i++)
    {
        NTT::ModIntX8 t0, t1;
        t0.load(in);
        t1.load(in + 8);
        NTT::NTTTemplate::dit8X2(t0, t1);
        t0.store(in);
        t1.store(in + 8);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < 64; ++i)
    {
        std::cout << in[i].data % 998244353 << " ";
        if (i % 8 == 7)
        {
            std::cout << "\n";
        }
    }
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
}
void check_ntt()
{
    using namespace std;
    std::ios::sync_with_stdio(false);
    cout.tie(nullptr);
    cin.tie(nullptr);
    constexpr size_t len1 = 1 << 22, len2 = len1;
    static unsigned a[len1];
    static unsigned b[len2];
    static unsigned c[len1 + len2 - 1]{};
    static unsigned d[len1 + len2 - 1]{};
    int n1 = 9, n2 = 8;
    for (auto &&i : a)
    {
        i = 998244352;
    }
    for (auto &&i : b)
    {
        i = 998244351;
    }
    auto t1 = chrono::steady_clock::now();
    // for (size_t i = 0; i < 1000; i++)
    {
        poly_multiply(a, len1 - 1, b, len2 - 1, c);
    }
    auto t2 = chrono::steady_clock::now();
    // for (size_t i = 0; i < 1000; i++)
    {
        poly_multiply1(a, len1 - 1, b, len2 - 1, d);
    }
    auto t3 = chrono::steady_clock::now();
    for (size_t i = 0; i < len1 + len2 - 1; i++)
    {
        if (c[i] != d[i])
        {
            std::cout << i << " " << c[i] << " " << d[i] << "\n";
            break;
        }
    }
    std::cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    std::cout << chrono::duration_cast<chrono::microseconds>(t3 - t2).count() << "us" << std::endl;
}
void test_convolution()
{
    constexpr size_t len = 1 << 7;
    using ModInt = hint::modint::MontInt32Lazy<998244353>;
    using NTT = hint::transform::ntt::radix2_avx::NTT<998244353, 3>;
    alignas(32) static ModInt p1[len];
    alignas(32) static ModInt p2[len];
    for (uint32_t i = 0; i < len / 2; ++i)
    {
        p1[i].data = i;
    }
    for (uint32_t i = 0; i < len / 2; ++i)
    {
        p2[i].data = len - i;
    }
    ModInt inv_r = len;
    inv_r = inv_r.inv().mulR();
    auto t1 = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < 128; i++)
    {
        // for (size_t i = 0; i < 1000000; i++)
        {
            NTT::template convolutionRecursion(p1, p2, len, inv_r);
            // NTT::NTTTemplate::dit(p1, len, 128);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < len; ++i)
    {
        std::cout << p1[i].norm().data << " ";
        if (i % 8 == 7)
        {
            std::cout << "\n";
        }
    }
    std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
}

void test_load_mask()
{
    uint32_t arr[8]{};
    using namespace hint;
    using namespace simd;
    using namespace modint;
    using ModInt = MontInt32Lazy<998244353>;
    using ModIntX8 = MontInt32X8<ModInt>;
    ModIntX8 t0 = ModInt(1);
    for (int i = 0; i <= 8; i++)
    {
        t0.storeN(arr, i);
        for (int j = 0; j < 8; j++)
        {
            std::cout << arr[j] << ' ';
        }
        std::cout << "\n";
    }
}

int main1()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    check_ntt();
    // ntt_check(4);
    // test_mul(); // 测试乘法
    // test_mul64();
    // test_ntt(); // 测试卷积
    // test_poly();
    // test_DITX8();
    // test_convolution();
    // test_load_mask();
    // test_solve();
    // solve();
    // std::cout << hint::transform::ntt::add_count << "\t" << hint::transform::ntt::mul_count << "\n";
    // ntt_perf_loop();
}
#include <random>
void test(int n)
{
    alignas(64) static uint32_t a[1 << 24], b[1 << 24], c[1 << 24];
    alignas(64) static uint32_t at[1 << 24], bt[1 << 24];
    std::mt19937 rnd(10);
    for (int i = 0; i < n; i++)
    {
        a[i] = rnd() % 10;
        b[i] = rnd() % 10;
        a[i + n] = 0;
        b[i + n] = 0;
        at[i] = a[i];
        bt[i] = b[i];
        at[i + n] = a[i + n];
        bt[i + n] = b[i + n];
    }

    int k = (1 << 27) / n;

    int st = clock();

    for (int i = 0; i < k; i++)
    {
        memcpy(a, at, n * 2 * 4);
        memcpy(b, bt, n * 2 * 4);
        poly_multiply(a, n - 1, b, n - 1, c);
    }

    int ed = clock();
    uint64_t fa = 0;
    uint64_t fb = 0;
    uint64_t fc = 0;

    uint64_t xp = 1;
    uint32_t x = 17;

    constexpr uint32_t mod = 469762049;

    // print(at,n);
    // print(bt,n);
    // print(c,2*n);

    for (int i = 0; i < n; i++)
    {
        fa = (fa + (uint64_t)at[i] * xp) % mod;
        fb = (fb + (uint64_t)bt[i] * xp) % mod;
        xp = xp * x % mod;
    }
    xp = 1;
    for (int i = 0; i < 2 * n; i++)
    {
        fc = (fc + (uint64_t)c[i] * xp) % mod;
        xp = xp * x % mod;
    }

    printf("%dx%d %d\n", k, n, ed - st);
    if (fa * fb % mod != fc)
        puts("err");
}

int main()
{
    for (int i = 1 << 14; i <= (1 << 20); i <<= 1)
    {
        test(i);
    }
}
