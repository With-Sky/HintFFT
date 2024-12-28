#include <vector>
#include <complex>
#include <iostream>
#include <cstring>
#include <ctime>
#include <climits>
#include <string>
#include <array>
#include <type_traits>
#include <immintrin.h>
#include <emmintrin.h>

#pragma GCC target("avx2")

namespace hint
{
    using Float32 = float;
    using Float64 = double;
    using Complex32 = std::complex<Float32>;
    using Complex64 = std::complex<Float64>;

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

    namespace transform
    {
        template <typename T>
        inline void transform2(T &sum, T &diff)
        {
            T temp0 = sum, temp1 = diff;
            sum = temp0 + temp1;
            diff = temp0 - temp1;
        }

        // 二进制逆序
        template <typename It>
        void binary_reverse_swap(It begin, It end)
        {
            const size_t len = end - begin;
            // 左下标小于右下标时交换,防止重复交换
            auto smaller_swap = [=](It it_left, It it_right)
            {
                if (it_left < it_right)
                {
                    std::swap(it_left[0], it_right[0]);
                }
            };
            // 若i的逆序数的迭代器为last,则返回i+1的逆序数的迭代器
            auto get_next_bitrev = [=](It last)
            {
                size_t k = len / 2, indx = last - begin;
                indx ^= k;
                while (k > indx)
                {
                    k >>= 1;
                    indx ^= k;
                };
                return begin + indx;
            };
            // 长度较短的普通逆序
            if (len <= 16)
            {
                for (auto i = begin + 1, j = begin + len / 2; i < end - 1; i++)
                {
                    smaller_swap(i, j);
                    j = get_next_bitrev(j);
                }
                return;
            }
            const size_t len_8 = len / 8;
            const auto last = begin + len_8;
            auto i0 = begin + 1, i1 = i0 + len / 2, i2 = i0 + len / 4, i3 = i1 + len / 4;
            for (auto j = begin + len / 2; i0 < last; i0++, i1++, i2++, i3++)
            {
                smaller_swap(i0, j);
                smaller_swap(i1, j + 1);
                smaller_swap(i2, j + 2);
                smaller_swap(i3, j + 3);
                smaller_swap(i0 + len_8, j + 4);
                smaller_swap(i1 + len_8, j + 5);
                smaller_swap(i2 + len_8, j + 6);
                smaller_swap(i3 + len_8, j + 7);
                j = get_next_bitrev(j);
            }
        }

        // 二进制逆序
        template <typename T>
        void binary_reverse_swap(T ary, const size_t len)
        {
            binary_reverse_swap(ary, ary + len);
        }

        // 多模式，自动类型，自检查快速数论变换
        namespace ntt
        {
            constexpr uint64_t MOD3 = 754974721, ROOT3 = 11;
            constexpr uint64_t MOD4 = 469762049, ROOT4 = 3;

            //  Montgomery for mod < 2^30
            //  default R = 2^32
            template <uint32_t MOD>
            class MontInt32Lazy
            {
            private:
                static_assert(hint_log2(MOD) < 30, "MOD can't be larger than 30 bits");
                uint32_t data;

            public:
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
                constexpr uint32_t rawData() const
                {
                    return data;
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
            };

            template <typename MontInt32Type>
            struct MontInt32X8
            {
                using MontInt = MontInt32Type;
                using Int32X8 = __m256i;
                __m256i data;

                MontInt32X8() : data(_mm256_setzero_si256()) {}
                MontInt32X8(MontInt x) : data(_mm256_set1_epi32(x.rawData())) {}
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
                    data = _mm256_loadu_si256((const __m256i *)p);
                }
                template <typename T>
                void load(const T *p)
                {
                    data = _mm256_load_si256((const __m256i *)p);
                    // data = *reinterpret_cast<const __m256i *>(p);
                }
                template <typename T>
                void storeu(T *p) const
                {
                    // _mm256_stream_si256((__m256i *)p, data);
                    _mm256_storeu_si256((__m256i *)p, data);
                }
                template <typename T>
                void store(T *p) const
                {
                    _mm256_store_si256((__m256i *)p, data);
                    // *reinterpret_cast<__m256i *>(p) = data;
                }
                uint32_t nthU32(size_t i) const
                {
                    return _mm256_extract_epi32(data, i);
                }
                uint64_t nthU64(size_t i) const
                {
                    return _mm256_extract_epi64(data, i);
                }
                void printU32() const
                {
                    std::cout << "[" << nthU32(0) << "," << nthU32(1)
                              << "," << nthU32(2) << "," << nthU32(3)
                              << "," << nthU32(4) << "," << nthU32(5)
                              << "," << nthU32(6) << "," << nthU32(7) << "]" << std::endl;
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

            // 快速计算两模数的中国剩余定理
            template <uint32_t MOD1, uint32_t MOD2>
            inline uint64_t crt2(uint32_t num1, uint32_t num2)
            {
                constexpr uint64_t inv1 = mod_inv<int64_t>(MOD1, MOD2);
                constexpr uint64_t inv2 = mod_inv<int64_t>(MOD2, MOD1);
                static_assert(check_inv<uint64_t>(inv1, MOD1, MOD2), "Inv1 error");
                static_assert(check_inv<uint64_t>(inv2, MOD2, MOD1), "Inv2 error");
                if (num1 > num2)
                {
                    return (uint64_t(num1 - num2) * uint64_t(inv2) % MOD1) * MOD2 + num2;
                }
                else
                {
                    return (uint64_t(num2 - num1) * uint64_t(inv1) % MOD2) * MOD1 + num1;
                }
            }

            namespace split_radix_avx
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
                // in: in_out0<4p, in_ou1<4p; in_out2<2p, in_ou3<2p
                // out: in_out0<4p, in_ou1<4p; in_out2<4p, in_ou3<4p
                template <uint32_t ROOT, typename ModIntType, typename T>
                inline void dit_butterfly244(T &in_out0, T &in_out1, T &in_out2, T &in_out3)
                {
                    T temp0, temp1, temp2, temp3;
                    temp0 = in_out0.largeNorm();
                    temp1 = in_out1.largeNorm();
                    temp2 = in_out2 + in_out3;
                    temp3 = in_out2.sub(in_out3);
                    temp3 = mul_w41<ROOT, ModIntType>(temp3);
                    in_out0 = temp0.add(temp2);
                    in_out2 = temp0.sub(temp2);
                    in_out1 = temp1.add(temp3);
                    in_out3 = temp1.sub(temp3);
                }

                // in: in_out0<2p, in_ou1<2p; in_out2<2p, in_ou3<2p
                // out: in_out0<2p, in_ou1<2p; in_out2<4p, in_ou3<4p
                template <uint32_t ROOT, typename ModIntType, typename T>
                inline void dif_butterfly244(T &in_out0, T &in_out1, T &in_out2, T &in_out3)
                {
                    T temp0, temp1, temp2, temp3;
                    temp0 = in_out0.add(in_out2);
                    temp2 = in_out0 - in_out2;
                    temp1 = in_out1.add(in_out3);
                    temp3 = in_out1.sub(in_out3);
                    temp3 = mul_w41<ROOT, ModIntType>(temp3);
                    in_out0 = temp0.largeNorm();
                    in_out1 = temp1.largeNorm();
                    in_out2 = temp2.add(temp3);
                    in_out3 = temp2.sub(temp3);
                }

                // in: in_out0<4p, in_ou1<4p
                // out: in_out0<4p, in_ou1<4p
                template <typename ModIntType>
                inline void dit_butterfly2(ModIntType &in_out0, ModIntType &in_out1, const ModIntType &omega)
                {
                    auto x = in_out0.largeNorm();
                    auto y = in_out1 * omega;
                    in_out0 = x.add(y);
                    in_out1 = x.sub(y);
                }

                // in: in_out0<2p, in_ou1<2p
                // out: in_out0<2p, in_ou1<2p
                template <typename ModIntType>
                inline void dif_butterfly2(ModIntType &in_out0, ModIntType &in_out1, const ModIntType &omega)
                {
                    auto x = in_out0 + in_out1;
                    auto y = in_out0.sub(in_out1);
                    in_out0 = x;
                    in_out1 = y * omega;
                }

                template <size_t MAX_LEN, uint32_t ROOT, typename ModInt>
                struct NTTShort
                {
                    static constexpr size_t NTT_LEN = MAX_LEN;
                    static constexpr int LOG_LEN = hint_log2(NTT_LEN);

                    using ModIntX8 = MontInt32X8<ModInt>;
                    using ModIntType = ModInt;

                    struct TableType
                    {
                        alignas(64) std::array<ModIntType, NTT_LEN> omega_table;
                        // Compute in compile time if need.
                        /*constexpr*/ TableType()
                        {
                            for (int omega_log_len = 0; omega_log_len <= LOG_LEN; omega_log_len++)
                            {
                                size_t omega_len = size_t(1) << omega_log_len, omega_count = omega_len / 2;
                                auto it = &omega_table[omega_len / 2];
                                ModIntType root = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / omega_len);
                                ModIntType omega(1);
                                for (size_t i = 0; i < omega_count; i++)
                                {
                                    it[i] = omega;
                                    omega *= root;
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

                    static TableType table;

                    static ModIntX8 omegax8(size_t ntt_len, int factor)
                    {
                        alignas(32) ModIntType w_arr[8]{};
                        ModIntType w(1), unit(qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / ntt_len * factor));
                        for (auto &&i : w_arr)
                        {
                            i = w;
                            w = w * unit;
                        }
                        return ModIntX8(w_arr);
                    }

                    static void dit(ModIntType in_out[], size_t len)
                    {
                        len = std::min(NTT_LEN, len);
                        if (len <= 16)
                        {
                            NTTShort<16, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        size_t rank = len;
                        if (hint_log2(len) % 2 == 0)
                        {
                            for (size_t i = 0; i < len; i += 16)
                            {
                                NTTShort<16, ROOT, ModIntType>::dit(in_out + i);
                            }
                            rank = 64;
                        }
                        else
                        {
                            for (size_t i = 0; i < len; i += 16)
                            {
                                NTTShort<8, ROOT, ModIntType>::dit8X2(in_out + i);
                            }
                            rank = 32;
                        }
                        for (; rank <= len; rank *= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i += 8)
                                {
                                    ModIntX8 temp0, temp1, temp2, temp3, omega;
                                    temp0.load(&it0[j + i]), temp1.load(&it1[j + i]), temp2.load(&it2[j + i]), temp3.load(&it3[j + i]);
                                    omega.load(&last_omega_it[i]);

                                    dit_butterfly2(temp0, temp1, omega);
                                    dit_butterfly2(temp2, temp3, omega);
                                    omega.load(&omega_it[i]);
                                    dit_butterfly2(temp0, temp2, omega);
                                    omega.load(&omega_it[gap + i]);
                                    dit_butterfly2(temp1, temp3, omega);

                                    temp0.store(&it0[j + i]), temp1.store(&it1[j + i]), temp2.store(&it2[j + i]), temp3.store(&it3[j + i]);
                                }
                            }
                        }
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        len = std::min(NTT_LEN, len);
                        if (len <= 16)
                        {
                            NTTShort<16, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        size_t rank = len;
                        for (; rank >= 32; rank /= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = table.getOmegaIt(rank), last_omega_it = table.getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i += 8)
                                {
                                    ModIntX8 temp0, temp1, temp2, temp3, omega;
                                    temp0.load(&it0[j + i]), temp1.load(&it1[j + i]), temp2.load(&it2[j + i]), temp3.load(&it3[j + i]);

                                    omega.load(&omega_it[i]);
                                    dif_butterfly2(temp0, temp2, omega);
                                    omega.load(&omega_it[gap + i]);
                                    dif_butterfly2(temp1, temp3, omega);
                                    omega.load(&last_omega_it[i]);
                                    dif_butterfly2(temp0, temp1, omega);
                                    dif_butterfly2(temp2, temp3, omega);

                                    temp0.store(&it0[j + i]), temp1.store(&it1[j + i]), temp2.store(&it2[j + i]), temp3.store(&it3[j + i]);
                                }
                            }
                        }
                        if (hint_log2(len) % 2 == 0)
                        {
                            for (size_t i = 0; i < len; i += 16)
                            {
                                NTTShort<16, ROOT, ModIntType>::dif(in_out + i);
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < len; i += 16)
                            {
                                NTTShort<8, ROOT, ModIntType>::dif8X2(in_out + i);
                            }
                        }
                    }
                };
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                typename NTTShort<LEN, ROOT, ModIntType>::TableType NTTShort<LEN, ROOT, ModIntType>::table;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr size_t NTTShort<LEN, ROOT, ModIntType>::NTT_LEN;
                template <size_t LEN, uint32_t ROOT, typename ModIntType>
                constexpr int NTTShort<LEN, ROOT, ModIntType>::LOG_LEN;

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<0, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<1, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<2, ROOT, ModIntType>
                {
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    static void dit(ModIntType in_out[])
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    static void dif(ModIntType in_out[])
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 2)
                        {
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 2)
                        {
                            return;
                        }
                        dif(in_out);
                    }
                };

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<4, ROOT, ModIntType>
                {
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    static void dit(ModIntType in_out[])
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
                    static void dif(ModIntType in_out[])
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
                    static ModIntX8 transform2X4(ModIntX8 in)
                    {
                        ModIntX8 lo = in.lshift32In64(); // 0, a
                        ModIntX8 hi = in.rshift32In64(); // b, 0
                        lo = lo.sub(in);                 // X, a - b + mod2
                        hi = hi.add(in);                 // a + b ,X
                        return ModIntX8::cross32(hi, lo).largeNorm();
                    }
                    static void dit4X4(ModIntX8 &A, ModIntX8 &B)
                    {
                        constexpr ModIntType W_4_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 4);
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_4_1, ModIntType(1), W_4_1, ModIntType(1), W_4_1, ModIntType(1), W_4_1};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp0 = transform2X4(A); // A0,A1,A2,A3,A4,A5,A6,A7
                        temp1 = transform2X4(B); // B0,B1,B2,B3,B4,B5,B6,B7

                        omega.load(w_arr);
                        temp2 = temp0.rshift64In128(); // A2,A3,X,X,A6,A7,X,X
                        temp3 = temp1.lshift64In128(); // X,X,B0,B1,X,X,B4,B5

                        temp0 = ModIntX8::cross64(temp0, temp3); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp1 = ModIntX8::cross64(temp2, temp1); // A2,A3,B2,B3,A6,A7,B6,B7

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
                        constexpr ModIntType W_4_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 4);
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_4_1, ModIntType(1), W_4_1, ModIntType(1), W_4_1, ModIntType(1), W_4_1};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp2 = A.rshift64In128(); // A2,A3,X,X,A6,A7,X,X
                        temp3 = B.lshift64In128(); // X,X,B0,B1,X,X,B4,B5

                        omega.load(w_arr);
                        temp0 = ModIntX8::cross64(A, temp3); // A0,A1,B0,B1,A4,A5,B4,B5
                        temp1 = ModIntX8::cross64(temp2, B); // A2,A3,B2,B3,A6,A7,B6,B7

                        temp2 = temp0 + temp1; // A0,A1,B0,B1,A4,A5,B4,B5
                        temp3 = temp0.sub(temp1);
                        temp3 = temp3 * omega; // (A2,A3,B2,B3,A6,A7,B6,B7)*w

                        temp0 = temp2.rshift64In128(); // B0,B1,X,X,B4,B5,X,X
                        temp1 = temp3.lshift64In128(); // X,X,A2,A3,X,X,A6,A7

                        temp2 = ModIntX8::cross64(temp2, temp1); // A0,A1,A2,A3,A4,A5,A6,A7
                        temp3 = ModIntX8::cross64(temp0, temp3); // B0,B1,B2,B3,B4,B5,B6,B7

                        A = transform2X4(temp2); // A
                        B = transform2X4(temp3); // B
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 4)
                        {
                            NTTShort<2, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 4)
                        {
                            NTTShort<2, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        dif(in_out);
                    }
                };

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<8, ROOT, ModIntType>
                {
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    static void dit(ModIntType in_out[])
                    {
                        static constexpr ModIntType w1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        static constexpr ModIntType w2 = qpow(w1, 2);
                        static constexpr ModIntType w3 = qpow(w1, 3);
                        auto temp0 = in_out[0].largeNorm();
                        auto temp1 = in_out[1].largeNorm();
                        auto temp2 = in_out[2].largeNorm();
                        auto temp3 = in_out[3].largeNorm();
                        auto temp4 = in_out[4].largeNorm();
                        auto temp5 = in_out[5].largeNorm();
                        auto temp6 = in_out[6].largeNorm();
                        auto temp7 = in_out[7].largeNorm();

                        transform2(temp0, temp1);
                        transform2(temp4, temp5);
                        auto sum = temp2.add(temp3);
                        auto dif = temp2.sub(temp3);
                        temp2 = sum.largeNorm();
                        temp3 = mul_w41<ROOT, ModIntType>(dif);
                        sum = temp6.add(temp7);
                        dif = temp6.sub(temp7);
                        temp6 = sum.largeNorm();
                        temp7 = mul_w41<ROOT, ModIntType>(dif);

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        sum = temp4.add(temp6);
                        dif = temp4.sub(temp6);
                        temp4 = sum.largeNorm();
                        temp6 = dif * w2;
                        sum = temp5.add(temp7);
                        dif = temp5.sub(temp7);
                        temp5 = sum * w1;
                        temp7 = dif * w3;

                        in_out[0] = temp0.add(temp4);
                        in_out[1] = temp1.add(temp5);
                        in_out[2] = temp2.add(temp6);
                        in_out[3] = temp3.add(temp7);
                        in_out[4] = temp0.sub(temp4);
                        in_out[5] = temp1.sub(temp5);
                        in_out[6] = temp2.sub(temp6);
                        in_out[7] = temp3.sub(temp7);
                    }
                    static void dif(ModIntType in_out[])
                    {
                        static constexpr ModIntType w1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        static constexpr ModIntType w2 = qpow(w1, 2);
                        static constexpr ModIntType w3 = qpow(w1, 3);
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];
                        auto temp4 = in_out[4];
                        auto temp5 = in_out[5];
                        auto temp6 = in_out[6];
                        auto temp7 = in_out[7];

                        transform2(temp0, temp4);
                        auto sum = temp1.add(temp5);
                        auto dif = temp1.sub(temp5);
                        temp1 = sum.largeNorm();
                        temp5 = dif * w1;
                        sum = temp2.add(temp6);
                        dif = temp2.sub(temp6);
                        temp2 = sum.largeNorm();
                        temp6 = dif * w2;
                        sum = temp3.add(temp7);
                        dif = temp3.sub(temp7);
                        temp3 = sum.largeNorm();
                        temp7 = dif * w3;

                        transform2(temp0, temp2);
                        transform2(temp4, temp6);
                        sum = temp1.add(temp3);
                        dif = temp1.sub(temp3);
                        temp1 = sum.largeNorm();
                        temp3 = mul_w41<ROOT, ModIntType>(dif);
                        sum = temp5.add(temp7);
                        dif = temp5.sub(temp7);
                        temp5 = sum.largeNorm();
                        temp7 = mul_w41<ROOT, ModIntType>(dif);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                        in_out[4] = temp4 + temp5;
                        in_out[5] = temp4 - temp5;
                        in_out[6] = temp6 + temp7;
                        in_out[7] = temp6 - temp7;
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 8)
                        {
                            NTTShort<4, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 8)
                        {
                            NTTShort<4, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        dif(in_out);
                    }
                    static void dit8X2(ModIntX8 &A, ModIntX8 &B)
                    {
                        constexpr ModIntType W_8_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        constexpr ModIntType W_8_2 = qpow(W_8_1, 2);
                        constexpr ModIntType W_8_3 = qpow(W_8_1, 3);
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_1, W_8_2, W_8_3, ModIntType(1), W_8_1, W_8_2, W_8_3};

                        NTTShort<4, ROOT, ModIntType>::dit4X4(A, B); // A0,A1,A2,A3,A4,A5,A6,A7; B0,B1,B2,B3,B4,B5,B6,B7
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
                        constexpr ModIntType W_8_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                        constexpr ModIntType W_8_2 = qpow(W_8_1, 2);
                        constexpr ModIntType W_8_3 = qpow(W_8_1, 3);
                        alignas(32) constexpr ModIntType w_arr[8]{ModIntType(1), W_8_1, W_8_2, W_8_3, ModIntType(1), W_8_1, W_8_2, W_8_3};

                        ModIntX8 temp0, temp1, temp2, temp3, omega;
                        temp0 = ModIntX8::packLo128(A, B); // A0,A1,A2,A3,B0,B1,B2,B3
                        temp1 = ModIntX8::packHi128(A, B); // A4,A5,A6,A7,B4,B5,B6,B7
                        omega.load(w_arr);
                        temp2 = temp0 + temp1;                 // A0,A1,A2,A3,B0,B1,B2,B3
                        temp3 = temp0.sub(temp1);              // A4,A5,A6,A7,B4,B5,B6,B7
                        temp3 = temp3 * omega;                 //(A4,A5,A6,A7,B4,B5,B6,B7)*w
                        A = ModIntX8::packLo128(temp2, temp3); // A0,A1,A2,A3,A4,A5,A6,A7
                        B = ModIntX8::packHi128(temp2, temp3); // B0,B1,B2,B3,B4,B5,B6,B7
                        NTTShort<4, ROOT, ModIntType>::dif4X4(A, B);
                    }
                    static void dit8X2(ModIntType in_out[])
                    {
                        ModIntX8 A, B;
                        A.load(&in_out[0]);
                        B.load(&in_out[8]);
                        dit8X2(A, B);
                        A.store(&in_out[0]);
                        B.store(&in_out[8]);
                    }
                    static void dif8X2(ModIntType in_out[])
                    {
                        ModIntX8 A, B;
                        A.load(&in_out[0]);
                        B.load(&in_out[8]);
                        dif8X2(A, B);
                        A.store(&in_out[0]);
                        B.store(&in_out[8]);
                    }
                };

                template <uint32_t ROOT, typename ModIntType>
                struct NTTShort<16, ROOT, ModIntType>
                {
                    using ModIntX8 = MontInt32X8<ModIntType>;
                    static void dit(ModIntType in_out[])
                    {
                        static const ModIntX8 omega = NTTShort<32, ROOT, ModIntType>::omegax8(16, 1);
                        ModIntX8 temp0, temp1;
                        temp0.load(&in_out[0]);
                        temp1.load(&in_out[8]);

                        NTTShort<8, ROOT, ModIntType>::dit8X2(temp0, temp1);
                        temp0 = temp0.largeNorm();
                        temp1 = temp1 * omega;

                        temp0.add(temp1).store(&in_out[0]);
                        temp0.sub(temp1).store(&in_out[8]);
                    }
                    static void dif(ModIntType in_out[])
                    {
                        static const ModIntX8 omega = NTTShort<32, ROOT, ModIntType>::omegax8(16, 1);
                        ModIntX8 temp0, temp1, sum, dif;
                        temp0.load(&in_out[0]);
                        temp1.load(&in_out[8]);

                        sum = temp0.add(temp1);
                        dif = temp0.sub(temp1);
                        temp0 = sum.largeNorm();
                        temp1 = dif * omega;
                        NTTShort<8, ROOT, ModIntType>::dif8X2(temp0, temp1);

                        temp0.store(&in_out[0]);
                        temp1.store(&in_out[8]);
                    }
                    static void dit(ModIntType in_out[], size_t len)
                    {
                        if (len < 16)
                        {
                            NTTShort<8, ROOT, ModIntType>::dit(in_out, len);
                            return;
                        }
                        dit(in_out);
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        if (len < 16)
                        {
                            NTTShort<8, ROOT, ModIntType>::dif(in_out, len);
                            return;
                        }
                        dif(in_out);
                    }
                };

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

                    static constexpr size_t L1_BYTE = size_t(1) << 18; // 32KB L1 cache size, change this if you know your cache size.
                    static constexpr size_t LONG_THRESHOLD = std::min(L1_BYTE / sizeof(ModIntType), NTT_MAX_LEN);
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
                    static void dit244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), NTT_MAX_LEN);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dit(in_out, ntt_len);
                            return;
                        }
                        const size_t quarter_len = ntt_len / 4;
                        dit244(in_out + quarter_len * 3, ntt_len / 4);
                        dit244(in_out + quarter_len * 2, ntt_len / 4);
                        dit244(in_out, ntt_len / 2);
                        const ModIntX8 unit1_x8 = unitx8(ntt_len, 1), unit3_x8 = unitx8(ntt_len, 3);
                        ModIntX8 omega1 = omegax8(ntt_len, 1), omega3 = omegax8(ntt_len, 3);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i += 8)
                        {
                            ModIntX8 temp0, temp1, temp2, temp3;
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);
                            temp2 = temp2 * omega1, temp3 = temp3 * omega3;

                            dit_butterfly244<ROOT, ModIntType>(temp0, temp1, temp2, temp3);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), (temp2).store(&it2[i]), (temp3).store(&it3[i]);

                            omega1 = omega1 * unit1_x8;
                            omega3 = omega3 * unit3_x8;
                        }
                    }
                    static void dif244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), NTT_MAX_LEN);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dif(in_out, ntt_len);
                            return;
                        }
                        const size_t quarter_len = ntt_len / 4;
                        const ModIntX8 unit1_x8 = unitx8(ntt_len, 1), unit3_x8 = unitx8(ntt_len, 3);
                        ModIntX8 omega1 = omegax8(ntt_len, 1), omega3 = omegax8(ntt_len, 3);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i += 8)
                        {
                            ModIntX8 temp0, temp1, temp2, temp3;
                            temp0.load(&it0[i]), temp1.load(&it1[i]), temp2.load(&it2[i]), temp3.load(&it3[i]);

                            dif_butterfly244<ROOT, ModIntType>(temp0, temp1, temp2, temp3);

                            temp0.store(&it0[i]), temp1.store(&it1[i]), (temp2 * omega1).store(&it2[i]), (temp3 * omega3).store(&it3[i]);

                            omega1 = omega1 * unit1_x8;
                            omega3 = omega3 * unit3_x8;
                        }
                        dif244(in_out, ntt_len / 2);
                        dif244(in_out + quarter_len * 3, ntt_len / 4);
                        dif244(in_out + quarter_len * 2, ntt_len / 4);
                    }
                    static void convolution(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len)
                    {
                        const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                        dif244(in1, ntt_len);
                        dif244(in2, ntt_len);
                        if (ntt_len < 16)
                        {
                            for (size_t i = 0; i < ntt_len; i++)
                            {
                                out[i] = in1[i] * in2[i] * inv_len;
                            }
                        }
                        else
                        {
                            const ModIntX8 inv8(inv_len);
                            for (size_t i = 0; i < ntt_len; i += 16)
                            {
                                ModIntX8 temp0, temp1;
                                temp0.load(&in1[i]), temp1.load(&in2[i]);
                                (temp0 * temp1 * inv8).store(&out[i]);
                                temp0.load(&in1[i + 8]), temp1.load(&in2[i + 8]);
                                (temp0 * temp1 * inv8).store(&out[i + 8]);
                            }
                        }
                        INTT::dit244(out, ntt_len);
                    }

                    static void convolutionRecursion(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len, bool normlize = true)
                    {
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dif(in1, ntt_len);
                            if (in1 != in2)
                            {
                                NTTTemplate::dif(in2, ntt_len);
                            }
                            if (normlize)
                            {
                                const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                                if (ntt_len < 16)
                                {
                                    for (size_t i = 0; i < ntt_len; i++)
                                    {
                                        out[i] = in1[i] * in2[i] * inv_len;
                                    }
                                }
                                else
                                {
                                    const ModIntX8 inv8(inv_len);
                                    for (size_t i = 0; i < ntt_len; i += 16)
                                    {
                                        ModIntX8 temp0, temp1;
                                        temp0.load(&in1[i]), temp1.load(&in2[i]);
                                        (temp0 * temp1 * inv8).store(&out[i]);
                                        temp0.load(&in1[i + 8]), temp1.load(&in2[i + 8]);
                                        (temp0 * temp1 * inv8).store(&out[i + 8]);
                                    }
                                }
                            }
                            else
                            {
                                if (ntt_len < 16)
                                {
                                    for (size_t i = 0; i < ntt_len; i++)
                                    {
                                        out[i] = in1[i] * in2[i];
                                    }
                                }
                                else
                                {
                                    for (size_t i = 0; i < ntt_len; i += 16)
                                    {
                                        ModIntX8 temp0, temp1;
                                        temp0.load(&in1[i]), temp1.load(&in2[i]);
                                        (temp0 * temp1).store(&out[i]);
                                        temp0.load(&in1[i + 8]), temp1.load(&in2[i + 8]);
                                        (temp0 * temp1).store(&out[i + 8]);
                                    }
                                }
                            }
                            INTT::NTTTemplate::dit(out, ntt_len);
                            return;
                        }
                        const size_t quarter_len = ntt_len / 4;
                        ModIntX8 unit1_x8 = unitx8(ntt_len, 1), unit3_x8 = unitx8(ntt_len, 3);
                        ModIntX8 omega1 = omegax8(ntt_len, 1), omega3 = omegax8(ntt_len, 3);
                        if (in1 != in2)
                        {
                            for (size_t i = 0; i < quarter_len; i += 8)
                            {
                                ModIntX8 temp0, temp1, temp2, temp3;
                                temp0.load(&in1[i]), temp1.load(&in1[quarter_len + i]), temp2.load(&in1[quarter_len * 2 + i]), temp3.load(&in1[quarter_len * 3 + i]);
                                dif_butterfly244<ROOT, ModIntType>(temp0, temp1, temp2, temp3);
                                temp0.store(&in1[i]), temp1.store(&in1[quarter_len + i]), (temp2 * omega1).store(&in1[quarter_len * 2 + i]), (temp3 * omega3).store(&in1[quarter_len * 3 + i]);

                                temp0.load(&in2[i]), temp1.load(&in2[quarter_len + i]), temp2.load(&in2[quarter_len * 2 + i]), temp3.load(&in2[quarter_len * 3 + i]);
                                dif_butterfly244<ROOT, ModIntType>(temp0, temp1, temp2, temp3);
                                temp0.store(&in2[i]), temp1.store(&in2[quarter_len + i]), (temp2 * omega1).store(&in2[quarter_len * 2 + i]), (temp3 * omega3).store(&in2[quarter_len * 3 + i]);

                                omega1 = omega1 * unit1_x8;
                                omega3 = omega3 * unit3_x8;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < quarter_len; i += 8)
                            {
                                ModIntX8 temp0, temp1, temp2, temp3;
                                temp0.load(&in1[i]), temp1.load(&in1[quarter_len + i]), temp2.load(&in1[quarter_len * 2 + i]), temp3.load(&in1[quarter_len * 3 + i]);
                                dif_butterfly244<ROOT, ModIntType>(temp0, temp1, temp2, temp3);
                                temp0.store(&in1[i]), temp1.store(&in1[quarter_len + i]), (temp2 * omega1).store(&in1[quarter_len * 2 + i]), (temp3 * omega3).store(&in1[quarter_len * 3 + i]);

                                omega1 = omega1 * unit1_x8;
                                omega3 = omega3 * unit3_x8;
                            }
                        }

                        convolutionRecursion(in1, in2, out, ntt_len / 2, false);
                        convolutionRecursion(in1 + quarter_len * 2, in2 + quarter_len * 2, out + quarter_len * 2, ntt_len / 4, false);
                        convolutionRecursion(in1 + quarter_len * 3, in2 + quarter_len * 3, out + quarter_len * 3, ntt_len / 4, false);

                        unit1_x8 = unitx8(ntt_len, 1, rootInv()), unit3_x8 = unitx8(ntt_len, 3, rootInv());
                        omega1 = omegax8(ntt_len, 1, rootInv()), omega3 = omegax8(ntt_len, 3, rootInv());
                        if (normlize)
                        {
                            const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                            const ModIntX8 inv8(inv_len);

                            omega1 = omega1 * inv8, omega3 = omega3 * inv8;
                            for (size_t i = 0; i < quarter_len; i += 8)
                            {
                                ModIntX8 temp0, temp1, temp2, temp3;
                                temp0.load(&out[i]), temp1.load(&out[quarter_len + i]), temp2.load(&out[quarter_len * 2 + i]), temp3.load(&out[quarter_len * 3 + i]);
                                temp0 = temp0 * inv8, temp1 = temp1 * inv8;
                                temp2 = temp2 * omega1, temp3 = temp3 * omega3;
                                dit_butterfly244<rootInv(), ModIntType>(temp0, temp1, temp2, temp3);
                                temp0.store(&out[i]), temp1.store(&out[quarter_len + i]), temp2.store(&out[quarter_len * 2 + i]), temp3.store(&out[quarter_len * 3 + i]);

                                omega1 = omega1 * unit1_x8;
                                omega3 = omega3 * unit3_x8;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < quarter_len; i += 8)
                            {
                                ModIntX8 temp0, temp1, temp2, temp3;
                                temp0.load(&out[i]), temp1.load(&out[quarter_len + i]), temp2.load(&out[quarter_len * 2 + i]), temp3.load(&out[quarter_len * 3 + i]);
                                temp2 = temp2 * omega1, temp3 = temp3 * omega3;
                                dit_butterfly244<rootInv(), ModIntType>(temp0, temp1, temp2, temp3);
                                temp0.store(&out[i]), temp1.store(&out[quarter_len + i]), temp2.store(&out[quarter_len * 2 + i]), temp3.store(&out[quarter_len * 3 + i]);

                                omega1 = omega1 * unit1_x8;
                                omega3 = omega3 * unit3_x8;
                            }
                        }
                    }
                };
                template <uint32_t MOD, uint32_t ROOT>
                constexpr int NTT<MOD, ROOT>::MOD_BITS;
                template <uint32_t MOD, uint32_t ROOT>
                constexpr int NTT<MOD, ROOT>::MAX_LOG_LEN;
                template <uint32_t MOD, uint32_t ROOT>
                constexpr size_t NTT<MOD, ROOT>::NTT_MAX_LEN;
            }
        }
    }
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

void poly_multiply(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    using namespace std;
    using namespace hint;
    using namespace transform::ntt::split_radix_avx;
    size_t conv_len = m + n + 1, ntt_len = int_ceil2(conv_len);
    using NTT = transform::ntt::split_radix_avx::NTT<998244353, 3>;
    using ModInt = NTT::ModIntType;
    using ModIntX8 = NTT::ModIntX8;
    // alignas(32) static uint32_t arr_a[1 << 21];
    // alignas(32) static uint32_t arr_b[1 << 21];
    // auto a_ntt = reinterpret_cast<ModInt *>(arr_a), b_ntt = reinterpret_cast<ModInt *>(arr_b);
    auto a_ntt = (NTT::ModIntType *)_mm_malloc(ntt_len * sizeof(NTT::ModIntType), 32);
    auto b_ntt = (NTT::ModIntType *)_mm_malloc(ntt_len * sizeof(NTT::ModIntType), 32);
    memset(a_ntt + n + 1, 0, (ntt_len - n - 1) * sizeof(NTT::ModIntType));
    memset(b_ntt + m + 1, 0, (ntt_len - m - 1) * sizeof(NTT::ModIntType));
    arrToMont<ModInt, ModIntX8>(a, n + 1, a_ntt);
    arrToMont<ModInt, ModIntX8>(b, m + 1, b_ntt);
    NTT::convolution(a_ntt, b_ntt, a_ntt, ntt_len);
    arrToInt<ModInt, ModIntX8>(a_ntt, conv_len, c);
    _mm_free(a_ntt);
    _mm_free(b_ntt);
}

#include <chrono>
void test_poly()
{
    using namespace std;
    ios::sync_with_stdio(false);
    cout.tie(nullptr);
    cin.tie(nullptr);
    constexpr size_t len = 1 << 22;
    static unsigned a[len];
    static unsigned b[len];
    static unsigned c[len * 2 - 1];
    for (auto &&i : a)
    {
        i = 5;
    }
    for (auto &&i : b)
    {
        i = 5;
    }
    auto t1 = chrono::steady_clock::now();
    // for (size_t i = 0; i < 1000; i++)
    {
        poly_multiply(a, len - 1, b, len - 1, c);
    }
    auto t2 = chrono::steady_clock::now();
    for (auto i : c)
    {
        // std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}

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
                // std::cout << uint32_t(omega) << "\n";
                p[0] = temp0 + temp1;
                p[dis] = temp0 - temp1;
                omega = omega * unit_omega;
            }
        }
    }
}

template <uint32_t ROOT, typename ModInt>
void ntt_dif(ModInt in_out[], size_t ntt_len)
{
    for (size_t rank = ntt_len; rank >= 2; rank /= 2)
    {
        ModInt unit_omega = hint::qpow(ModInt(ROOT), (ModInt::mod() - 1) / rank);
        size_t dis = rank / 2;
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
    using ntt = split_radix_avx::NTT<mod, root>;

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
        ntt::dif244(b.data(), len);
        ntt::dif244(b.data(), len);
        ntt::dit244(b.data(), len);
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
std::vector<T> poly_multiply(const std::vector<T> &in1, const std::vector<T> &in2)
{
    using NTT = hint::transform::ntt::split_radix_avx::NTT<469762049, 3>;
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    std::vector<T> result(out_len);
    size_t ntt_len = hint::int_floor2(out_len);
    auto buffer1 = (NTT::ModIntType *)_mm_malloc(ntt_len * sizeof(NTT::ModIntType), 32);
    auto buffer2 = (NTT::ModIntType *)_mm_malloc(ntt_len * sizeof(NTT::ModIntType), 32);
    std::fill(buffer1 + len1, buffer1 + ntt_len, NTT::ModIntType{});
    std::fill(buffer2 + len2, buffer2 + ntt_len, NTT::ModIntType{});
    std::copy(in1.begin(), in1.end(), buffer1);
    std::copy(in2.begin(), in2.end(), buffer2);
    auto t1 = std::chrono::high_resolution_clock::now();
    NTT::convolution(buffer1, buffer2, buffer1, ntt_len);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "ntt_convolution time: " << t << "us" << std::endl;
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(buffer1[i]);
    }
    _mm_free(buffer1);
    _mm_free(buffer2);
    return result;
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
    std::vector<uint32_t> res = poly_multiply(in1, in2);
    auto t2 = std::chrono::steady_clock::now();
    result_test<uint32_t>(res, ele); // 结果校验
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}
int main()
{
    // ntt_check(4);
    // test_mul(); // 测试乘法
    // test_mul64();
    // test_ntt(); // 测试卷积
    test_poly();
    // test_solve();
    // solve();
    // std::cout << hint::transform::ntt::add_count << "\t" << hint::transform::ntt::mul_count << "\n";
    // ntt_perf_loop();
}