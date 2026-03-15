// TSKY 2025/10/31
#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <climits>
#include <string>
#include <cstring>
#include <array>
#include <cassert>
#include <type_traits>

#if defined(_WIN64) // Windows MSVC X64
#include <intrin.h>
#define HINT_WIN64
#endif

#if defined(__x86_64__) || defined(__amd64__) // None MSVC X64
#include <x86gprintrin.h>
#define HINT_X86_64
#endif

#if defined(__SIZEOF_INT128__) // GCC int128
#define HINT_INT128
#endif

namespace hint
{
    using HintULL = unsigned long long;

    template <typename IntTy>
    constexpr bool is_2pow(IntTy n)
    {
        return (n > 0) && (n & (n - 1)) == 0;
    }

    /// @brief Floor round to the nearest power of 2
    /// @tparam T
    /// @param n Integer that not negative
    /// @return Power of 2 that is not larger than n
    template <typename T>
    constexpr T int_floor2(T n)
    {
        constexpr int bits = sizeof(n) * 8;
        for (int i = 1; i < bits; i *= 2)
        {
            n |= (n >> i);
        }
        return (n >> 1) + 1;
    }

    /// @brief Ceiling round to the nearest power of 2
    /// @tparam T
    /// @param n Integer that not negative
    /// @return Power of 2 that is not smaller than n
    template <typename T>
    constexpr T int_ceil2(T n)
    {
        constexpr int bits = sizeof(n) * 8;
        if (n > 0)
        {
            n--;
        }
        for (int i = 1; i < bits; i *= 2)
        {
            n |= (n >> i);
        }
        return n + 1;
    }

    template <typename T>
    constexpr T all_one(int bits)
    {
        T temp = T(1) << (bits - 1);
        return temp - 1 + temp;
    }
    constexpr int hint_ctz(uint32_t x)
    {
        int r0 = 16, r1 = 8, r2 = 4, r3 = 2, r4 = 1;
        x &= (-x);
        if (x & 0x0000FFFF)
        {
            r0 = 0;
        }
        if (x & 0x00FF00FF)
        {
            r1 = 0;
        }
        if (x & 0x0F0F0F0F)
        {
            r2 = 0;
        }
        if (x & 0x33333333)
        {
            r3 = 0;
        }
        if (x & 0x55555555)
        {
            r4 = 0;
        }
        return r0 + r1 + r2 + r3 + r4 + (x == 0);
    }

    constexpr int hint_ctz(uint64_t x)
    {
        if (uint32_t(x))
        {
            return hint_ctz(uint32_t(x));
        }
        return hint_ctz(uint32_t(x >> 32)) + 32;
    }

    // Leading zeros
    constexpr int hint_clz(uint32_t x)
    {
        constexpr uint32_t MASK32 = uint32_t(0xFFFF) << 16;
        int res = sizeof(uint32_t) * CHAR_BIT;
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
        if (x >> 32)
        {
            return hint_clz(uint32_t(x >> 32));
        }
        return hint_clz(uint32_t(x)) + 32;
    }
    // Fast power
    template <typename T, typename T1>
    constexpr T qpow(T m, T1 n)
    {
        T result = 1;
        while (true)
        {
            if ((n & 1) != 0)
            {
                result *= m;
            }
            if (n == 0)
            {
                break;
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
        while (true)
        {
            if ((n & 1) != 0)
            {
                result *= m;
                result %= mod;
            }
            if (n == 0)
            {
                break;
            }
            m *= m;
            m %= mod;
            n >>= 1;
        }
        return result;
    }
    template <typename T>
    constexpr T hint_gcd(T a, T b)
    {
        if (a < b)
        {
            std::swap(a, b);
        }
        if (0 == b)
        {
            return a;
        }
        const int i = hint::hint_ctz(a);
        a >>= i;
        const int j = hint::hint_ctz(b);
        b >>= j;
        const int k = std::min(i, j);
        while (true)
        {
            if (a < b)
            {
                std::swap(a, b);
            }
            if (b == 0)
            {
                break;
            }
            a -= b;
            a >>= hint::hint_ctz(a);
        }
        return a << k;
    }
    template <typename T>
    constexpr T hint_lcm(T a, T b)
    {
        return a / hint::hint_gcd(a, b) * b;
    }
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

    template <typename IntTy>
    inline IntTy exgcd_iter(IntTy a, IntTy b, IntTy &x, IntTy &y)
    {
        auto exec = [](IntTy &m, IntTy &n, IntTy q)
        {
            IntTy temp = m - n * q;
            m = n, n = temp;
        };
        x = 1, y = 0;
        IntTy x1 = 0, y1 = 1;
        while (b > 0)
        {
            IntTy q = a / b;
            exec(x, x1, q);
            exec(y, y1, q);
            exec(a, b, q);
        }
        return a;
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

    namespace utility
    {

        template <typename T1, typename T2>
        bool mem_overlap(T1 begin1, T1 end1, T2 begin2, T2 end2)
        {
            return begin1 <= end2 && begin2 <= end1;
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

        template <typename Ui64>
        constexpr void mul64x64to128_base(uint64_t a, uint64_t b, Ui64 &low, Ui64 &high)
        {
            static_assert(sizeof(Ui64) == sizeof(uint64_t), "mul64x64to128_base: low and high must be 64bit");
            uint64_t ah = a >> 32, bh = b >> 32;
            a = uint32_t(a), b = uint32_t(b);
            uint64_t r0 = a * b, r1 = a * bh, r2 = ah * b, r3 = ah * bh;
            r3 += (r1 >> 32) + (r2 >> 32);
            r1 = uint32_t(r1), r2 = uint32_t(r2);
            r1 += r2;
            r1 += (r0 >> 32);
            high = r3 + (r1 >> 32);
            low = (r1 << 32) | uint32_t(r0);
        }

        template <typename Ui64>
        inline void mul64x64to128(uint64_t a, uint64_t b, Ui64 &low, Ui64 &high)
        {
            static_assert(sizeof(Ui64) == sizeof(uint64_t), "mul64x64to128: low and high must be 64bit");
#if defined(HINT_INT128) // Has __uint128_t
#pragma message("Using __uint128_t to compute 64bit x 64bit to 128bit")
            __uint128_t x(a);
            x *= b;
            low = uint64_t(x), high = uint64_t(x >> 64);
#else
#if defined(HINT_WIN64) // Has _umul128
#pragma message("Using _umul128 to compute 64bit x 64bit to 128bit")
            HintULL lo, hi;
            lo = _umul128(a, b, &hi);
            low = lo, high = hi;
#else // No _umul128 or __uint128_t
#pragma message("Using basic function to compute 64bit x 64bit to 128bit")
            mul64x64to128_base(a, b, low, high);
#endif
#endif
        }

        template <typename UintTy>
        inline void mul_binary(UintTy a, UintTy b, UintTy &low, UintTy &high)
        {
            constexpr int BITS = sizeof(UintTy) * CHAR_BIT;
            uint64_t prod = uint64_t(a) * uint64_t(b);
            low = UintTy(prod), high = UintTy(prod >> BITS);
        }
        inline void mul_binary(uint64_t a, uint64_t b, uint64_t &low, uint64_t &high)
        {
            mul64x64to128(a, b, low, high);
        }

        constexpr uint32_t div128by32(uint64_t &dividend_hi64, uint64_t &dividend_lo64, uint32_t divisor)
        {
            uint32_t quot_hi32 = 0, quot_lo32 = 0;
            uint64_t dividend = dividend_hi64 >> 32;
            quot_hi32 = dividend / divisor;
            dividend %= divisor;

            dividend = (dividend << 32) | uint32_t(dividend_hi64);
            quot_lo32 = dividend / divisor;
            dividend %= divisor;
            dividend_hi64 = (uint64_t(quot_hi32) << 32) | quot_lo32;

            dividend = (dividend << 32) | uint32_t(dividend_lo64 >> 32);
            quot_hi32 = dividend / divisor;
            dividend %= divisor;

            dividend = (dividend << 32) | uint32_t(dividend_lo64);
            quot_lo32 = dividend / divisor;
            dividend %= divisor;
            dividend_lo64 = (uint64_t(quot_hi32) << 32) | quot_lo32;
            return dividend;
        }

        // 96bit integer divided by 64bit integer, input make sure the quotient smaller than 2^32.
        constexpr uint32_t div96by64to32(uint32_t dividend_hi32, uint64_t &dividend_lo64, uint64_t divisor)
        {
            if (0 == dividend_hi32)
            {
                uint32_t quotient = dividend_lo64 / divisor;
                dividend_lo64 %= divisor;
                return quotient;
            }
            uint64_t divid2 = (uint64_t(dividend_hi32) << 32) | (dividend_lo64 >> 32);
            uint64_t divis1 = divisor >> 32;
            divisor = uint32_t(divisor);
            uint64_t qhat = divid2 / divis1;
            divid2 %= divis1;
            divid2 = (divid2 << 32) | uint32_t(dividend_lo64);
            uint64_t prod = qhat * divisor;
            divis1 <<= 32;
            if (prod > divid2)
            {
                qhat--;
                prod -= divisor;
                divid2 += divis1;
                // if divid2 <= divis1, the addtion of divid2 is overflow, so prod must not be larger than divid2.
                if ((divid2 > divis1) && (prod > divid2))
                {
                    qhat--;
                    prod -= divisor;
                    divid2 += divis1;
                }
            }
            divid2 -= prod;
            dividend_lo64 = divid2;
            return uint32_t(qhat);
        }

        // 128bit integer divided by 64bit integer, input make sure the quotient smaller than 2^64.
        constexpr uint64_t div128by64to64(uint64_t dividend_hi64, uint64_t &dividend_lo64, uint64_t divisor)
        {
            int k = 0;
            if (divisor < (uint64_t(1) << 63))
            {
                k = hint::hint_clz(divisor);
                divisor <<= k; // Normalization.
                dividend_hi64 = (dividend_hi64 << k) | (dividend_lo64 >> (64 - k));
                dividend_lo64 <<= k;
            }
            uint32_t divid_hi32 = dividend_hi64 >> 32;
            uint64_t divid_lo64 = (dividend_hi64 << 32) | (dividend_lo64 >> 32);
            uint64_t quotient = div96by64to32(divid_hi32, divid_lo64, divisor);

            divid_hi32 = divid_lo64 >> 32;
            dividend_lo64 = uint32_t(dividend_lo64) | (divid_lo64 << 32);
            quotient = (quotient << 32) | div96by64to32(divid_hi32, dividend_lo64, divisor);
            dividend_lo64 >>= k;
            return quotient;
        }

        constexpr uint32_t mulMod(uint32_t a, uint32_t b, uint32_t mod)
        {
            return uint64_t(a) * b % mod;
        }
        constexpr uint64_t mulMod(uint64_t a, uint64_t b, uint64_t mod)
        {
            uint64_t prod_lo = 0, prod_hi = 0;
            mul64x64to128_base(a, b, prod_lo, prod_hi);
            div128by64to64(prod_hi, prod_lo, mod);
            return prod_lo;
        }

        // uint64_t to std::string
        inline std::string ui64to_string_base10(uint64_t input, uint8_t digits)
        {
            std::string result(digits, '0');
            for (uint8_t i = 0; i < digits; i++)
            {
                result[digits - i - 1] = static_cast<char>(input % 10 + '0');
                input /= 10;
            }
            return result;
        }
        namespace extend_int
        {
            class Uint128
            {
            public:
                Uint128() = default;
                constexpr Uint128(uint64_t l, uint64_t h = 0) : low(l), high(h) {}

                friend constexpr Uint128 operator+(Uint128 lhs, Uint128 rhs)
                {
                    rhs.low += lhs.low;
                    rhs.high += lhs.high + (rhs.low < lhs.low);
                    return rhs;
                }
                friend constexpr Uint128 operator-(Uint128 lhs, Uint128 rhs)
                {
                    rhs.low = lhs.low - rhs.low;
                    rhs.high = lhs.high - rhs.high - (rhs.low > lhs.low);
                    return rhs;
                }
                constexpr Uint128 operator+(uint64_t rhs)
                {
                    rhs = low + rhs;
                    return Uint128(rhs, high + (rhs < low));
                }
                constexpr Uint128 operator-(uint64_t rhs)
                {
                    rhs = low - rhs;
                    return Uint128(rhs, high - (rhs > low));
                }
                // Only compute the low * rhs.low
                friend Uint128 operator*(Uint128 lhs, Uint128 rhs)
                {
                    mul64x64to128(lhs.low, rhs.low, lhs.low, lhs.high);
                    return lhs;
                }
                // Only compute the low * rhs
                Uint128 operator*(uint64_t rhs) const
                {
                    Uint128 res;
                    mul64x64to128(low, rhs, res.low, res.high);
                    return res;
                }
                // Only compute the 128bit / 64 bit
                friend constexpr Uint128 operator/(Uint128 lhs, Uint128 rhs)
                {
                    return lhs / rhs.low;
                }
                // Only compute the 128bit % 64 bit
                friend constexpr Uint128 operator%(Uint128 lhs, Uint128 rhs)
                {
                    return lhs % rhs.low;
                }
                // Only compute the 128bit / 64 bit
                constexpr Uint128 operator/(uint64_t rhs) const
                {
                    Uint128 quot = *this;
                    quot.selfDivRem(rhs);
                    return quot;
                }
                // Only compute the 128bit % 64 bit
                constexpr Uint128 operator%(uint64_t rhs) const
                {
                    Uint128 quot = *this;
                    uint64_t rem = quot.selfDivRem(rhs);
                    return Uint128(rem);
                }
                constexpr Uint128 &operator+=(const Uint128 &rhs)
                {
                    return *this = *this + rhs;
                }
                constexpr Uint128 &operator-=(const Uint128 &rhs)
                {
                    return *this = *this - rhs;
                }
                constexpr Uint128 &operator+=(uint64_t rhs)
                {
                    return *this = *this + rhs;
                }
                constexpr Uint128 &operator-=(uint64_t rhs)
                {
                    return *this = *this - rhs;
                }
                // Only compute the low * rhs.low
                constexpr Uint128 &operator*=(const Uint128 &rhs)
                {
                    mul64x64to128_base(low, rhs.low, low, high);
                    return *this;
                }
                constexpr Uint128 &operator/=(const Uint128 &rhs)
                {
                    return *this = *this / rhs;
                }
                constexpr Uint128 &operator%=(const Uint128 &rhs)
                {
                    return *this = *this % rhs;
                }
                // Return *this % divisor, *this /= divisor
                constexpr uint64_t selfDivRem(uint64_t divisor)
                {
                    if ((divisor >> 32) == 0)
                    {
                        return div128by32(high, low, uint32_t(divisor));
                    }
                    uint64_t divid1 = high % divisor, divid0 = low;
                    high /= divisor;
                    low = div128by64to64(divid1, divid0, divisor);
                    return divid0;
                }
                static constexpr Uint128 mul64x64(uint64_t a, uint64_t b)
                {
                    Uint128 res{};
                    mul64x64to128_base(a, b, res.low, res.high);
                    return res;
                }
                friend constexpr bool operator<(Uint128 lhs, Uint128 rhs)
                {
                    if (lhs.high != rhs.high)
                    {
                        return lhs.high < rhs.high;
                    }
                    return lhs.low < rhs.low;
                }
                friend constexpr bool operator>(Uint128 lhs, Uint128 rhs)
                {
                    return rhs < lhs;
                }
                friend constexpr bool operator<=(Uint128 lhs, Uint128 rhs)
                {
                    return !(lhs > rhs);
                }
                friend constexpr bool operator>=(Uint128 lhs, Uint128 rhs)
                {
                    return !(lhs < rhs);
                }
                friend constexpr bool operator==(Uint128 lhs, Uint128 rhs)
                {
                    return lhs.high == rhs.high && lhs.low == rhs.low;
                }
                friend constexpr bool operator!=(Uint128 lhs, Uint128 rhs)
                {
                    return !(lhs == rhs);
                }
                constexpr Uint128 operator<<(int shift) const
                {
                    if (0 == shift)
                    {
                        return *this;
                    }
                    shift %= 128;
                    shift = shift < 0 ? shift + 128 : shift;
                    if (shift < 64)
                    {
                        return Uint128(low << shift, (high << shift) | (low >> (64 - shift)));
                    }
                    return Uint128(0, low << (shift - 64));
                }
                constexpr Uint128 operator>>(int shift) const
                {
                    if (0 == shift)
                    {
                        return *this;
                    }
                    shift %= 128;
                    shift = shift < 0 ? shift + 128 : shift;
                    if (shift < 64)
                    {
                        return Uint128((low >> shift) | (high << (64 - shift)), high >> shift);
                    }
                    return Uint128(high >> (shift - 64), 0);
                }
                constexpr Uint128 &operator<<=(int shift)
                {
                    return *this = *this << shift;
                }
                constexpr Uint128 &operator>>=(int shift)
                {
                    return *this = *this >> shift;
                }
                constexpr operator uint64_t() const
                {
                    return low;
                }
                std::string toStringBase10() const
                {
                    if (0 == high)
                    {
                        return std::to_string(low);
                    }
                    constexpr uint64_t BASE(10000'0000'0000'0000);
                    Uint128 copy(*this);
                    std::string s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    return std::to_string(uint64_t(copy.selfDivRem(BASE))) + s;
                }
                void printDec() const
                {
                    std::cout << std::dec << toStringBase10() << '\n';
                }
                void printHex() const
                {
                    std::cout << std::hex << "0x" << high << ' ' << low << std::dec << '\n';
                }
                constexpr uint64_t high64() const
                {
                    return high;
                }
                constexpr uint64_t low64() const
                {
                    return low;
                }

            private:
                uint64_t low, high;
            };

            class Uint192
            {
            public:
                using ConstRef = const Uint192 &;
                Uint192() = default;
                constexpr Uint192(uint64_t lo, uint64_t mi = 0, uint64_t hi = 0) : low(lo), mid(mi), high(hi) {}
                constexpr Uint192(Uint128 n) : low(n.low64()), mid(n.high64()), high(0) {}

                constexpr Uint192 &operator+=(ConstRef rhs)
                {
                    bool cf = false;
                    low = add_half(low, rhs.low, cf);
                    mid = add_carry(mid, rhs.mid, cf);
                    high = high + rhs.high + cf;
                    return *this;
                }
                constexpr Uint192 &operator-=(ConstRef rhs)
                {
                    bool bf = false;
                    low = sub_half(low, rhs.low, bf);
                    mid = sub_borrow(mid, rhs.mid, bf);
                    high = high - rhs.high - bf;
                    return *this;
                }
                constexpr Uint192 &operator/=(ConstRef rhs)
                {
                    return *this = *this / rhs;
                }
                constexpr Uint192 &operator%=(ConstRef rhs)
                {
                    return *this = *this % rhs;
                }
                friend constexpr Uint192 operator+(Uint192 lhs, ConstRef rhs)
                {
                    return lhs += rhs;
                }
                friend constexpr Uint192 operator-(Uint192 lhs, ConstRef rhs)
                {
                    return lhs += rhs;
                }
                constexpr Uint192 operator/(uint64_t rhs) const
                {
                    Uint192 result(*this);
                    result.selfDivRem(rhs);
                    return result;
                }
                constexpr Uint192 operator%(uint64_t rhs) const
                {
                    Uint192 result(*this);
                    return result.selfDivRem(rhs);
                }
                constexpr Uint192 subNorm(Uint192 mod) const
                {
                    bool bf = false;
                    mod.low = sub_half(low, mod.low, bf);
                    mod.mid = sub_borrow(mid, mod.mid, bf);
                    mod.high = high - mod.high - bf;
                    // mask = 0b111...1 if *this < mod
                    // res = mod if *this >= mod
                    auto mask = uint64_t(0) - uint64_t(mod.high > high), nmask = ~mask;
                    mod.low = (mod.low & nmask) | (low & mask);
                    mod.mid = (mod.mid & nmask) | (mid & mask);
                    mod.high = (mod.high & nmask) | (high & mask);
                    return mod;
                }
                constexpr Uint192 operator<<(int shift) const
                {
                    if (shift == 0)
                    {
                        return *this;
                    }
                    shift %= 192;
                    shift = shift < 0 ? shift + 192 : shift;
                    if (shift < 64)
                    {
                        return Uint192(low << shift, (mid << shift) | (low >> (64 - shift)), (high << shift) | (mid >> (64 - shift)));
                    }
                    else if (shift < 128)
                    {
                        shift -= 64;
                        return Uint192(0, low << shift, (mid << shift) | (low >> (64 - shift)));
                    }
                    return Uint192(0, 0, low << (shift - 128));
                }
                friend constexpr bool operator<(Uint192 lhs, ConstRef rhs)
                {
                    bool bf = false;
                    sub_half(lhs.low, rhs.low, bf);
                    sub_borrow(lhs.mid, rhs.mid, bf);
                    uint64_t high = lhs.high - rhs.high - bf;
                    return high > lhs.high;
                }
                friend constexpr bool operator>(ConstRef lhs, ConstRef rhs)
                {
                    return rhs < lhs;
                }
                friend constexpr bool operator<=(ConstRef lhs, ConstRef rhs)
                {
                    return !(lhs > rhs);
                }
                friend constexpr bool operator>=(ConstRef lhs, ConstRef rhs)
                {
                    return !(lhs < rhs);
                }
                friend constexpr bool operator==(ConstRef lhs, ConstRef rhs)
                {
                    return lhs.high == rhs.high && lhs.mid == rhs.mid && lhs.low == rhs.low;
                }
                friend constexpr bool operator!=(ConstRef lhs, ConstRef rhs)
                {
                    return !(lhs == rhs);
                }
                static Uint192 mul128x64(Uint128 a, uint64_t b)
                {
                    Uint192 result;
                    uint64_t lo;
                    mul64x64to128(b, a.low64(), result.low, result.mid);
                    mul64x64to128(b, a.high64(), lo, result.high);
                    result.mid += lo;
                    result.high += (result.mid < lo);
                    return result;
                }
                static constexpr Uint192 mul64x64x64(uint64_t a, uint64_t b, uint64_t c)
                {
                    Uint192 result{};
                    uint64_t lo{}, hi{};
                    mul64x64to128_base(a, b, lo, hi);
                    mul64x64to128_base(c, lo, result.low, result.mid);
                    mul64x64to128_base(c, hi, lo, result.high);
                    result.mid += lo;
                    result.high += (result.mid < lo);
                    return result;
                }
                constexpr uint64_t selfDivRem(uint64_t divisor)
                {
                    uint64_t divid1 = high % divisor, divid0 = mid;
                    high /= divisor;
                    mid = div128by64to64(divid1, divid0, divisor);
                    divid1 = divid0, divid0 = low;
                    low = div128by64to64(divid1, divid0, divisor);
                    return divid0;
                }
                constexpr Uint192 rShift64() const
                {
                    return Uint192(mid, high, 0);
                }
                constexpr operator uint64_t() const
                {
                    return low;
                }
                constexpr uint64_t high64() const
                {
                    return high;
                }
                constexpr uint64_t mid64() const
                {
                    return mid;
                }
                constexpr uint64_t low64() const
                {
                    return low;
                }
                std::string toStringBase10() const
                {
                    if (high == 0)
                    {
                        return Uint128(mid, low).toStringBase10();
                    }
                    constexpr uint64_t BASE(10000'0000'0000'0000);
                    Uint192 copy(*this);
                    std::string s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    s = ui64to_string_base10(uint64_t(copy.selfDivRem(BASE)), 16) + s;
                    return std::to_string(uint64_t(copy.selfDivRem(BASE))) + s;
                }
                void printDec() const
                {
                    std::cout << std::dec << toStringBase10() << '\n';
                }
                void printHex() const
                {
                    std::cout << std::hex << "0x" << high << ' ' << mid << ' ' << low << std::dec << '\n';
                }

            private:
                uint64_t low, mid, high;
            };

            template <typename UInt128Ty>
            constexpr uint64_t high64(const UInt128Ty &n)
            {
                return n >> 64;
            }
            constexpr uint64_t high64(const Uint128 &n)
            {
                return n.high64();
            }

#ifdef HINT_INT128
            using Uint128Defaulf = __uint128_t;
#else
            using Uint128Defaulf = Uint128;
#endif // UINT128T
        }
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

        namespace ntt
        {
            constexpr uint64_t MOD0 = 2485986994308513793, ROOT0 = 5; // 69 * 2^55 + 1
            constexpr uint64_t MOD1 = 1945555039024054273, ROOT1 = 5; // 27 * 2^56 + 1
            constexpr uint64_t MOD2 = 4179340454199820289, ROOT2 = 3; // 29 * 2^57 + 1
            constexpr uint32_t MOD3 = 998244353, ROOT3 = 3;           // 119 * 2^23 + 1
            constexpr uint32_t MOD4 = 754974721, ROOT4 = 11;          // 45 * 2^24 + 1
            constexpr uint32_t MOD5 = 469762049, ROOT5 = 3;           // 7 * 2^26 + 1

            //  Montgomery ModInt
            template <uint64_t MOD>
            class MontIntLazy
            {
            public:
                static constexpr int MOD_BITS = hint_log2(MOD) + 1;
                static_assert(MOD_BITS <= 30 || (32 < MOD_BITS && MOD_BITS <= 62), "MOD_BITS not in range [0, 30] or [33, 62]");

                using Uint128Fast = utility::extend_int::Uint128Defaulf;
                using Uint128 = utility::extend_int::Uint128;
                using IntType = typename std::conditional<MOD_BITS <= 30, uint32_t, uint64_t>::type;
                using ProdTypeFast = typename std::conditional<MOD_BITS <= 30, uint64_t, Uint128Fast>::type;
                using ProdType = typename std::conditional<MOD_BITS <= 30, uint64_t, Uint128>::type;
                static constexpr int R_BITS = sizeof(IntType) * CHAR_BIT;

                static constexpr IntType getR()
                {
                    constexpr IntType HALF = (IntType(1) << (R_BITS - 1)) % mod();
                    return HALF * 2 % mod();
                }
                static constexpr IntType R = getR();                               // R % MOD
                static constexpr IntType R2 = utility::mulMod(R, R, IntType(MOD)); // R^2 % MOD
                static constexpr IntType MOD_INV = inv_mod2pow(MOD, R_BITS);       // MOD^-1 % R
                static constexpr IntType MOD_INV_NEG = IntType(0) - MOD_INV;       // -MOD^-1 % R
                static constexpr IntType MOD2 = MOD * 2;                           // MOD * 2
                static_assert(IntType(MOD * MOD_INV) == 1, "MOD_INV not correct");

                constexpr MontIntLazy() = default;
                constexpr MontIntLazy(IntType n) : data(toMont(n)) {}

                constexpr IntType raw() const
                {
                    return data;
                }
                constexpr MontIntLazy &operator+=(MontIntLazy rhs)
                {
                    data += rhs.data;
                    data = norm2(data);
                    return *this;
                }
                constexpr MontIntLazy &operator-=(MontIntLazy rhs)
                {
                    const IntType mask = IntType(0) - (data < rhs.data);
                    data = data - rhs.data + (MOD2 & mask);
                    return *this;
                }
                constexpr MontIntLazy &operator*=(MontIntLazy rhs)
                {
                    data = redc(ProdType(data) * rhs.data);
                    return *this;
                }

                friend constexpr MontIntLazy operator+(MontIntLazy lhs, MontIntLazy rhs)
                {
                    return lhs += rhs;
                }
                friend constexpr MontIntLazy operator-(MontIntLazy lhs, MontIntLazy rhs)
                {
                    return lhs -= rhs;
                }
                friend MontIntLazy operator*(MontIntLazy lhs, MontIntLazy rhs)
                {
                    ProdTypeFast prod = ProdTypeFast(lhs.data) * rhs.data;
                    lhs.data = redcLazy(prod);
                    return lhs;
                }
                constexpr MontIntLazy operator-() const
                {
                    MontIntLazy res = this->norm1();
                    res.data = mod() - data;
                    return res;
                }
                constexpr MontIntLazy norm1() const
                {
                    MontIntLazy res{};
                    res.data = norm1(data);
                    return res;
                }
                constexpr MontIntLazy norm2() const
                {
                    MontIntLazy res{};
                    res.data = norm2(data);
                    return res;
                }
                template <IntType N = 1>
                constexpr MontIntLazy norm() const
                {
                    MontIntLazy res{};
                    res.data = norm<N>(data);
                    return res;
                }
                static constexpr IntType norm1(IntType n)
                {
                    return norm<1>(n);
                }
                static constexpr IntType norm2(IntType n)
                {
                    return norm<2>(n);
                }
                template <IntType N = 1>
                static constexpr IntType norm(IntType n)
                {
                    constexpr IntType MOD_N = mod<N>();
                    const IntType mask = IntType(0) - (n >= MOD_N);
                    return n - (MOD_N & mask);
                }
                constexpr MontIntLazy add(MontIntLazy rhs) const
                {
                    rhs.data = data + rhs.data;
                    return rhs;
                }
                constexpr MontIntLazy sub(MontIntLazy rhs) const
                {
                    rhs.data = data - rhs.data + MOD2;
                    return rhs;
                }
                constexpr operator IntType() const
                {
                    return toInt(data);
                }
                template <IntType N = 1>
                static constexpr IntType mod()
                {
                    constexpr IntType MOD_N = MOD * N;
                    return MOD_N;
                }
                static constexpr MontIntLazy montR()
                {
                    constexpr MontIntLazy res(R);
                    return res;
                }
                // R / 32 <= MOD
                constexpr MontIntLazy shrinkToMod4X(std::true_type) const
                {
                    MontIntLazy res{};
                    res.data = this->data % MOD;
                    return res;
                }
                // R / 16 <= MOD < R / 8
                constexpr MontIntLazy shrinkToMod4X(std::false_type) const
                {
                    return this->norm<8>().template norm<4>();
                }
                // R / 16 <= MOD < R / 8
                constexpr MontIntLazy shrinkToMod4(std::true_type) const
                {
                    constexpr IntType R_16 = IntType(1) << (R_BITS - 4);
                    using SMALL_MOD = std::integral_constant<bool, (MOD < R_16)>;
                    return shrinkToMod4X(SMALL_MOD{});
                }
                // R / 8 <= MOD < R / 4
                constexpr MontIntLazy shrinkToMod4(std::false_type) const
                {
                    return this->norm<4>();
                }
                // R <= MOD * 8 , R > MOD * 4
                constexpr MontIntLazy shrinkToMod4() const
                {
                    constexpr IntType R_8 = IntType(1) << (R_BITS - 3);
                    using SMALL_MOD = std::integral_constant<bool, (MOD < R_8)>;
                    return shrinkToMod4(SMALL_MOD{});
                }

                static constexpr uint32_t toMont(uint32_t n)
                {
                    return redc(uint64_t(n) * R2);
                }
                static constexpr uint32_t toInt(uint32_t n)
                {
                    return redc(uint64_t(n));
                }
                static constexpr uint64_t toMont(uint64_t n)
                {
                    uint64_t lo{}, hi{};
                    utility::mul64x64to128_base(n, R2, lo, hi);
                    return redc(Uint128{lo, hi});
                }
                static constexpr uint64_t toInt(uint64_t n)
                {
                    return redc(Uint128{n});
                }

                static constexpr uint32_t redcLazy(uint64_t n)
                {
                    uint32_t prod = uint32_t(n) * MOD_INV_NEG;
                    return (uint64_t(prod) * mod() + n) >> 32;
                }
                static constexpr uint32_t redc(uint64_t n)
                {
                    uint32_t res = redcLazy(n);
                    return norm1(res);
                }
                static uint64_t redcLazy(Uint128Fast n)
                {
                    uint64_t prod1 = uint64_t(n) * MOD_INV_NEG;
                    auto prod2 = Uint128Fast(prod1) * mod() + n;
                    return utility::extend_int::high64(prod2);
                }
                static constexpr uint64_t redc(Uint128 n)
                {
                    uint64_t prod1 = uint64_t(n) * MOD_INV_NEG, lo{}, hi{};
                    utility::mul64x64to128_base(prod1, mod(), lo, hi);
                    Uint128 prod2{lo, hi};
                    prod2 += n;
                    return norm<1>(prod2.high64());
                }

                static constexpr MontIntLazy one()
                {
                    constexpr MontIntLazy res{1};
                    return res;
                }
                static constexpr MontIntLazy negOne()
                {
                    constexpr MontIntLazy res(mod() - 1);
                    return res;
                }
                constexpr MontIntLazy inv() const
                {
                    return hint::qpow(*this, mod() - 2);
                }

            private:
                IntType data;
            };

            template <uint64_t MOD>
            constexpr int MontIntLazy<MOD>::MOD_BITS;
            template <uint64_t MOD>
            constexpr int MontIntLazy<MOD>::R_BITS;
            template <uint64_t MOD>
            constexpr typename MontIntLazy<MOD>::IntType MontIntLazy<MOD>::R;
            template <uint64_t MOD>
            constexpr typename MontIntLazy<MOD>::IntType MontIntLazy<MOD>::R2;
            template <uint64_t MOD>
            constexpr typename MontIntLazy<MOD>::IntType MontIntLazy<MOD>::MOD_INV;
            template <uint64_t MOD>
            constexpr typename MontIntLazy<MOD>::IntType MontIntLazy<MOD>::MOD_INV_NEG;
            template <uint64_t MOD>
            constexpr typename MontIntLazy<MOD>::IntType MontIntLazy<MOD>::MOD2;

            // in: in_out0<4p, in_ou1<4p
            // out: in_out0<4p, in_ou1<4p
            template <typename ModIntType>
            inline void dit_butterfly2(ModIntType &in_out0, ModIntType &in_out1, const ModIntType &omega)
            {
                auto x = in_out0.norm2();
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
            template <typename ModInt>
            class BinRevTable
            {
            public:
                static constexpr int MAX_LOG_LEN = CHAR_BIT * sizeof(ModInt);
                static constexpr size_t MAX_LEN = size_t(1) << MAX_LOG_LEN;

                using IntType = std::conditional_t<sizeof(ModInt) <= 4, uint32_t, uint64_t>;

                BinRevTable(IntType root_in, size_t factor = 1, size_t div = 1) : root(root_in)
                {
                    constexpr int LOG_LEN = hint::hint_ctz(ModInt::mod() - 1);
                    assert(hint::is_2pow(div));
                    table[0] = getOmega(2 * div, factor, false);
                    for (int i = 1; i <= LOG_LEN; i++)
                    {
                        const size_t rev_indx = 1;
                        const size_t last_indx = ((size_t(1) << i) - 1) << 1;
                        table[i] = getOmega((size_t(1) << (i + 1)) * div, (last_indx - rev_indx) * factor, true);
                    }
                }

                ModInt evenToOdd(ModInt last_even) const
                {
                    last_even = last_even * table[0];
                    return last_even.norm1();
                }

                ModInt omega1() const
                {
                    return table[0];
                }

                ModInt getNext(ModInt last, IntType indx) const
                {
                    if (0 == indx)
                    {
                        return ModInt::one();
                    }
                    indx = indx % 2 ? 0 : hint_ctz(indx);
                    last = last * table[indx];
                    return last.norm1();
                }

                ModInt getOmega(size_t n, size_t index, bool conj = false) const
                {
                    index = (ModInt::mod() - 1) / n * index;
                    ModInt omega = qpow(root, index);
                    return conj ? omega.inv() : omega;
                }

            private:
                ModInt table[MAX_LOG_LEN];
                ModInt root;
            };

            template <uint64_t MOD, uint64_t ROOT>
            struct NTT
            {
                static constexpr uint64_t mod()
                {
                    return MOD;
                }
                static constexpr uint64_t root()
                {
                    return ROOT;
                }
                static constexpr uint64_t rootInv()
                {
                    constexpr uint64_t IROOT = mod_inv<int64_t>(ROOT, MOD);
                    return IROOT;
                }

                static_assert(root() < mod(), "ROOT must be smaller than MOD");
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

                using ModInt = MontIntLazy<MOD>;
                using IntType = typename ModInt::IntType;
                using TableType = BinRevTable<ModInt>;
                static_assert(std::is_trivial<ModInt>::value, "ModInt must be trivial");
                static const TableType table, itable;

                static constexpr size_t L1_BYTE = size_t(1) << 15;
                static constexpr size_t LONG_BYTE = size_t(1) << 29;
                static constexpr size_t ITER_THRESHOLD = L1_BYTE / sizeof(ModInt);
                static constexpr size_t LONG_THRESHOLD = LONG_BYTE / sizeof(ModInt);
                static constexpr size_t SHORT_THRESHOLD = 15;

                static size_t findFitLen(size_t conv_len) noexcept
                {
                    size_t result = findFitLen(conv_len, SHORT_THRESHOLD);
                    // 长度为2的幂次且过长时将造成L1缓存颠簸，需要将长度转为非2的幂次
                    constexpr size_t DIV_LEN = int_floor2(SHORT_THRESHOLD), MUL_LEN = DIV_LEN + 1;
                    static_assert(MUL_LEN <= SHORT_THRESHOLD, "MUL_LEN can't be larger than SHORT_THRESHOLD");
                    if (result >= LONG_THRESHOLD && is_2pow(result))
                    {
                        result /= DIV_LEN;
                        result *= MUL_LEN;
                    }
                    return result;
                }
                static void convolution(IntType in_out[], IntType in[], size_t conv_len,
                                        ModInt weight = ModInt::one()) noexcept
                {
                    assert(checkConvLen(conv_len)); // check if conv_len is too long
                    auto p1 = reinterpret_cast<ModInt *>(in_out), p2 = reinterpret_cast<ModInt *>(in);
                    if (conv_len <= SHORT_THRESHOLD)
                    {
                        std::copy(in_out, in_out + conv_len, p1);
                        std::copy(in, in + conv_len, p2);
                        convolutionCyclicShort(p1, p2, conv_len);
                        for (size_t i = 0; i < conv_len; i++)
                        {
                            in_out[i] = (p1[i] * weight).norm1();
                        }
                        return;
                    }
                    ModInt buffer[64], ibuffer[64];
                    convolutionCyclic(in_out, in, conv_len, conv_len, conv_len, buffer, ibuffer, weight);
                }

                static void convolution(IntType in_out[], IntType in[], size_t len1, size_t len2, size_t conv_len,
                                        ModInt weight = ModInt::one()) noexcept
                {
                    assert(len1 + len2 - 1 <= conv_len);
                    assert(checkConvLen(conv_len)); // check if conv_len is too long
                    auto p1 = reinterpret_cast<ModInt *>(in_out), p2 = reinterpret_cast<ModInt *>(in);
                    if (conv_len <= ITER_THRESHOLD)
                    {
                        std::fill(in_out + len1, in_out + conv_len, IntType{});
                        std::fill(in + len2, in + conv_len, IntType{});
                    }
                    if (conv_len <= SHORT_THRESHOLD)
                    {
                        std::copy(in_out, in_out + conv_len, p1);
                        std::copy(in, in + conv_len, p2);
                        convolutionCyclicShort(p1, p2, conv_len);
                        for (size_t i = 0; i < conv_len; i++)
                        {
                            in_out[i] = (p1[i] * weight).norm1();
                        }
                        return;
                    }
                    ModInt buffer[64], ibuffer[64];
                    convolutionCyclic(in_out, in, len1, len2, conv_len, buffer, ibuffer, weight);
                }

            private:
                static size_t forwardCyclic(ModInt in_out[], size_t len, ModInt buffer[], bool assign_buf) noexcept
                {
                    auto p_buf = buffer;
                    for (size_t rank = len; rank > SHORT_THRESHOLD * 2; rank /= 2, p_buf++)
                    {
                        size_t stride = rank / 2, i = 1;
                        auto it0 = in_out, it1 = it0 + stride;
                        for (const auto end = it1; it0 < end; it0++, it1++)
                        {
                            auto t0 = it0[0].norm2(), t1 = it1[0].norm2();
                            it0[0] = t0.add(t1), it1[0] = t0.sub(t1);
                        }
                        ModInt omega = ModInt::one();
                        for (auto begin = in_out + rank; begin < in_out + len; begin += rank, i++)
                        {
                            omega = table.getNext(omega, i);
                            it0 = begin, it1 = it0 + stride;
                            for (const auto end = it1; it0 < end; it0++, it1++)
                            {
                                dit_butterfly2(it0[0], it1[0], omega);
                            }
                        }
                        if (assign_buf)
                        {
                            p_buf[0] = omega;
                        }
                    }
                    return p_buf - buffer;
                }
                static size_t forwardIter(ModInt in_out[], size_t len, ModInt buffer[], size_t idx, bool assign_buf) noexcept
                {
                    auto p_buf = buffer;
                    for (size_t rank = len; rank > SHORT_THRESHOLD * 2; rank /= 2, idx *= 2, p_buf++)
                    {
                        size_t stride = rank / 2, i = idx;
                        ModInt omega = p_buf[0];
                        for (auto begin = in_out; begin < in_out + len; begin += rank, i++)
                        {
                            auto it0 = begin, it1 = it0 + stride;
                            omega = table.getNext(omega, i);
                            for (const auto end = it1; it0 < end; it0++, it1++)
                            {
                                dit_butterfly2(it0[0], it1[0], omega);
                            }
                        }
                        if (assign_buf)
                        {
                            p_buf[0] = omega;
                        }
                    }
                    return p_buf - buffer;
                }
                static void backwardCyclic(ModInt in_out[], size_t len, size_t rank, ModInt ibuffer[]) noexcept
                {
                    for (; rank <= len; rank *= 2, ibuffer++)
                    {
                        size_t stride = rank / 2, i = 1;
                        auto it0 = in_out, it1 = it0 + stride, end = it1;
                        for (; it0 < end; it0++, it1++)
                        {
                            transform2(it0[0], it1[0]);
                        }
                        ModInt omega = ModInt::one();
                        for (auto begin = in_out + rank; begin < in_out + len; begin += rank, i++)
                        {
                            omega = itable.getNext(omega, i);
                            it0 = begin, it1 = it0 + stride;
                            for (const auto end = it1; it0 < end; it0++, it1++)
                            {
                                dif_butterfly2(it0[0], it1[0], omega);
                            }
                        }
                        ibuffer[0] = omega;
                    }
                }
                static void backwardIter(ModInt in_out[], size_t len, size_t rank, ModInt ibuffer[], size_t idx) noexcept
                {
                    for (; rank <= len; rank *= 2, idx /= 2, ibuffer++)
                    {
                        size_t stride = rank / 2, i = idx;
                        ModInt omega = ibuffer[0];
                        for (auto begin = in_out; begin < in_out + len; begin += rank, i++)
                        {
                            auto it0 = begin, it1 = it0 + stride;
                            omega = itable.getNext(omega, i);
                            for (const auto end = it1; it0 < end; it0++, it1++)
                            {
                                dif_butterfly2(it0[0], it1[0], omega);
                            }
                        }
                        ibuffer[0] = omega;
                    }
                }

                static void forwardIter(ModInt inout0[], ModInt inout1[], size_t rank, ModInt omega) noexcept
                {
                    auto it0 = inout0, it1 = it0 + rank, end = it1;
                    auto it2 = inout1, it3 = it2 + rank;
                    for (; it0 < end; it0++, it1++, it2++, it3++)
                    {
                        dit_butterfly2(it0[0], it1[0], omega);
                        dit_butterfly2(it2[0], it3[0], omega);
                    }
                }

                static size_t convolutionIterCyclic(ModInt in_out[], ModInt in[], size_t conv_len,
                                                    ModInt buffer[], ModInt ibuffer[]) noexcept
                {
                    if (in_out != in)
                    {
                        forwardCyclic(in_out, conv_len, buffer, false);
                    }
                    size_t layers = forwardCyclic(in, conv_len, buffer, true) + 1;
                    size_t rank = conv_len >> layers;
                    ModInt omega = buffer[layers];
                    auto it0 = in_out, it1 = in, end = it0 + conv_len;
                    for (size_t i = 0; it0 < end; it0 += rank * 2, it1 += rank * 2, i++)
                    {
                        omega = table.getNext(omega, i);
                        forwardIter(it0, it1, rank, omega);
                        convolutionShort(it0, it1, rank, omega);
                        convolutionShort(it0 + rank, it1 + rank, rank, -omega);
                    }
                    buffer[layers] = omega;
                    size_t factor = size_t(1) << layers;
                    backwardCyclic(in_out, conv_len, rank * 2, ibuffer);
                    return factor;
                }
                static void convolutionIter(ModInt in_out[], ModInt in[], size_t conv_len,
                                            ModInt buffer[], ModInt ibuffer[], size_t idx) noexcept
                {
                    if (in_out != in)
                    {
                        forwardIter(in_out, conv_len, buffer, idx, false);
                    }
                    size_t layers = forwardIter(in, conv_len, buffer, idx, true) + 1;
                    size_t rank = conv_len >> layers;
                    ModInt omega = buffer[layers];
                    idx <<= (layers - 1);
                    auto it0 = in_out, it1 = in, end = it0 + conv_len;
                    for (size_t i = idx; it0 < end; it0 += rank * 2, it1 += rank * 2, i++)
                    {
                        omega = table.getNext(omega, i);
                        forwardIter(it0, it1, rank, omega);
                        convolutionShort(it0, it1, rank, omega);
                        convolutionShort(it0 + rank, it1 + rank, rank, -omega);
                    }
                    buffer[layers] = omega;
                    backwardIter(in_out, conv_len, rank * 2, ibuffer, idx);
                }

                static void convolutionCyclic(IntType in_out[], IntType in[], size_t len1, size_t len2, size_t conv_len,
                                              ModInt buffer[], ModInt ibuffer[], ModInt weight) noexcept
                {
                    auto p1 = reinterpret_cast<ModInt *>(in_out), p2 = reinterpret_cast<ModInt *>(in);
                    if (conv_len <= ITER_THRESHOLD)
                    {
                        for (size_t i = 0; i < conv_len; i++)
                        {
                            p1[i] = p1[i].shrinkToMod4();
                            p2[i] = p2[i].shrinkToMod4();
                        }
                        size_t factor = convolutionIterCyclic(p1, p2, conv_len, buffer, ibuffer);
                        ModInt inv = ModInt(factor).inv() * weight;
                        inv *= ModInt::montR();
                        for (size_t i = 0; i < conv_len; i++)
                        {
                            p1[i] = (inv * p1[i]).norm1();
                        }
                        return;
                    }
                    const size_t stride = conv_len / 4;
                    ModInt omega = table.omega1();
                    auto forwardIter = [stride, omega](ModInt inout[], size_t len)
                    {
                        auto it0 = inout, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride;
                        const auto end3 = inout + len, end2 = std::min(end3, it2 + stride);
                        const auto end1 = std::min(end3, it1 + stride), end0 = std::min(end3, it0 + stride);
                        for (; it3 < end3; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0].shrinkToMod4().norm2(), t1 = it1[0].shrinkToMod4().norm2();
                            auto t2 = it2[0].shrinkToMod4().norm2(), t3 = it3[0].shrinkToMod4().norm2();
                            transform2(t0, t2);
                            auto diff = t1.sub(t3);
                            t1 = t1 + t3;
                            t3 = diff * omega;
                            it0[0] = t0.add(t1), it1[0] = t0.sub(t1), it2[0] = t2.add(t3), it3[0] = t2.sub(t3);
                        }
                        for (; it2 < end2; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0].shrinkToMod4().norm2(), t1 = it1[0].shrinkToMod4().norm2();
                            auto t2 = it2[0].shrinkToMod4().norm2(), t3 = t1;
                            transform2(t0, t2);
                            t3 = t3 * omega;
                            it0[0] = t0.add(t1), it1[0] = t0.sub(t1), it2[0] = t2.add(t3), it3[0] = t2.sub(t3);
                        }
                        for (; it1 < end1; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0].shrinkToMod4().norm2(), t1 = it1[0].shrinkToMod4().norm2();
                            auto t2 = t0, t3 = t1;
                            t3 = t3 * omega;
                            it0[0] = t0.add(t1), it1[0] = t0.sub(t1), it2[0] = t2.add(t3), it3[0] = t2.sub(t3);
                        }
                        for (; it0 < end0; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0].shrinkToMod4();
                            it0[0] = t0, it1[0] = t0, it2[0] = t0, it3[0] = t0;
                        }
                        if (it0 < inout + stride)
                        {
                            const size_t size = (inout + stride - it0) * sizeof(ModInt);
                            std::memset(&it0[0], 0, size);
                            std::memset(&it1[0], 0, size);
                            std::memset(&it2[0], 0, size);
                            std::memset(&it3[0], 0, size);
                        }
                    };
                    forwardIter(p1, len1);
                    if (p2 != p1)
                    {
                        forwardIter(p2, len2);
                    }
                    size_t ret = convolutionCyclic(p1, p2, stride, buffer, ibuffer);
                    convolution(p1 + stride, p2 + stride, stride, buffer, ibuffer, 1);
                    convolution(p1 + stride * 2, p2 + stride * 2, stride, buffer, ibuffer, 2);
                    convolution(p1 + stride * 3, p2 + stride * 3, stride, buffer, ibuffer, 3);
                    auto it0 = p1, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride, end = it1;
                    ModInt omega_inv = itable.omega1();
                    ModInt inv = ModInt(ret * 4).inv() * weight;
                    inv *= ModInt::montR();
                    for (; it0 < end; it0++, it1++, it2++, it3++)
                    {
                        auto t0 = it0[0], t1 = it1[0], t2 = it2[0], t3 = it3[0];
                        transform2(t0, t1);
                        auto diff = t2.sub(t3);
                        t2 = t2 + t3;
                        t3 = diff * omega_inv;
                        it0[0] = (t0.add(t2) * inv).norm1(), it1[0] = (t1.add(t3) * inv).norm1();
                        it2[0] = (t0.sub(t2) * inv).norm1(), it3[0] = (t1.sub(t3) * inv).norm1();
                    }
                }
                static size_t convolutionCyclic(ModInt in_out[], ModInt in[], size_t conv_len,
                                                ModInt buffer[], ModInt ibuffer[]) noexcept
                {
                    if (conv_len <= ITER_THRESHOLD)
                    {
                        return convolutionIterCyclic(in_out, in, conv_len, buffer, ibuffer);
                    }
                    const size_t stride = conv_len / 4;
                    ModInt omega = buffer[0] = table.omega1();
                    auto forwardIter = [stride, omega](ModInt inout[])
                    {
                        auto it0 = inout, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride, end = it1;
                        for (; it0 < end; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0].norm2(), t1 = it1[0].norm2(), t2 = it2[0].norm2(), t3 = it3[0].norm2();
                            transform2(t0, t2);
                            auto diff = t1.sub(t3);
                            t1 = t1 + t3;
                            t3 = diff * omega;
                            it0[0] = t0.add(t1), it1[0] = t0.sub(t1), it2[0] = t2.add(t3), it3[0] = t2.sub(t3);
                        }
                    };
                    forwardIter(in_out);
                    if (in != in_out)
                    {
                        forwardIter(in);
                    }
                    buffer++;
                    ibuffer++;
                    size_t ret = convolutionCyclic(in_out, in, stride, buffer, ibuffer);
                    convolution(in_out + stride, in + stride, stride, buffer, ibuffer, 1);
                    convolution(in_out + stride * 2, in + stride * 2, stride, buffer, ibuffer, 2);
                    convolution(in_out + stride * 3, in + stride * 3, stride, buffer, ibuffer, 3);
                    ibuffer--;
                    ModInt omega_inv = ibuffer[0] = itable.omega1();
                    auto it0 = in_out, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride, end = it1;
                    for (; it0 < end; it0++, it1++, it2++, it3++)
                    {
                        auto t0 = it0[0], t1 = it1[0], t2 = it2[0], t3 = it3[0];
                        transform2(t0, t1);
                        auto diff = t2.sub(t3);
                        t2 = t2 + t3;
                        t3 = diff * omega_inv;
                        it0[0] = t0 + t2, it1[0] = t1 + t3, it2[0] = t0 - t2, it3[0] = t1 - t3;
                    }
                    return ret * 4;
                }
                static void convolution(ModInt in_out[], ModInt in[], size_t conv_len,
                                        ModInt buffer[], ModInt ibuffer[], size_t idx) noexcept
                {
                    if (conv_len <= ITER_THRESHOLD)
                    {
                        convolutionIter(in_out, in, conv_len, buffer, ibuffer, idx);
                        return;
                    }
                    const size_t stride = conv_len / 4;
                    idx *= 2;
                    ModInt omega1 = table.getNext(buffer[0], idx), omega2 = buffer[0] = table.evenToOdd(omega1);
                    ModInt omega0 = (omega1 * omega1).norm1();
                    auto forwardIter = [stride, omega0, omega1, omega2](ModInt inout[])
                    {
                        auto it0 = inout, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride, end = it1;
                        for (; it0 < end; it0++, it1++, it2++, it3++)
                        {
                            auto t0 = it0[0], t1 = it1[0], t2 = it2[0], t3 = it3[0];
                            dit_butterfly2(t0, t2, omega0);
                            dit_butterfly2(t1, t3, omega0);
                            dit_butterfly2(t0, t1, omega1);
                            dit_butterfly2(t2, t3, omega2);
                            it0[0] = t0, it1[0] = t1, it2[0] = t2, it3[0] = t3;
                        }
                    };
                    forwardIter(in_out);
                    if (in != in_out)
                    {
                        forwardIter(in);
                    }
                    buffer++;
                    ibuffer++;
                    idx *= 2;
                    convolution(in_out, in, stride, buffer, ibuffer, idx);
                    convolution(in_out + stride, in + stride, stride, buffer, ibuffer, idx + 1);
                    convolution(in_out + stride * 2, in + stride * 2, stride, buffer, ibuffer, idx + 2);
                    convolution(in_out + stride * 3, in + stride * 3, stride, buffer, ibuffer, idx + 3);
                    ibuffer--;
                    idx /= 2;
                    ModInt omega_inv1 = itable.getNext(ibuffer[0], idx), omega_inv2 = ibuffer[0] = itable.evenToOdd(omega_inv1);
                    ModInt omega_inv0 = (omega_inv1 * omega_inv1).norm1();
                    auto it0 = in_out, it1 = it0 + stride, it2 = it1 + stride, it3 = it2 + stride, end = it1;
                    for (; it0 < end; it0++, it1++, it2++, it3++)
                    {
                        auto t0 = it0[0], t1 = it1[0], t2 = it2[0], t3 = it3[0];
                        dif_butterfly2(t0, t1, omega_inv1);
                        dif_butterfly2(t2, t3, omega_inv2);
                        dif_butterfly2(t0, t2, omega_inv0);
                        dif_butterfly2(t1, t3, omega_inv0);
                        it0[0] = t0, it1[0] = t1, it2[0] = t2, it3[0] = t3;
                    }
                }
                static void convolutionLinear(const ModInt in1[], const ModInt in2[], ModInt out[], size_t conv_len) noexcept
                {
                    if (0 == conv_len)
                    {
                        return;
                    }
                    ModInt x = in1[0].norm2().norm1();
                    for (size_t i = 0; i < conv_len; i++)
                    {
                        out[i] = in2[i] * x;
                    }
                    const size_t last_idx = conv_len - 1;
                    for (size_t i = 1; i < conv_len; i++)
                    {
                        x = in1[i].norm2().norm1();
                        auto out_it = out + i;
                        for (size_t j = 0; j < last_idx; j++)
                        {
                            out_it[j] += in2[j] * x;
                        }
                        out_it[last_idx] = in2[last_idx] * x;
                    }
                }
                // in_out = in_out * in % (x ^ conv_len - 1)
                static void convolutionCyclicShort(ModInt in_out[], ModInt in[], size_t conv_len) noexcept
                {
                    ModInt temp[SHORT_THRESHOLD * 2];
                    const size_t rem_len = conv_len - conv_len % 4;
                    convolutionLinear(in_out, in, temp, conv_len);
                    const size_t last_idx = conv_len - 1;
                    auto cyclic_it = temp + conv_len;
                    for (size_t i = 0; i < last_idx; i++)
                    {
                        in_out[i] = (temp[i] + cyclic_it[i]);
                    }
                    in_out[last_idx] = temp[last_idx];
                }
                // in_out = in_out * in % (x ^ conv_len - omega)
                static void convolutionShort(ModInt in_out[], ModInt in[], size_t conv_len, ModInt omega) noexcept
                {
                    ModInt temp[SHORT_THRESHOLD * 2];
                    const size_t rem_len = conv_len - conv_len % 4;
                    convolutionLinear(in_out, in, temp, conv_len);
                    const size_t last_idx = conv_len - 1;
                    auto cyclic_it = temp + conv_len;
                    for (size_t i = 0; i < last_idx; i++)
                    {
                        in_out[i] = temp[i] + cyclic_it[i] * omega;
                    }
                    in_out[last_idx] = temp[last_idx];
                }
                static constexpr bool checkConvLen(size_t conv_len) noexcept
                {
                    int tail_zeros = hint_ctz(conv_len);
                    size_t head = conv_len >> tail_zeros;
                    if (head > SHORT_THRESHOLD)
                    {
                        return false;
                    }
                    while (head * 2 <= SHORT_THRESHOLD && tail_zeros > 0)
                    {
                        head *= 2;
                        tail_zeros--;
                    }
                    return (size_t(1) << tail_zeros) <= NTT_MAX_LEN;
                }
                static constexpr size_t findFitLen(size_t len, size_t factor_range) noexcept
                {
                    const size_t conv_len = hint::int_ceil2(len);
                    size_t result = conv_len;
                    int shift = 2;
                    for (size_t i = 3; i <= factor_range; i += 2)
                    {
                        if ((size_t(1) << shift) <= i)
                        {
                            shift++;
                        }
                        size_t try_len = (conv_len >> shift) * i;
                        if (try_len >= len && try_len < result)
                        {
                            result = try_len;
                        }
                    }
                    return result;
                }
            };
            template <uint64_t MOD, uint64_t ROOT>
            const typename NTT<MOD, ROOT>::TableType NTT<MOD, ROOT>::table{NTT<MOD, ROOT>::root(), 1, 2};
            template <uint64_t MOD, uint64_t ROOT>
            const typename NTT<MOD, ROOT>::TableType NTT<MOD, ROOT>::itable{NTT<MOD, ROOT>::rootInv(), 1, 2};
            template <uint64_t MOD, uint64_t ROOT>
            constexpr int NTT<MOD, ROOT>::MOD_BITS;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr int NTT<MOD, ROOT>::MAX_LOG_LEN;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::NTT_MAX_LEN;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::L1_BYTE;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::LONG_BYTE;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::ITER_THRESHOLD;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::LONG_THRESHOLD;
            template <uint64_t MOD, uint64_t ROOT>
            constexpr size_t NTT<MOD, ROOT>::SHORT_THRESHOLD;

            using NTT64_1 = NTT<MOD0, ROOT0>;
            using NTT64_2 = NTT<MOD1, ROOT1>;
            using NTT64_3 = NTT<MOD2, ROOT2>;
            using NTT32_1 = NTT<MOD3, ROOT3>;
            using NTT32_2 = NTT<MOD4, ROOT4>;
            using NTT32_3 = NTT<MOD5, ROOT5>;
        }
    }
}

template <typename ModInt>
class BinRevTable
{
public:
    static constexpr int MAX_LOG_LEN = CHAR_BIT * sizeof(ModInt);
    static constexpr size_t MAX_LEN = size_t(1) << MAX_LOG_LEN;

    BinRevTable(uint64_t root_in, size_t factor = 1, size_t div = 1) : root(root_in)
    {
        assert(hint::is_2pow(div));
        table[0] = getOmega(2 * div, factor);
        inv_table[0] = getOmega(2 * div, factor, true);
        constexpr int LOG_LEN = hint::hint_ctz(ModInt::mod() - 1);
        for (int i = 1; i <= LOG_LEN; i++)
        {
            const size_t rev_indx = 1;
            const size_t last_indx = ((size_t(1) << i) - 1) << 1;
            table[i] = getOmega((size_t(1) << (i + 1)) * div, (last_indx - rev_indx) * factor, true);
            inv_table[i] = getOmega((size_t(1) << (i + 1)) * div, (last_indx - rev_indx) * factor, false);
        }
    }

    ModInt evenToOdd(ModInt last_even) const
    {
        return last_even * table[0];
    }

    ModInt getNext(ModInt last, size_t indx) const
    {
        return last * table[hint::hint_ctz(indx)];
    }
    ModInt getNextInv(ModInt last, size_t indx) const
    {
        return last * inv_table[hint::hint_ctz(indx)];
    }

    ModInt getOmega(size_t n, size_t index, bool conj = false) const
    {
        index = (ModInt::mod() - 1) / n * index;
        auto res = hint::qpow(root, index);
        if (conj)
        {
            res = hint::qpow(res, ModInt::mod() - 2);
        }
        return res;
    }

private:
    ModInt table[MAX_LOG_LEN], inv_table[MAX_LOG_LEN];
    ModInt cur, root;
    size_t index;
};
constexpr uint32_t bitrev32(uint32_t n)
{
    constexpr uint32_t mask55 = 0x55555555;
    constexpr uint32_t mask33 = 0x33333333;
    constexpr uint32_t mask0f = 0x0f0f0f0f;
    constexpr uint32_t maskff = 0x00ff00ff;
    n = ((n & mask55) << 1) | ((n >> 1) & mask55);
    n = ((n & mask33) << 2) | ((n >> 2) & mask33);
    n = ((n & mask0f) << 4) | ((n >> 4) & mask0f);
    n = ((n & maskff) << 8) | ((n >> 8) & maskff);
    return (n << 16) | (n >> 16);
}
constexpr size_t bitrev(size_t n, int len)
{
    return bitrev32(n) >> (32 - len);
}
// using Mint = hint::transform::ntt::MontInt32Lazy<998244353>;
// BinRevTable<Mint> table(3, 1, 1);

constexpr size_t SHORT_LEN = 15;

template <typename Mint>
inline void conv(const Mint in1[], const Mint in2[], Mint out[], size_t len)
{
    const size_t rem_len = len - len % 4;
    for (size_t i = 0; i < len; i++, out++)
    {
        Mint x = in1[i];
        size_t j = 0;
        for (; j < rem_len; j += 4)
        {
            out[j] += x * in2[j];
            out[j + 1] += x * in2[j + 1];
            out[j + 2] += x * in2[j + 2];
            out[j + 3] += x * in2[j + 3];
        }
        for (; j < len; j++)
        {
            out[j] += x * in2[j];
        }
    }
}
template <typename Mint>
inline void conv_short(Mint in_out[], Mint in[], size_t len, Mint omega)
{
    Mint temp[SHORT_LEN * 2]{};
    conv(in_out, in, temp, len);
    for (size_t i = 0; i < len; i++)
    {
        in_out[i] = temp[i] + temp[i + len] * omega;
    }
}
template <typename Mint>
inline void conv_short_cyclic(Mint in_out[], Mint in[], size_t len)
{
    Mint temp[SHORT_LEN * 2]{};
    conv(in_out, in, temp, len);
    for (size_t i = 0; i < len; i++)
    {
        in_out[i] = temp[i] + temp[i + len];
    }
}
template <typename Mint>
inline void conv_short_negacyclic(Mint in_out[], Mint in[], size_t len)
{
    Mint temp[SHORT_LEN * 2]{};
    conv(in_out, in, temp, len);
    for (size_t i = 0; i < len; i++)
    {
        in_out[i] = temp[i] - temp[i + len];
    }
}
template <typename Mint>
inline void conv(Mint in_out[], Mint in[], size_t len, Mint buf[], Mint ibuf[], const BinRevTable<Mint> &tb, size_t idx = 0)
{
    buf[0] = tb.getNext(buf[0], idx);
    ibuf[0] = tb.getNextInv(ibuf[0], idx);
    Mint omega = buf[0];
    size_t stride = len / 2;
    auto forwardIter = [stride](Mint inout[], Mint o)
    {
        for (size_t i = 0; i < stride; i++)
        {
            auto x = inout[i], y = inout[i + stride] * o;
            inout[i] = x + y;
            inout[i + stride] = x - y;
        }
    };
    std::cout << "idx: " << idx << "\t";
    std::cout << omega << std::endl;
    forwardIter(in_out, omega);
    forwardIter(in, omega);
    if (stride <= SHORT_LEN)
    {
        // std::cout << "in:" << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in_out[i] << ' ';
        // }
        // std::cout << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in[i] << ' ';
        // }
        // std::cout << std::endl;
        conv_short(in_out, in, stride, omega);
        conv_short(in_out + stride, in + stride, stride, -omega);
        // std::cout << "out:" << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in_out[i] << ' ';
        // }
        // std::cout << std::endl;
    }
    else
    {
        conv(in_out, in, stride, buf + 1, ibuf + 1, tb, idx * 2);
        conv(in_out + stride, in + stride, stride, buf + 1, ibuf + 1, tb, idx * 2 + 1);
    }
    Mint omega_inv = ibuf[0];
    std::cout << "inv idx: " << idx << "\t";
    std::cout << omega_inv << std::endl;
    for (size_t i = 0; i < stride; i++)
    {
        auto x = in_out[i], y = in_out[i + stride];
        in_out[i] = (x + y);
        in_out[i + stride] = (x - y) * omega_inv;
    }
}
template <typename Mint>
inline size_t convCyclic(Mint in_out[], Mint in[], size_t len, Mint buf[], Mint ibuf[], const BinRevTable<Mint> &tb)
{
    size_t stride = len / 2;
    auto forwardIter = [stride](Mint inout[])
    {
        for (size_t i = 0; i < stride; i++)
        {
            auto x = inout[i], y = inout[i + stride];
            inout[i] = x + y;
            inout[i + stride] = x - y;
        }
    };
    forwardIter(in_out);
    forwardIter(in);
    // std::cout << "len: " << len << std::endl;
    size_t ret = 1;
    if (stride <= SHORT_LEN)
    {
        // std::cout << "in:" << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in_out[i] << ' ';
        // }
        // std::cout << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in[i] << ' ';
        // }
        // std::cout << std::endl;
        conv_short_cyclic(in_out, in, stride);
        conv_short_negacyclic(in_out + stride, in + stride, stride);
        // std::cout << "out:" << std::endl;
        // for (size_t i = 0; i < len; i++)
        // {
        //     std::cout << in_out[i] << ' ';
        // }
        // std::cout << std::endl;
    }
    else
    {
        ret = convCyclic(in_out, in, stride, buf + 1, ibuf + 1, tb);
        conv(in_out + stride, in + stride, stride, buf + 1, ibuf + 1, tb, 1);
    }
    for (size_t i = 0; i < stride; i++)
    {
        auto x = in_out[i], y = in_out[i + stride];
        in_out[i] = (x + y);
        in_out[i + stride] = (x - y);
    }
    return ret * 2;
}
template <typename Mint>
inline void conv_cyclic(Mint in_out[], Mint in[], size_t len)
{
    if (len == 0)
    {
        return;
    }
    Mint buffer[64]{}, ibuffer[64]{};
    for (size_t i = 0; i < 64; i++)
    {
        buffer[i] = ibuffer[i] = 1;
    }
    BinRevTable<Mint> table(3, 1, 2);
    size_t ret = convCyclic(in_out, in, len, buffer, ibuffer, table);
    Mint inv = hint::qpow(Mint(ret), Mint::mod() - 2);
    for (size_t i = 0; i < len; i++)
    {
        in_out[i] *= inv;
    }
}

void test_rev_modint()
{
    using Mint = hint::transform::ntt::MontIntLazy<998244353>;
    BinRevTable<Mint> table(3, 1, 2);
    size_t len = 64;
    int log_len = hint::hint_log2(len);
    std::cout << "len: " << len << std::endl;
    std::cout << "log_len: " << log_len << std::endl;
    for (size_t i = 0; i < len / 2; i++)
    {
        std::cout << i << ": " << std::endl;
        std::cout << table.getOmega(len, bitrev(i, log_len - 1), false) << " " << table.getOmega(len, bitrev(i, log_len - 1), true) << std::endl;
    }
}
size_t find_conv_len(size_t len, size_t factor_range)
{
    const size_t conv_len = hint::int_ceil2(len);
    size_t result = conv_len;
    int shift = 2;
    for (size_t i = 3; i <= factor_range; i += 2)
    {
        if ((size_t(1) << shift) <= i)
        {
            shift++;
        }
        size_t try_len = (conv_len >> shift) * i;
        if (try_len >= len && try_len < result)
        {
            result = try_len;
        }
    }
    return result;
}
void test_conv()
{
    // using NTT = hint::transform::ntt::NTT<hint::transform::ntt::MOD2, hint::transform::ntt::ROOT2>;
    using NTT = hint::transform::ntt::NTT<998244353, 3>;

    using Mint = NTT::IntType;
    std::vector<Mint> a, b;
    // static Mint a[1 << 23], b[1 << 23];
    int log_len;
    std::cin >> log_len;
    size_t len = 1 << log_len;
    // len *= 9;
    // len *= 8;
    // std::cin >> len;
    size_t conv_len = NTT::findFitLen(len);
    a.resize(conv_len);
    b.resize(conv_len);
    for (int i = 0; i < len / 2; i++)
    {
        a[i] = 2;
        b[i] = 5;
    }
    // len = hint::int_ceil2(len);
    // size_t conv_len = hint::int_ceil2(len);
    std::cout << conv_len << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    // conv_cyclic(a, b, len);
    // for (int i = 0; i < 1000; i++)
    NTT::convolution(&a[0], &b[0], conv_len);
    auto t2 = std::chrono::steady_clock::now();
    // conv_short(a, b, 16, 1);
    for (size_t i = 0; i < len; i++)
    {
        // std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}

#include <type_traits>
void test_copy()
{
    using Mint = hint::transform::ntt::MontIntLazy<998244353>;
    std::vector<Mint> a(1 << 23);
    std::vector<uint32_t> b(1 << 23);

    size_t len = 1 << 5;
    size_t shift = rand();
    auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 1000; i++)
    {
        std::copy(a.begin(), a.begin() + len, a.begin() + shift);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < 1000; i++)
    {
        std::copy(b.begin(), b.begin() + len, b.begin() + shift);
    }
    auto t3 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us" << std::endl;
}

void test_modint()
{
    constexpr uint64_t mod = 998244353;
    using Mint = hint::transform::ntt::MontIntLazy<mod>;
    Mint a = 0;
    for (size_t i = 0; i < mod - 1; i++)
    {
        a = a.add(Mint(1));
        a = a.norm2();
    }
    for (size_t i = 0; i < mod - 1; i++)
    {
        a = a.sub(Mint(1));
        a = a.norm2();
    }
    std::cout << a << std::endl;
}

template <typename T>
std::vector<T> poly_multiply(const std::vector<T> &in1, const std::vector<T> &in2)
{
    using namespace hint::transform::ntt;
    using NTT = hint::transform::ntt::NTT<998244353, 3>;
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    std::vector<T> result(out_len);
    size_t ntt_len = NTT::findFitLen(out_len);
    std::vector<NTT::IntType> buffer1(ntt_len), buffer2(ntt_len);
    std::copy(in1.begin(), in1.end(), buffer1.begin());
    std::copy(in2.begin(), in2.end(), buffer2.begin());
    NTT::convolution(buffer1.data(), buffer2.data(), ntt_len);
    std::copy(buffer1.begin(), buffer1.begin() + out_len, result.begin());
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
    // std::cin >> n;
    size_t len = size_t(1) << n; // 变换长度
    len = 117440512;
    uint64_t ele = 2;
    std::vector<uint32_t> in1(len / 2, ele);
    std::vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    auto t1 = std::chrono::steady_clock::now();
    std::vector<uint32_t> res = poly_multiply(in1, in2);
    auto t2 = std::chrono::steady_clock::now();
    result_test(res, ele); // 结果校验
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}

#include "bind_cpu.hpp"

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    // test_copy();
    bind_cpu(0);
    test_conv();
    // test_ntt();
    // test_modint();
    // ntt_perf();
    // test_mul(); // 测试乘法
    // test_mul64();
    // test_ntt(); // 测试卷积
    // test_rev_modint();
    // conv(nullptr, nullptr, 16);
    // std::cout << table.getNext(1, 1) << std::endl;
    // test_poly();
    // std::cout << hint::transform::ntt::add_count << "\t" << hint::transform::ntt::mul_count << "\n";
    // ntt_perf_loop();
}