#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <climits>
#include <string>
#include <array>
#include <type_traits>

// Windows 64位快速乘法宏
#if defined(_WIN64)
#include <intrin.h>
#define UMUL128
#endif //_WIN64

// Unix 64位快速乘法宏
#if defined(__unix__) && defined(__x86_64__) && defined(__GNUC__)
#define UINT128T
#endif //__unix__

namespace hint
{
    using Float32 = float;
    using Float64 = double;
    using Complex32 = std::complex<Float32>;
    using Complex64 = std::complex<Float64>;

    constexpr Float64 HINT_PI = 3.141592653589793238462643;
    constexpr Float64 HINT_2PI = HINT_PI * 2;
    std::string ui64to_string(uint64_t input, uint8_t digits)
    {
        std::string result(digits, '0');
        for (uint8_t i = 0; i < digits; i++)
        {
            result[digits - i - 1] = static_cast<char>(input % 10 + '0');
            input /= 10;
        }
        return result;
    }
    // 模板快速幂
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

    // 模板快速幂
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

    // bits个二进制全为1的数,等于2^bits-1
    template <typename T>
    constexpr T all_one(int bits)
    {
        T temp = T(1) << (bits - 1);
        return temp - 1 + temp;
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

    // 整数log2
    template <typename UintTy>
    constexpr int hint_log2(UintTy n)
    {
        constexpr int bits = 8 * sizeof(UintTy);
        constexpr UintTy mask = all_one<UintTy>(bits / 2) << (bits / 2);
        UintTy m = mask;
        int res = 0, shift = bits / 2;
        while (shift > 0)
        {
            if ((n & m))
            {
                res += shift;
                n >>= shift;
            }
            shift /= 2;
            m >>= shift;
        }
        return res;
    }

    template <typename IntTy>
    constexpr int hint_clz(IntTy x)
    {
        return sizeof(IntTy) * CHAR_BIT - 1 - hint_log2(x);
    }
    template <typename IntTy>
    constexpr int hint_ctz(IntTy x)
    {
        return hint_log2(x ^ (x - 1));
    }
    namespace hint_transform
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
        namespace hint_ntt
        {
            constexpr uint64_t MOD1 = 1945555039024054273, ROOT1 = 5;
            constexpr uint64_t MOD2 = 4179340454199820289, ROOT2 = 3;
            constexpr uint64_t MOD3 = 754974721, ROOT3 = 11;
            constexpr uint64_t MOD4 = 469762049, ROOT4 = 3;
            constexpr uint64_t MOD5 = 3489660929, ROOT5 = 3;
            constexpr uint64_t MOD6 = 3221225473, ROOT6 = 5;

            // Compute Integer multiplication, 64bit x 64bit to 128bit, basic algorithm
            // first is low 64bit, second is high 64bit
            constexpr std::pair<uint64_t, uint64_t> mul64x64to128_base(uint64_t a, uint64_t b)
            {
                uint64_t ah = a >> 32, bh = b >> 32;
                a = uint32_t(a), b = uint32_t(b);
                uint64_t r0 = a * b, r1 = a * bh, r2 = ah * b, r3 = ah * bh;
                r3 += (r1 >> 32) + (r2 >> 32);
                r1 = uint32_t(r1), r2 = uint32_t(r2);
                r1 += r2;
                r1 += (r0 >> 32);
                r3 += (r1 >> 32);
                r0 = (r1 << 32) | uint32_t(r0);
                return std::make_pair(r0, r3);
            }

            inline std::pair<uint64_t, uint64_t> mul64x64to128(uint64_t a, uint64_t b)
            {
#ifdef UMUL128
#pragma message("Using _umul128 to compute 64bit x 64bit to 128bit")
                unsigned long long lo, hi;
                lo = _umul128(a, b, &hi);
                return std::make_pair(lo, hi);
#else // No UMUL128
#ifdef UINT128T
#pragma message("Using __uint128_t to compute 64bit x 64bit to 128bit")
                __uint128_t x(a);
                x *= b;
                return std::make_pair(uint64_t(x), uint64_t(x >> 64));
#else // No __uint128_t
#pragma message("Using basic function to compute 64bit x 64bit to 128bit")
                return mul64x64to128_base(a, b);
#endif // UINT128T
#endif // UMUL128
            }

            constexpr std::pair<uint64_t, uint64_t> div128by32(uint64_t dividend_hi64, uint64_t &dividend_lo64, uint32_t divisor)
            {
                uint64_t quot_hi64 = 0, quot_lo64 = 0;
                uint64_t q = 0, r = dividend_hi64 >> 32;
                q = r / divisor;
                r = r % divisor;
                quot_hi64 = q << 32;

                r = (r << 32) | uint32_t(dividend_hi64);
                q = r / divisor;
                r = r % divisor;
                quot_hi64 |= q;

                r = (r << 32) | (dividend_lo64 >> 32);
                q = r / divisor;
                r = r % divisor;
                quot_lo64 = q << 32;

                r = (r << 32) | uint32_t(dividend_lo64);
                q = r / divisor;
                r = r % divisor;
                quot_lo64 |= q;
                dividend_lo64 = r;
                return std::make_pair(quot_lo64, quot_hi64);
            }

            // 整数96位除以64位，输入数据需要保证商不大于32位
            constexpr uint32_t div96by64to32(uint32_t dividend_hi64, uint64_t &dividend_lo64, uint64_t divisor)
            {
                uint64_t divid2 = (uint64_t(dividend_hi64) << 32) | (dividend_lo64 >> 32);
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
                    // divid2 > divis1是判断上一次加法后是否出现溢出，若溢出，prod不可能大于divid2
                    if ((divid2 > divis1) && (prod > divid2))
                    {
                        qhat--;
                        prod -= divisor;
                        divid2 += divis1;
                    }
                }
                divid2 -= prod;
                dividend_lo64 = divid2;
                return qhat;
            }
            // 整数128位除以64位，输入数据需要保证商不大于64位
            constexpr uint64_t div128by64to64(uint64_t dividend_hi64, uint64_t &dividend_lo64, uint64_t divisor)
            {
                if ((divisor >> 32) == 0)
                {
                    return div128by32(dividend_hi64, dividend_lo64, uint32_t(divisor)).first;
                }
                uint32_t q1 = 0, q0 = 0;
                uint32_t divid_hi32 = dividend_hi64 >> 32;
                uint64_t divid_lo64 = (dividend_hi64 << 32) | (dividend_lo64 >> 32);
                if (divid_hi32 != 0)
                {
                    q1 = div96by64to32(divid_hi32, divid_lo64, divisor);
                }
                divid_hi32 = divid_lo64 >> 32;
                dividend_lo64 = uint32_t(dividend_lo64) | (divid_lo64 << 32);
                q0 = div96by64to32(divid_hi32, dividend_lo64, divisor);
                return (uint64_t(q1) << 32) | q0;
            }

            class Uint128
            {
            private:
                uint64_t lo, hi;

            public:
                constexpr Uint128() : Uint128(0, 0) {}
                constexpr Uint128(uint64_t l) : Uint128(l, 0) {}
                constexpr Uint128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
                constexpr Uint128(std::pair<uint64_t, uint64_t> p) : lo(p.first), hi(p.second) {}

                constexpr Uint128 operator+(Uint128 rhs) const
                {
                    rhs.lo += lo;
                    rhs.hi += hi + (rhs.lo < lo);
                    return rhs;
                }
                constexpr Uint128 operator-(Uint128 rhs) const
                {
                    rhs.lo = lo - rhs.lo;
                    rhs.hi = hi - rhs.hi - (rhs.lo > lo);
                    return rhs;
                }
                constexpr Uint128 operator+(uint64_t rhs) const
                {
                    rhs = lo + rhs;
                    return Uint128(rhs, hi + (rhs < lo));
                }
                constexpr Uint128 operator-(uint64_t rhs) const
                {
                    rhs = lo - rhs;
                    return Uint128(rhs, hi - (rhs > lo));
                }
                // Only compute the lo * rhs.lo
                Uint128 operator*(const Uint128 &rhs) const
                {
                    return mul64x64to128(lo, rhs.lo);
                }
                // Only compute the 128bit / 64 bit
                constexpr Uint128 operator/(const Uint128 &rhs) const
                {
                    uint64_t divisor = rhs.lo;
                    if ((divisor >> 32) == 0)
                    {
                        uint64_t rem = lo;
                        return div128by32(hi, rem, divisor);
                    }
                    int k = hint_clz(divisor);
                    divisor <<= k;
                    Uint128 divid = operator<<(k);
                    return div128by64to64(divid.hi, divid.lo, divisor);
                }
                // Only compute the 128bit % 64 bit
                constexpr Uint128 operator%(const Uint128 &rhs) const
                {
                    uint64_t divisor = rhs.lo;
                    if ((divisor >> 32) == 0)
                    {
                        uint64_t rem = lo;
                        div128by32(hi, rem, divisor);
                        return Uint128(rem);
                    }
                    const int k = hint_clz(divisor);
                    divisor <<= k;
                    Uint128 divid = *this << k;
                    div128by64to64(divid.hi, divid.lo, divisor);
                    return Uint128(divid.lo) >> k;
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
                // Only compute the lo * rhs.lo
                constexpr Uint128 &operator*=(const Uint128 &rhs)
                {
                    return *this = mul64x64to128_base(lo, rhs.lo);
                }
                constexpr Uint128 &operator/=(const Uint128 &rhs)
                {
                    return *this = *this / rhs;
                }
                constexpr Uint128 &operator%=(const Uint128 &rhs)
                {
                    return *this = *this % rhs;
                }
                constexpr bool operator>(const Uint128 &rhs) const
                {
                    if (hi != rhs.hi)
                    {
                        return hi > rhs.hi;
                    }
                    return lo > rhs.lo;
                }
                constexpr bool operator<(const Uint128 &rhs) const
                {
                    if (hi != rhs.hi)
                    {
                        return hi < rhs.hi;
                    }
                    return lo < rhs.lo;
                }
                constexpr bool operator>=(const Uint128 &rhs) const
                {
                    return !(*this < rhs);
                }
                constexpr bool operator<=(const Uint128 &rhs) const
                {
                    return !(*this > rhs);
                }
                constexpr bool operator==(const Uint128 &rhs) const
                {
                    return hi == rhs.hi && lo == rhs.lo;
                }
                constexpr bool operator!=(const Uint128 &rhs) const
                {
                    return !(*this == rhs);
                }
                constexpr Uint128 operator<<(int shift) const
                {
                    if (shift == 0)
                    {
                        return *this;
                    }
                    shift %= 128;
                    shift = shift < 0 ? shift + 128 : shift;
                    if (shift < 64)
                    {
                        return Uint128(lo << shift, (hi << shift) | (lo >> (64 - shift)));
                    }
                    return Uint128(0, lo << (shift - 64));
                }
                constexpr Uint128 operator>>(int shift) const
                {
                    if (shift == 0)
                    {
                        return *this;
                    }
                    shift %= 128;
                    shift = shift < 0 ? shift + 128 : shift;
                    if (shift < 64)
                    {
                        return Uint128((lo >> shift) | (hi << (64 - shift)), hi >> shift);
                    }
                    return Uint128(hi >> (shift - 64), 0);
                }
                constexpr Uint128 &operator<<=(int shift)
                {
                    return *this = *this << shift;
                }
                constexpr Uint128 &operator>>=(int shift)
                {
                    return *this = *this >> shift;
                }
                constexpr uint64_t high64() const
                {
                    return hi;
                }
                constexpr uint64_t low64() const
                {
                    return lo;
                }
                constexpr operator uint64_t() const
                {
                    return low64();
                }
                void printDec() const
                {
                    if (hi == 0)
                    {
                        std::cout << std::dec << lo << "\n";
                        return;
                    }
                    constexpr Uint128 BASE(1e16);
                    Uint128 copy(*this);
                    std::string s;
                    s = ui64to_string(uint64_t(copy % BASE), 16) + s;
                    copy /= BASE;
                    s = ui64to_string(uint64_t(copy % BASE), 16) + s;
                    copy /= BASE;
                    std::cout << std::to_string(uint64_t(copy % BASE)) + s << "\n";
                }
                void printHex() const
                {
                    std::cout << std::hex << "0x" << hi << " 0x" << lo << "\n";
                }
            };

            // Montgomery for mod > 2^32
            // default R = 2^64
            template <typename Int128Type = Uint128>
            class Montgomery64
            {
            public:
                uint64_t mod = 0;         // modulus, must be odd and < 2^64
                uint64_t mod_inv = 0;     // mod^-1
                uint64_t mod_inv_neg = 0; //-mod^-1
                uint64_t mod2 = 0;        // mod*2
                uint64_t r2 = 0;          // r*r%mod

            public:
                constexpr Montgomery64(uint64_t mod_in) : mod(mod_in), mod2(mod_in * 2)
                {
                    mod_inv = inv_mod2pow(mod, 64);      //(mod_inv * mod)%(2^64) = 1
                    mod_inv_neg = uint64_t(0 - mod_inv); //(mod_inv_neg + mod_inv)%(2^64) = 0
                    Int128Type R = (Int128Type(1) << 64) % Int128Type(mod);
                    R *= R;
                    r2 = uint64_t(R % Int128Type(mod));
                }
                uint64_t redcFastLazy(const Int128Type &input) const
                {
                    Int128Type n = uint64_t(input) * mod_inv_neg;
                    n = n * Int128Type(mod);
                    n += input;
                    return n >> 64;
                }
                uint64_t redcFast(const Int128Type &input) const
                {
                    uint64_t n = redcFastLazy(input);
                    return n < mod ? n : n - mod;
                }
                constexpr uint64_t redc(const Int128Type &input) const
                {
                    Int128Type n = uint64_t(input) * mod_inv_neg;
                    n *= Int128Type(mod);
                    n += input;
                    uint64_t m = n >> 64;
                    return m < mod ? m : m - mod;
                }

                uint64_t mulMontRunTime(uint64_t a, uint64_t b) const
                {
                    return redcFast(Int128Type(a) * Int128Type(b));
                }
                uint64_t mulMontRunTimeLazy(uint64_t a, uint64_t b) const
                {
                    return redcFastLazy(Int128Type(a) * Int128Type(b));
                }
                constexpr uint64_t mulMontCompileTime(uint64_t a, uint64_t b) const
                {
                    Int128Type prod(a);
                    prod *= Int128Type(b);
                    return redc(prod);
                }
                constexpr uint64_t addMont(uint64_t a, uint64_t b) const
                {
                    b = a + b;
                    return b < mod ? b : b - mod;
                }
                constexpr uint64_t addMontLazy(uint64_t a, uint64_t b) const
                {
                    b = a + b;
                    return b < mod2 ? b : b - mod2;
                }
                constexpr uint64_t subMont(uint64_t a, uint64_t b) const
                {
                    b = a - b;
                    return b > a ? b + mod : b;
                }
                constexpr uint64_t subMontLazy(uint64_t a, uint64_t b) const
                {
                    b = a - b;
                    return b > a ? b + mod2 : b;
                }
                constexpr uint64_t largeNorm2(uint64_t n) const
                {
                    return n >= mod2 ? n - mod2 : n;
                }
                constexpr uint64_t toMont(uint64_t a) const
                {
                    return mulMontCompileTime(a, r2);
                }
                constexpr uint64_t toInt(uint64_t a) const
                {
                    return redc(Int128Type(a));
                }
                constexpr bool selfCheck() const
                {
                    return uint64_t((mod_inv * mod) == 1) && (mod_inv_neg + mod_inv == 0);
                }
            };

            template <uint64_t MOD, typename Int128Type = Uint128>
            class MontInt64Lazy
            {
            private:
                static_assert(MOD > UINT32_MAX, "Montgomery64 modulus must be greater than 2^32");
                static_assert(hint_log2(MOD) < 62, "MOD can't be larger than 62 bits");
                uint64_t data;

            public:
                using IntType = uint64_t;
                using MontgomeryType = Montgomery64<Int128Type>;
                static constexpr MontgomeryType montgomery = MontgomeryType(MOD);
                static_assert(montgomery.selfCheck(), "Montgomery64 modulus is not correct");

                constexpr MontInt64Lazy() : data(0) {}
                constexpr MontInt64Lazy(uint64_t n) : data(montgomery.toMont(n)) {}

                constexpr MontInt64Lazy operator+(MontInt64Lazy rhs) const
                {
                    rhs.data = montgomery.addMontLazy(data, rhs.data);
                    return rhs;
                }
                constexpr MontInt64Lazy operator-(MontInt64Lazy rhs) const
                {
                    rhs.data = montgomery.subMontLazy(data, rhs.data);
                    return rhs;
                }
                MontInt64Lazy operator*(MontInt64Lazy rhs) const
                {
                    rhs.data = montgomery.mulMontRunTimeLazy(data, rhs.data);
                    return rhs;
                }
                constexpr MontInt64Lazy &operator+=(const MontInt64Lazy &rhs)
                {
                    data = montgomery.addMontLazy(data, rhs.data);
                    return *this;
                }
                constexpr MontInt64Lazy &operator-=(const MontInt64Lazy &rhs)
                {
                    data = montgomery.subMontLazy(data, rhs.data);
                    return *this;
                }
                constexpr MontInt64Lazy &operator*=(const MontInt64Lazy &rhs)
                {
                    data = montgomery.mulMontCompileTime(data, rhs.data);
                    return *this;
                }
                constexpr MontInt64Lazy largeNorm2() const
                {
                    MontInt64Lazy res;
                    res.data = montgomery.largeNorm2(data);
                    return res;
                }
                constexpr MontInt64Lazy rawAdd(MontInt64Lazy n) const
                {
                    n.data = data + n.data;
                    return n;
                }
                constexpr MontInt64Lazy rawSub(MontInt64Lazy n) const
                {
                    n.data = data - n.data + mod2();
                    return n;
                }
                constexpr uint64_t montToInt() const
                {
                    return montgomery.toInt(data);
                }
                constexpr operator uint64_t() const
                {
                    return montToInt();
                }
                static constexpr uint64_t mod()
                {
                    return MOD;
                }
                static constexpr uint64_t mod2()
                {
                    return MOD * 2;
                }
            };
            template <uint64_t MOD, typename Int128Type>
            constexpr typename MontInt64Lazy<MOD, Int128Type>::MontgomeryType MontInt64Lazy<MOD, Int128Type>::montgomery;

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
                    constexpr uint32_t mod_neg_inv = uint32_t(-modInv());
                    return mod_neg_inv;
                }
                static_assert((mod() * modInv()) == 1, "mod_inv not correct");

                static constexpr uint32_t toMont(uint32_t n)
                {
                    return (uint64_t(n) << 32) % MOD;
                }
                static constexpr uint32_t redcLazy(uint64_t n)
                {
                    uint64_t prod = uint32_t(n) * modNegInv();
                    return (prod * mod() + n) >> 32;
                }
                static constexpr uint32_t redc(uint64_t n)
                {
                    n = redcLazy(n);
                    return n < mod() ? n : n - mod();
                }
                constexpr MontInt32Lazy operator+(MontInt32Lazy rhs) const
                {
                    rhs.data = data + rhs.data;
                    rhs.data = rhs.data < mod2() ? rhs.data : rhs.data - mod2();
                    return rhs;
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
                    rhs.data = data + rhs.data;
                    data = rhs.data < mod2() ? rhs.data : rhs.data - mod2();
                    return *this;
                }
                constexpr MontInt32Lazy &operator-=(const MontInt32Lazy &rhs)
                {
                    rhs.data = data - rhs.data;
                    data = rhs.data > data ? rhs.data + mod2() : rhs.data;
                    return *this;
                }
                constexpr MontInt32Lazy &operator*=(const MontInt32Lazy &rhs)
                {
                    data = redc(uint64_t(data) * rhs.data);
                    return *this;
                }
                constexpr MontInt32Lazy largeNorm2() const
                {
                    MontInt32Lazy res;
                    res.data = data >= mod2() ? data - mod2() : data;
                    return res;
                }
                constexpr MontInt32Lazy rawAdd(MontInt32Lazy n) const
                {
                    n.data = data + n.data;
                    return n;
                }
                constexpr MontInt32Lazy rawSub(MontInt32Lazy n) const
                {
                    n.data = data - n.data + mod2();
                    return n;
                }
                constexpr uint32_t montToInt() const
                {
                    return redc(data);
                }
                constexpr operator uint32_t() const
                {
                    return montToInt();
                }
            };

            template <uint32_t MOD>
            class ModInt32
            {
            private:
                uint32_t data;

            public:
                using IntType = uint32_t;

                constexpr ModInt32() {}
                constexpr ModInt32(uint32_t in) : data(in) {}

                constexpr ModInt32 largeNorm2() const
                {
                    return data;
                }
                constexpr uint64_t mul64(ModInt32 in) const
                {
                    return uint64_t(data) * in.data;
                }
                constexpr ModInt32 operator+(ModInt32 in) const
                {
                    uint32_t diff = MOD - data;
                    return in.data > diff ? in.data - diff : in.data + data;
                }
                constexpr ModInt32 operator-(ModInt32 in) const
                {
                    in.data = data - in.data;
                    return in.data > data ? in.data + MOD : in.data;
                }
                constexpr ModInt32 operator*(ModInt32 in) const
                {
                    return mul64(in) % MOD;
                }
                constexpr ModInt32 &operator+=(ModInt32 in)
                {
                    return *this = *this + in;
                }
                constexpr ModInt32 &operator-=(ModInt32 in)
                {
                    return *this = *this - in;
                }
                constexpr ModInt32 &operator*=(ModInt32 in)
                {
                    return *this = *this * in;
                }
                constexpr operator uint32_t() const
                {
                    return data;
                }
                static constexpr uint32_t mod()
                {
                    return MOD;
                }
                constexpr ModInt32 rawAdd(ModInt32 n) const
                {
                    return *this + n;
                }
                constexpr ModInt32 rawSub(ModInt32 n) const
                {
                    return *this - n;
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
            template <uint64_t MOD1, uint64_t MOD2, typename Int128Type = Uint128>
            inline Int128Type qcrt(uint64_t num1, uint64_t num2)
            {
                constexpr uint64_t inv1 = mod_inv<int64_t>(MOD1, MOD2);
                constexpr uint64_t inv2 = mod_inv<int64_t>(MOD2, MOD1);
                static_assert(check_inv<Int128Type>(inv1, MOD1, MOD2), "Inv1 error");
                static_assert(check_inv<Int128Type>(inv2, MOD2, MOD1), "Inv2 error");
                if (num1 > num2)
                {
                    return (Int128Type(num1 - num2) * Int128Type(inv2) % Int128Type(MOD1)) * Int128Type(MOD2) + num2;
                }
                else
                {
                    return (Int128Type(num2 - num1) * Int128Type(inv1) % Int128Type(MOD2)) * Int128Type(MOD1) + num1;
                }
            }
            // 快速计算两模数的中国剩余定理
            template <uint32_t MOD1, uint32_t MOD2>
            inline uint64_t qcrt(uint32_t num1, uint32_t num2)
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
            namespace split_radix
            {
                template <uint64_t ROOT, typename ModIntType>
                inline ModIntType mul_w41(ModIntType n)
                {
                    constexpr ModIntType W_4_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 4);
                    return n * W_4_1;
                }
                template <uint64_t ROOT, typename ModIntType>
                inline ModIntType mul_w81(ModIntType n)
                {
                    constexpr ModIntType W_8_1 = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / 8);
                    return n * W_8_1;
                }
                template <uint64_t ROOT, typename ModIntType>
                inline ModIntType mul_w83(ModIntType n)
                {
                    constexpr ModIntType W_8_3 = qpow(ModIntType(ROOT), ((ModIntType::mod() - 1) / 8) * 3);
                    return n * W_8_3;
                }

                // in: in_out0<4p, in_ou1<4p; in_out2<2p, in_ou3<2p
                // out: in_out0<4p, in_ou1<4p; in_out2<4p, in_ou3<4p
                template <uint64_t ROOT, typename ModIntType>
                inline void dit_butterfly244(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3)
                {
                    ModIntType temp0, temp1, temp2, temp3;
                    temp0 = in_out0.largeNorm2();
                    temp1 = in_out1.largeNorm2();
                    temp2 = in_out2 + in_out3;
                    temp3 = in_out2.rawSub(in_out3);
                    temp3 = mul_w41<ROOT>(temp3);
                    in_out0 = temp0.rawAdd(temp2);
                    in_out2 = temp0.rawSub(temp2);
                    in_out1 = temp1.rawAdd(temp3);
                    in_out3 = temp1.rawSub(temp3);
                }
                // in: in_out0<2p, in_ou1<2p; in_out2<2p, in_ou3<2p
                // out: in_out0<2p, in_ou1<2p; in_out2<4p, in_ou3<4p
                template <uint64_t ROOT, typename ModIntType>
                inline void dif_butterfly244(ModIntType &in_out0, ModIntType &in_out1, ModIntType &in_out2, ModIntType &in_out3)
                {
                    ModIntType temp0, temp1, temp2, temp3;
                    temp0 = in_out0.rawAdd(in_out2);
                    temp2 = in_out0 - in_out2;
                    temp1 = in_out1.rawAdd(in_out3);
                    temp3 = in_out1.rawSub(in_out3);
                    temp3 = mul_w41<ROOT>(temp3);
                    in_out0 = temp0.largeNorm2();
                    in_out1 = temp1.largeNorm2();
                    in_out2 = temp2.rawAdd(temp3);
                    in_out3 = temp2.rawSub(temp3);
                }
                // in: in_out0<4p, in_ou1<4p
                // out: in_out0<4p, in_ou1<4p
                template <typename ModIntType>
                inline void dit_butterfly2(ModIntType &in_out0, ModIntType &in_out1, const ModIntType &omega)
                {
                    auto x = in_out0.largeNorm2();
                    auto y = in_out1 * omega;
                    in_out0 = x.rawAdd(y);
                    in_out1 = x.rawSub(y);
                }
                // in: in_out0<2p, in_ou1<2p
                // out: in_out0<2p, in_ou1<2p
                template <typename ModIntType>
                inline void dif_butterfly2(ModIntType &in_out0, ModIntType &in_out1, const ModIntType &omega)
                {
                    auto x = in_out0 + in_out1;
                    auto y = in_out0.rawSub(in_out1);
                    in_out0 = x;
                    in_out1 = y * omega;
                }
                template <size_t LEN, uint64_t ROOT, typename ModIntType>
                struct NTTShort
                {
                    static constexpr size_t ntt_len = LEN;
                    static constexpr int log_len = hint_log2(ntt_len);
                    using TableType = std::array<ModIntType, ntt_len>;

                    static constexpr ModIntType *getOmegaIt(size_t len)
                    {
                        return &table[len / 2];
                    }
                    static constexpr TableType getNTTTable()
                    {
                        for (int omega_log_len = 0; omega_log_len <= log_len; omega_log_len++)
                        {
                            size_t omega_len = size_t(1) << omega_log_len, omega_count = omega_len / 2;
                            auto it = getOmegaIt(omega_len);
                            ModIntType root = qpow(ModIntType(ROOT), (ModIntType::mod() - 1) / omega_len);
                            ModIntType omega(1);
                            for (size_t i = 0; i < omega_count; i++)
                            {
                                it[i] = omega;
                                omega *= root;
                            }
                        }
                        return table;
                    }

                    static TableType table;

                    static void dit(ModIntType in_out[], size_t len)
                    {
                        size_t rank = len;
                        if (hint_log2(len) % 2 == 0)
                        {
                            NTTShort<4, ROOT, ModIntType>::dit(in_out, len);
                            for (size_t i = 4; i < len; i += 4)
                            {
                                NTTShort<4, ROOT, ModIntType>::dit(in_out + i);
                            }
                            rank = 16;
                        }
                        else
                        {
                            NTTShort<8, ROOT, ModIntType>::dit(in_out, len);
                            for (size_t i = 8; i < len; i += 8)
                            {
                                NTTShort<8, ROOT, ModIntType>::dit(in_out + i);
                            }
                            rank = 32;
                        }
                        for (; rank <= len; rank *= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = getOmegaIt(rank), last_omega_it = getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i++)
                                {
                                    auto temp0 = it0[j + i], temp1 = it1[j + i], temp2 = it2[j + i], temp3 = it3[j + i], omega = last_omega_it[i];
                                    dit_butterfly2(temp0, temp1, omega);
                                    dit_butterfly2(temp2, temp3, omega);
                                    dit_butterfly2(temp0, temp2, omega_it[i]);
                                    dit_butterfly2(temp1, temp3, omega_it[gap + i]);
                                    it0[j + i] = temp0, it1[j + i] = temp1, it2[j + i] = temp2, it3[j + i] = temp3;
                                }
                            }
                        }
                    }
                    static void dif(ModIntType in_out[], size_t len)
                    {
                        size_t rank = len;
                        for (; rank >= 16; rank /= 4)
                        {
                            size_t gap = rank / 4;
                            auto omega_it = getOmegaIt(rank), last_omega_it = getOmegaIt(rank / 2);
                            auto it0 = in_out, it1 = in_out + gap, it2 = in_out + gap * 2, it3 = in_out + gap * 3;
                            for (size_t j = 0; j < len; j += rank)
                            {
                                for (size_t i = 0; i < gap; i++)
                                {
                                    auto temp0 = it0[j + i], temp1 = it1[j + i], temp2 = it2[j + i], temp3 = it3[j + i], omega = last_omega_it[i];
                                    dif_butterfly2(temp0, temp2, omega_it[i]);
                                    dif_butterfly2(temp1, temp3, omega_it[gap + i]);
                                    dif_butterfly2(temp0, temp1, omega);
                                    dif_butterfly2(temp2, temp3, omega);
                                    it0[j + i] = temp0, it1[j + i] = temp1, it2[j + i] = temp2, it3[j + i] = temp3;
                                }
                            }
                        }
                        if (hint_log2(rank) % 2 == 0)
                        {
                            NTTShort<4, ROOT, ModIntType>::dif(in_out, len);
                            for (size_t i = 4; i < len; i += 4)
                            {
                                NTTShort<4, ROOT, ModIntType>::dif(in_out + i);
                            }
                        }
                        else
                        {
                            NTTShort<8, ROOT, ModIntType>::dif(in_out, len);
                            for (size_t i = 8; i < len; i += 8)
                            {
                                NTTShort<8, ROOT, ModIntType>::dif(in_out + i);
                            }
                        }
                    }
                };
                template <size_t LEN, uint64_t ROOT, typename ModIntType>
                typename NTTShort<LEN, ROOT, ModIntType>::TableType NTTShort<LEN, ROOT, ModIntType>::table = NTTShort<LEN, ROOT, ModIntType>::getNTTTable();

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<0, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<1, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[]) {}
                    static void dif(ModIntType in_out[]) {}
                    static void dit(ModIntType in_out[], size_t len) {}
                    static void dif(ModIntType in_out[], size_t len) {}
                };

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<2, ROOT, ModIntType>
                {
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

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<4, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        temp3 = mul_w41<ROOT>(temp3);

                        in_out[0] = temp0 + temp2;
                        in_out[1] = temp1 + temp3;
                        in_out[2] = temp0 - temp2;
                        in_out[3] = temp1 - temp3;
                    }
                    static void dif(ModIntType in_out[])
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        temp3 = mul_w41<ROOT>(temp3);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
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

                template <uint64_t ROOT, typename ModIntType>
                struct NTTShort<8, ROOT, ModIntType>
                {
                    static void dit(ModIntType in_out[])
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

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        transform2(temp4, temp5);
                        transform2(temp6, temp7);
                        temp3 = mul_w41<ROOT>(temp3);
                        temp7 = mul_w41<ROOT>(temp7);

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp5, temp7);
                        temp5 = temp5 * w1;
                        temp6 = temp6 * w2;
                        temp7 = temp7 * w3;

                        in_out[0] = temp0 + temp4;
                        in_out[1] = temp1 + temp5;
                        in_out[2] = temp2 + temp6;
                        in_out[3] = temp3 + temp7;
                        in_out[4] = temp0 - temp4;
                        in_out[5] = temp1 - temp5;
                        in_out[6] = temp2 - temp6;
                        in_out[7] = temp3 - temp7;
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
                        transform2(temp1, temp5);
                        transform2(temp2, temp6);
                        transform2(temp3, temp7);
                        temp5 = temp5 * w1;
                        temp6 = temp6 * w2;
                        temp7 = temp7 * w3;

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp5, temp7);
                        temp3 = mul_w41<ROOT>(temp3);
                        temp7 = mul_w41<ROOT>(temp7);

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
                };

                template <uint64_t MOD, uint64_t ROOT, typename Int128Type = Uint128>
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
                    static constexpr uint64_t iroot()
                    {
                        return mod_inv<int64_t>(root(), mod());
                    }
                    static constexpr bool selfCheck()
                    {
                        Int128Type n = root();
                        n *= Int128Type(iroot());
                        n %= Int128Type(mod());
                        return n == Int128Type(1);
                    }
                    static_assert(root() < mod(), "ROOT must be smaller than MOD");
                    static_assert(selfCheck(), "IROOT * ROOT % MOD must be 1");
                    static constexpr int mod_bits = hint_log2(mod()) + 1;
                    static constexpr int max_log_len = hint_ctz(mod() - 1);

                    static constexpr size_t getMaxLen()
                    {
                        if (max_log_len < sizeof(size_t) * CHAR_BIT)
                        {
                            return size_t(1) << max_log_len;
                        }
                        return size_t(1) << (sizeof(size_t) * CHAR_BIT - 1);
                    }
                    static constexpr size_t ntt_max_len = getMaxLen();

                    using INTT = NTT<mod(), iroot(), Int128Type>;
                    using ModInt32Type = typename std::conditional<(mod_bits > 30), ModInt32<uint32_t(MOD)>, MontInt32Lazy<uint32_t(MOD)>>::type;
                    using ModInt64Type = MontInt64Lazy<MOD, Int128Type>;
                    using ModIntType = typename std::conditional<(mod_bits > 32), ModInt64Type, ModInt32Type>::type;
                    using IntType = typename ModIntType::IntType;

                    static constexpr size_t L2_BYTE = size_t(1) << 20; // 1MB L2 cache size, change this if you know your cache size.
                    static constexpr size_t LONG_THRESHOLD = L2_BYTE / sizeof(ModIntType);
                    using NTTTemplate = NTTShort<LONG_THRESHOLD, root(), ModIntType>;

                    static void dit244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), ntt_max_len);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dit(in_out, ntt_len);
                            return;
                        }
                        size_t quarter_len = ntt_len / 4;
                        dit244(in_out + quarter_len * 3, ntt_len / 4);
                        dit244(in_out + quarter_len * 2, ntt_len / 4);
                        dit244(in_out, ntt_len / 2);
                        const ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        const ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i++)
                        {
                            ModIntType temp0 = it0[i], temp1 = it1[i], temp2 = it2[i] * omega1, temp3 = it3[i] * omega3;
                            dit_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            it0[i] = temp0, it1[i] = temp1, it2[i] = temp2, it3[i] = temp3;
                            omega1 = omega1 * unit_omega1;
                            omega3 = omega3 * unit_omega3;
                        }
                    }
                    static void dif244(ModIntType in_out[], size_t ntt_len)
                    {
                        ntt_len = std::min(int_floor2(ntt_len), ntt_max_len);
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            NTTTemplate::dif(in_out, ntt_len);
                            return;
                        }
                        size_t quarter_len = ntt_len / 4;
                        const ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        const ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        auto it0 = in_out, it1 = in_out + quarter_len, it2 = in_out + quarter_len * 2, it3 = in_out + quarter_len * 3;
                        for (size_t i = 0; i < quarter_len; i++)
                        {
                            ModIntType temp0 = it0[i], temp1 = it1[i], temp2 = it2[i], temp3 = it3[i];
                            dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            it0[i] = temp0, it1[i] = temp1, it2[i] = temp2 * omega1, it3[i] = temp3 * omega3;
                            omega1 = omega1 * unit_omega1;
                            omega3 = omega3 * unit_omega3;
                        }
                        dif244(in_out, ntt_len / 2);
                        dif244(in_out + quarter_len * 3, ntt_len / 4);
                        dif244(in_out + quarter_len * 2, ntt_len / 4);
                    }
                    static void convolution(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len, bool normlize = true)
                    {
                        dif244(in1, ntt_len);
                        dif244(in2, ntt_len);
                        if (normlize)
                        {
                            const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                            for (size_t i = 0; i < ntt_len; i++)
                            {
                                out[i] = in1[i] * in2[i] * inv_len;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < ntt_len; i++)
                            {
                                out[i] = in1[i] * in2[i];
                            }
                        }
                        INTT::dit244(out, ntt_len);
                    }
                    static void convolution_rec(ModIntType in1[], ModIntType in2[], ModIntType out[], size_t ntt_len, bool normlize = true)
                    {
                        if (ntt_len <= LONG_THRESHOLD)
                        {
                            convolution(in1, in2, out, ntt_len, normlize);
                            return;
                        }
                        const size_t quarter_len = ntt_len / 4;
                        ModIntType unit_omega1 = qpow(ModIntType(root()), (mod() - 1) / ntt_len);
                        ModIntType unit_omega3 = qpow(unit_omega1, 3);
                        ModIntType omega1(1), omega3(1);
                        for (size_t i = 0; i < quarter_len; i++)
                        {
                            ModIntType temp0 = in1[i], temp1 = in1[quarter_len + i], temp2 = in1[quarter_len * 2 + i], temp3 = in1[quarter_len * 3 + i];
                            dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            in1[i] = temp0, in1[quarter_len + i] = temp1, in1[quarter_len * 2 + i] = temp2 * omega1, in1[quarter_len * 3 + i] = temp3 * omega3;

                            temp0 = in2[i], temp1 = in2[quarter_len + i], temp2 = in2[quarter_len * 2 + i], temp3 = in2[quarter_len * 3 + i];
                            dif_butterfly244<ROOT>(temp0, temp1, temp2, temp3);
                            in2[i] = temp0, in2[quarter_len + i] = temp1, in2[quarter_len * 2 + i] = temp2 * omega1, in2[quarter_len * 3 + i] = temp3 * omega3;

                            omega1 = omega1 * unit_omega1;
                            omega3 = omega3 * unit_omega3;
                        }

                        convolution(in1, in2, out, ntt_len / 2, false);
                        convolution(in1 + quarter_len * 2, in2 + quarter_len * 2, out + quarter_len * 2, ntt_len / 4, false);
                        convolution(in1 + quarter_len * 3, in2 + quarter_len * 3, out + quarter_len * 3, ntt_len / 4, false);

                        unit_omega1 = qpow(ModIntType(iroot()), (mod() - 1) / ntt_len);
                        unit_omega3 = qpow(unit_omega1, 3);
                        if (normlize)
                        {
                            const ModIntType inv_len(qpow(ModIntType(ntt_len), mod() - 2));
                            omega1 = inv_len, omega3 = inv_len;
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = out[i] * inv_len, temp1 = out[quarter_len + i] * inv_len, temp2 = out[quarter_len * 2 + i] * omega1, temp3 = out[quarter_len * 3 + i] * omega3;
                                dit_butterfly244<iroot()>(temp0, temp1, temp2, temp3);
                                out[i] = temp0, out[quarter_len + i] = temp1, out[quarter_len * 2 + i] = temp2, out[quarter_len * 3 + i] = temp3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }
                        else
                        {
                            omega1 = 1, omega3 = 1;
                            for (size_t i = 0; i < quarter_len; i++)
                            {
                                ModIntType temp0 = out[i], temp1 = out[quarter_len + i], temp2 = out[quarter_len * 2 + i] * omega1, temp3 = out[quarter_len * 3 + i] * omega3;
                                dit_butterfly244<iroot()>(temp0, temp1, temp2, temp3);
                                out[i] = temp0, out[quarter_len + i] = temp1, out[quarter_len * 2 + i] = temp2, out[quarter_len * 3 + i] = temp3;

                                omega1 = omega1 * unit_omega1;
                                omega3 = omega3 * unit_omega3;
                            }
                        }
                    }
                };
            }
            using NTT1 = split_radix::NTT<MOD1, ROOT1>; // using 64bit integer
            using NTT2 = split_radix::NTT<MOD2, ROOT2>; // using 64bit integer
            using NTT3 = split_radix::NTT<MOD3, ROOT3>; // using 32bit integer, Montgomery speed up
            using NTT4 = split_radix::NTT<MOD4, ROOT4>; // using 32bit integer, Montgomery speed up
            using NTT5 = split_radix::NTT<MOD5, ROOT5>; // using 32bit integer
            using NTT6 = split_radix::NTT<MOD6, ROOT6>; // using 32bit integer
        }
    }
}

void poly_multiply1(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    using namespace std;
    using namespace hint;
    using namespace hint_transform::hint_ntt::split_radix;
    size_t conv_len = m + n + 1, ntt_len = int_ceil2(conv_len);
    using ntt = hint_transform::hint_ntt::split_radix::NTT<998244353, 3>;
    using ModInt = ntt::ModIntType;
    ModInt *a_ntt = new ModInt[ntt_len];
    ModInt *b_ntt = new ModInt[ntt_len];
    std::fill(a_ntt + n + 1, a_ntt + ntt_len, ModInt{});
    std::fill(b_ntt + m + 1, b_ntt + ntt_len, ModInt{});
    std::copy(a, a + n + 1, a_ntt);
    std::copy(b, b + m + 1, b_ntt);
    ntt::convolution(a_ntt, b_ntt, a_ntt, ntt_len);
    size_t rem_len = conv_len % 16, i = 0;
    for (; i < conv_len; i++)
    {
        c[i] = uint32_t(a_ntt[i]);
    }
    delete[] a_ntt;
    delete[] b_ntt;
}

inline void abs_mul32_ntt(const uint32_t in1[], size_t len1,
                          const uint32_t in2[], size_t len2,
                          uint32_t out[])
{
    using namespace hint::hint_transform::hint_ntt;
    size_t out_len = len1 + len2, conv_len = out_len - 1;
    size_t ntt_len = hint::int_ceil2(conv_len);
    std::vector<NTT1::ModIntType> buffer1(ntt_len);
    {
        std::vector<NTT1::ModIntType> buffer2(ntt_len);
        std::copy(in2, in2 + len2, buffer2.begin());
        std::copy(in1, in1 + len1, buffer1.begin());
        NTT1::convolution(buffer1.data(), buffer2.data(), buffer1.data(), ntt_len);
    }
    std::vector<NTT2::ModIntType> buffer3(ntt_len);
    {

        std::vector<NTT2::ModIntType> buffer4(ntt_len);
        std::copy(in2, in2 + len2, buffer4.begin());
        std::copy(in1, in1 + len1, buffer3.begin());
        NTT2::convolution(buffer3.data(), buffer4.data(), buffer3.data(), ntt_len);
    }
    Uint128 carry = 0;
    for (size_t i = 0; i < conv_len; i++)
    {
        carry += qcrt<NTT1::mod(), NTT2::mod()>(buffer1[i], buffer3[i]);
        out[i] = uint32_t(carry);
        carry >>= 32;
    }
    out[conv_len] = uint32_t(carry);
}

inline void abs_mul16_ntt(const uint16_t in1[], size_t len1,
                          const uint16_t in2[], size_t len2,
                          uint16_t out[])
{
    using namespace hint::hint_transform::hint_ntt;
    size_t out_len = len1 + len2, conv_len = out_len - 1;
    size_t ntt_len = hint::int_ceil2(conv_len);
    std::vector<NTT1::ModIntType> buffer1(ntt_len);
    {
        std::vector<NTT1::ModIntType> buffer2(ntt_len);
        std::copy(in2, in2 + len2, buffer2.begin());
        std::copy(in1, in1 + len1, buffer1.begin());
        NTT1::convolution(buffer1.data(), buffer2.data(), buffer1.data(), ntt_len);
    }
    uint64_t carry = 0;
    for (size_t i = 0; i < conv_len; i++)
    {
        carry += uint64_t(buffer1[i]);
        out[i] = uint16_t(carry);
        carry >>= 16;
    }
    out[conv_len] = uint16_t(carry);
}

void test_mul()
{
    std::ios::sync_with_stdio(false);
    int shift;
    std::cin >> shift;
    size_t len = 1 << shift;
    std::vector<uint32_t> a(len, UINT32_MAX), b = a;
    std::vector<uint32_t> c(len * 2);
    auto t1 = std::chrono::steady_clock::now();
    abs_mul32_ntt(a.data(), len, b.data(), len, c.data());
    auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < c.size(); i++)
    {
        std::cout << std::hex << c[i] << "\n";
    }
    std::cout << std::dec << "time:" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}

template <uint64_t ROOT, typename ModInt>
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
    }
}

template <uint64_t ROOT, typename ModInt>
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
    using namespace hint_transform;
    using namespace hint_ntt;
    constexpr uint64_t mod1 = 1945555039024054273, root1 = 5;
    constexpr uint64_t mod2 = 4179340454199820289, root2 = 3;
    constexpr uint64_t mod = 998244353, root = 3;

    // using ModInt = MontInt32Lazy<mod>;
    using ntt = split_radix::NTT<mod, root>;
    using ntt2 = split_radix::NTT<mod, root>;
    using ModInt = ntt::ModIntType;

    // std::cin >> s;
    size_t len = 1 << s;
    size_t times = 1000; // std::max<size_t>(1, (1 << 25) / len);
    std::vector<ModInt> a(len);
    std::vector<ModInt> b(len);
    std::vector<uint32_t> c(len);
    for (size_t i = 0; i < len; i++)
    {
        a[i] = uint64_t(i);
        b[i] = uint64_t(i);
        c[i] = uint64_t(i);
    }

    auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < times; i++)
    {
        // ntt_dif<root>(a.data(), len);
        // ntt_dif<root>(a.data(), len);
        // ntt_dit<root>(a.data(), len);
        // ntt2::dit2488(a.data(), len);
        // poly::fast_number_theoretic_transform_core::NTT(c.data(), len);
        // poly::fast_number_theoretic_transform_core::NTT(c.data(), len);
        // poly::fast_number_theoretic_transform_core::INTT<true>(c.data(), len);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < times; i++)
    {
        ntt::dif244(b.data(), len);
        ntt::dif244(b.data(), len);
        ntt::dit244(b.data(), len);
        // ntt_dif<root>(a.data(), len);
        // ntt_dif<root>(a.data(), len);
        // ntt_dit<root>(b.data(), len);
    }
    auto t3 = std::chrono::steady_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    auto time2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
    for (size_t i = 0; i < std::min<size_t>(len, 1024); i++)
    {
        if (uint64_t(a[i]) != uint64_t(b[i]))
        {
            std::cout << i << ":\t" << uint64_t(a[i]) << "\t" << uint64_t(b[i]) << "\n";
            return;
        }
    }
    std::cout << s << ":\n";
    std::cout << time1 << "\t" << time2 << "\t" << time1 / time2 << "\n";
}

template <typename T>
std::vector<T> poly_multiply(const std::vector<T> &in1, const std::vector<T> &in2)
{
    using namespace hint::hint_transform::hint_ntt;
    using NTT = NTT1;
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    std::vector<T> result(out_len);
    size_t ntt_len = hint::int_floor2(out_len);
    std::vector<NTT::ModIntType> buffer1(ntt_len), buffer2(ntt_len);
    std::copy(in1.begin(), in1.end(), buffer1.begin());
    std::copy(in2.begin(), in2.end(), buffer2.begin());
    NTT::convolution_rec(buffer1.data(), buffer2.data(), buffer1.data(), ntt_len);
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(buffer1[i]);
    }
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
    size_t len = size_t(1) << n; // 变换长度
    uint64_t ele = 9999;
    std::vector<uint64_t> in1(len / 2, ele);
    std::vector<uint64_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    auto t1 = std::chrono::steady_clock::now();
    std::vector<uint64_t> res = poly_multiply(in1, in2);
    auto t2 = std::chrono::steady_clock::now();
    result_test<uint64_t>(res, ele); // 结果校验
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}

void ntt_perf(int s = 24)
{
    using namespace hint;
    using namespace hint_transform;
    using namespace hint_ntt;
    constexpr uint64_t mod1 = 1945555039024054273, root1 = 5;
    constexpr uint64_t mod2 = 4179340454199820289, root2 = 3;
    constexpr uint64_t mod = mod2, root = root2;

    using ntt = split_radix::NTT<mod, root, Uint128>;
    using ModInt = ntt::ModIntType;

    size_t len = 1 << s;
    size_t times = std::max<size_t>(1, (1 << 22) / len);
    std::vector<ModInt> a(len);

    auto t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < len; i++)
    {
        a[i] = uint64_t(i);
    }
    for (size_t i = 0; i < times; i++)
    {
        ntt::dif244(a.data(), len);
        ntt::dif244(a.data(), len);
        ntt::dit244(a.data(), len);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < len; i++)
    {
        a[i] = uint64_t(i);
    }
    for (size_t i = 0; i < times; i++)
    {
        ntt_dif<root>(a.data(), len);
        ntt_dif<root>(a.data(), len);
        ntt_dit<root>(a.data(), len);
    }
    auto t3 = std::chrono::steady_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    auto time2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
    std::cout << s << ":\n";
    std::cout << time1 << "\t" << time2 << "\t" << time2 / time1 << "\n";
}

void ntt_perf_loop()
{
    for (int i = 10; i <= 30; i++)
    {
        ntt_perf(i);
    }
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
    for (size_t i = 0; i < 1000; i++)
    {
        poly_multiply1(a, len - 1, b, len - 1, c);
    }
    auto t2 = chrono::steady_clock::now();
    for (auto i : c)
    {
        // std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}

int main()
{
#ifdef UMUL128
    std::cout << "fast win\n";
#endif

#ifdef UINT128T
    std::cout << "fast unix\n";
#endif

    // ntt_check(23);
    test_mul();//测试乘法
    // test_ntt();//测试卷积
    // test_poly();
    // std::cout << hint::hint_transform::hint_ntt::add_count << "\t" << hint::hint_transform::hint_ntt::mul_count << "\n";
    // ntt_perf_loop();
}