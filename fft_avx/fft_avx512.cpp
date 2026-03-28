#ifndef FFT_AVX_HPP
#define FFT_AVX_HPP
#define __AVX512F__
#include "hint_simd.hpp"
#include <complex>
#include <array>
#include <cstdint>
#include <cassert>
#include <cstring>

namespace hint
{
    using Float32 = float;
    using Float64 = double;
    using Complex32 = std::complex<Float32>;
    using Complex64 = std::complex<Float64>;

    constexpr Float64 HINT_PI = 3.141592653589793238462643;
    constexpr Float64 HINT_2PI = HINT_PI * 2;
    constexpr Float64 COS_PI_8 = 0.707106781186547524400844;

    constexpr size_t FFT_MAX_LEN = size_t(1) << 23;

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

    template <typename T>
    constexpr T int_ceil2(T n)
    {
        constexpr int bits = sizeof(n) * 8;
        n--;
        for (int i = 1; i < bits; i *= 2)
        {
            n |= (n >> i);
        }
        return n + 1;
    }

    template <typename IntTy>
    constexpr bool is_2pow(IntTy n)
    {
        return n != 0 && (n & (n - 1)) == 0;
    }

    // 求整数的对数
    template <typename T>
    constexpr int hint_log2(T n)
    {
        constexpr int bits = sizeof(n) * 8;
        int l = -1, r = bits;
        while ((l + 1) != r)
        {
            int mid = (l + r) / 2;
            if ((T(1) << mid) > n)
            {
                r = mid;
            }
            else
            {
                l = mid;
            }
        }
        return l;
    }
    constexpr int hint_ctz(uint32_t x)
    {
        int r0 = 31;
        x &= (-x);
        if (x & 0x55555555)
        {
            r0 &= ~1;
        }
        if (x & 0x33333333)
        {
            r0 &= ~2;
        }
        if (x & 0x0F0F0F0F)
        {
            r0 &= ~4;
        }
        if (x & 0x00FF00FF)
        {
            r0 &= ~8;
        }
        if (x & 0x0000FFFF)
        {
            r0 &= ~16;
        }
        r0 += (x == 0);
        return r0;
    }

    constexpr int hint_ctz(uint64_t x)
    {
        int r0 = 63;
        x &= (-x);
        if (x & 0x5555555555555555)
        {
            r0 &= ~1; // -1
        }
        if (x & 0x3333333333333333)
        {
            r0 &= ~2; // -2
        }
        if (x & 0x0F0F0F0F0F0F0F0F)
        {
            r0 &= ~4; // -4
        }
        if (x & 0x00FF00FF00FF00FF)
        {
            r0 &= ~8; // -8
        }
        if (x & 0x0000FFFF0000FFFF)
        {
            r0 &= ~16; // -16
        }
        if (x & 0x00000000FFFFFFFF)
        {
            r0 &= ~32; // -32
        }
        r0 += (x == 0);
        return r0;
    }

    constexpr int hint_popcnt(uint32_t n)
    {
        constexpr uint32_t mask55 = 0x55555555;
        constexpr uint32_t mask33 = 0x33333333;
        constexpr uint32_t mask0f = 0x0f0f0f0f;
        constexpr uint32_t maskff = 0x00ff00ff;
        n = (n & mask55) + ((n >> 1) & mask55);
        n = (n & mask33) + ((n >> 2) & mask33);
        n = (n & mask0f) + ((n >> 4) & mask0f);
        n = (n & maskff) + ((n >> 8) & maskff);
        return uint16_t(n) + (n >> 16);
    }
    constexpr int hint_popcnt(uint64_t n)
    {
        constexpr uint64_t mask5555 = 0x5555555555555555;
        constexpr uint64_t mask3333 = 0x3333333333333333;
        constexpr uint64_t mask0f0f = 0x0f0f0f0f0f0f0f0f;
        constexpr uint64_t mask00ff = 0x00ff00ff00ff00ff;
        constexpr uint64_t maskffff = 0x0000ffff0000ffff;
        n = (n & mask5555) + ((n >> 1) & mask5555);
        n = (n & mask3333) + ((n >> 2) & mask3333);
        n = (n & mask0f0f) + ((n >> 4) & mask0f0f);
        n = (n & mask00ff) + ((n >> 8) & mask00ff);
        n = (n & maskffff) + ((n >> 16) & maskffff);
        return uint32_t(n) + (n >> 32);
    }

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
    constexpr uint32_t bitrev(uint32_t n, int len)
    {
        assert(len <= 32);
        return bitrev32(n) >> (32 - len);
    }

    template <typename T>
    void fill_zero(T *begin, T *end)
    {
        std::memset(begin, 0, (end - begin) * sizeof(T));
    }

    // FFT与类FFT变换的命名空间
    namespace transform
    {
        using namespace hint_simd;

        template <typename T>
        inline void transform2(T &sum, T &diff)
        {
            T temp0 = sum, temp1 = diff;
            sum = temp0 + temp1;
            diff = temp0 - temp1;
        }

        template <typename T>
        inline void transform2(const T a, const T b, T &sum, T &diff)
        {
            sum = a + b;
            diff = a - b;
        }
        namespace fft
        {
            using F64 = Float64;
            using C64 = std::complex<F64>;
            using F64X8 = hint_simd::Float64X8;
            using C64X8 = hint_simd::Complex64X8;
            template <typename Float, size_t OMEGA_LEN>
            class TableFix
            {
                alignas(64) std::array<Float, OMEGA_LEN * 2> table;

            public:
                TableFix(size_t theta_divider, size_t factor, size_t stride)
                {
                    const Float theta = -HINT_2PI * factor / theta_divider;
                    assert(OMEGA_LEN % stride == 0);
                    for (size_t begin = 0, index = 0; begin < OMEGA_LEN * 2; begin += stride * 2)
                    {
                        for (size_t j = 0; j < stride; j++, index++)
                        {
                            table[begin + j] = std::cos(theta * index);
                            table[begin + j + stride] = std::sin(theta * index);
                        }
                    }
                }
                constexpr const Float &operator[](size_t index) const
                {
                    return table[index];
                }
                constexpr const Float *getOmegaIt(size_t index) const
                {
                    return &table[index];
                }
            };
            template <typename Float, int LOG_BEGIN, int LOG_END, int DIV>
            class TableFixMulti
            {
                static_assert(LOG_END >= LOG_BEGIN);
                static_assert(is_2pow(DIV));
                static constexpr size_t TABLE_CPX_LEN = (size_t(1) << (LOG_END + 1)) / DIV;
                // alignas(64) std::array<Float, TABLE_CPX_LEN * 2> table;
                AlignMem<Float> table;
                size_t log_current;
                size_t stride;
                Float factor;

            public:
                TableFixMulti(Float factor_in, size_t stride_in = 8, bool preload = true)
                    : table(TABLE_CPX_LEN * 2), log_current(LOG_BEGIN), stride(stride_in), factor(factor_in)
                {
                    assert(((size_t(1) << LOG_BEGIN) / DIV) % stride == 0);
                    auto it = getBeginLog(LOG_BEGIN);
                    size_t len = size_t(1) << LOG_BEGIN, cpx_len = len / DIV;
                    Float theta = -HINT_2PI * factor / len;
                    for (size_t i = 0; i < cpx_len; i++)
                    {
                        it[0] = std::cos(theta * i), it[stride] = std::sin(theta * i);
                        it += (i % stride == stride - 1 ? stride + 1 : 1);
                    }
                    if (preload)
                    {
                        initBottomUp(LOG_END);
                    }
                }
                void initBottomUp(size_t log_rank)
                {
                    static_assert(std::is_same<Float, Float64>::value);
                    assert(stride == 8); // TODO: support other stride
                    assert(log_rank <= LOG_END);
                    for (int log_len = log_current + 1; log_len <= log_rank; log_len++)
                    {
                        size_t len = size_t(1) << log_len, cpx_len = len / DIV;
                        Float theta = -HINT_2PI * factor / len;
                        auto it = getBeginLog(log_len), it_last = getBeginLog(log_len - 1);
                        C64X8 unit(std::cos(theta), std::sin(theta));
                        for (auto end = it + cpx_len * 2; it < end; it += 32, it_last += 16)
                        {
                            C64X8 omega0, omega1;
                            omega0.load(it_last);
                            omega1 = omega0.mul(unit);
                            transpose64_2X8(omega0.real, omega1.real);
                            transpose64_2X8(omega0.imag, omega1.imag);
                            omega0.store(it), omega1.store(it + 16);
                        }
                    }
                    log_current = std::max(log_current, log_rank);
                }
                constexpr const Float *getBeginLog(int log_rank) const
                {
                    return getBegin(size_t(1) << log_rank);
                }
                constexpr Float *getBeginLog(int log_rank)
                {
                    return getBegin(size_t(1) << log_rank);
                }
                constexpr const Float *getBegin(size_t rank) const
                {
                    return &table[rank * 2 / DIV];
                }
                constexpr Float *getBegin(size_t rank)
                {
                    return &table[rank * 2 / DIV];
                }
            };

            struct FFT
            {
                // x0_out <- x0 + x1, x1_out <- (x0 - x1) * -i
                template <typename Float>
                static void transform2MulNegI(Float &r0, Float &i0, Float &r1, Float &i1)
                {
                    auto tr = r1, ti = i1;
                    r1 = i0 - ti, i1 = tr - r0;
                    r0 = r0 + tr, i0 = i0 + ti;
                }
                // x0_out <- x0 + x1 * i, x1_out <- x0 - x1 * i
                template <typename Float>
                static void mulITransform2(Float &r0, Float &i0, Float &r1, Float &i1)
                {
                    auto tr = r1, ti = i1;
                    r1 = r0 + ti, i1 = i0 - tr;
                    r0 = r0 - ti, i0 = i0 + tr;
                }

                template <typename Float>
                static void difSplit(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r0, r2), transform2(i0, i2);
                    transform2MulNegI(r1, i1, r3, i3);
                    transform2(r2, r3), transform2(i2, i3);
                }
                template <typename Float>
                static void iditSplit(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r2, r3), transform2(i2, i3);
                    mulITransform2(r1, i1, r3, i3);
                    transform2(r0, r2), transform2(i0, i2);
                }

                template <typename Float>
                static void dif4(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    difSplit(r0, i0, r1, i1, r2, i2, r3, i3);
                    transform2(r0, r1), transform2(i0, i1);
                }
                template <typename Float>
                static void idit4(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r0, r1), transform2(i0, i1);
                    iditSplit(r0, i0, r1, i1, r2, i2, r3, i3);
                }
            };
            class FFT512 : public FFT
            {
            public:
                static constexpr size_t LOG_SHORT = 10;
                static constexpr size_t LOG_MAX = 23;
                static constexpr size_t SHORT_LEN = size_t(1) << LOG_SHORT;
                static constexpr size_t MAX_LEN = size_t(1) << LOG_MAX;

            private:
                using TableFix8 = TableFix<Float64, 8>;
                using TableFix32 = TableFix<Float64, 32>;
                static const TableFix8 table_16_1, table_32_1, table_32_2, table_32_3;
                static const TableFix32 table_64_1, table_128_1, table_128_3, table_256_1, table_256_3, table_256_5, table_256_7;
                static const TableFixMulti<Float64, 9, LOG_MAX, 8> multi_table_1, multi_table_2, multi_table_3;

                static constexpr const Float64 *tb16_1 = &table_16_1[0];
                static constexpr const Float64 *tb32_1 = &table_32_1[0];
                static constexpr const Float64 *tb32_2 = &table_32_2[0];
                static constexpr const Float64 *tb32_3 = &table_32_3[0];
                static constexpr const Float64 *tb64_1 = &table_64_1[0];
                static constexpr const Float64 *tb128_1 = &table_128_1[0];
                static constexpr const Float64 *tb128_3 = &table_128_3[0];
                static constexpr const Float64 *tb256_1 = &table_256_1[0];
                static constexpr const Float64 *tb256_3 = &table_256_3[0];
                static constexpr const Float64 *tb256_5 = &table_256_5[0];
                static constexpr const Float64 *tb256_7 = &table_256_7[0];

            public:
                static C64X8 mulOmega8_1(C64X8 x) noexcept
                {
                    constexpr uint64_t MAGIC_HSQRT2 = 4604544271217802189ull; // sqrt(2) / 2
                    const F64X8 half_sqrt2 = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_HSQRT2));
                    return C64X8(x.real + x.imag, x.imag - x.real) * half_sqrt2;
                }
                static C64X8 mulOmega8_3(C64X8 x) noexcept
                {
                    constexpr uint64_t MAGIC_HSQRT2 = 13827916308072577997ull; // -sqrt(2) / 2
                    const F64X8 half_sqrt2 = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_HSQRT2));
                    return C64X8(x.real - x.imag, x.real + x.imag) * half_sqrt2;
                }
                static C64X8 mulOmega8_5(C64X8 x) noexcept
                {
                    constexpr uint64_t MAGIC_HSQRT2 = 13827916308072577997ull; // sqrt(2) / 2
                    const F64X8 half_sqrt2 = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_HSQRT2));
                    return C64X8(x.real + x.imag, x.imag - x.real) * half_sqrt2;
                }
                static C64X8 mulOmega8_7(C64X8 x) noexcept
                {
                    constexpr uint64_t MAGIC_HSQRT2 = 4604544271217802189ull; // sqrt(2) / 2
                    const F64X8 half_sqrt2 = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_HSQRT2));
                    return C64X8(x.real - x.imag, x.real + x.imag) * half_sqrt2;
                }

                static void difSplitX8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3) noexcept
                {
                    difSplit(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                }
                static void iditSplitX8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3) noexcept
                {
                    iditSplit(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                }

                static void dif4X8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3) noexcept
                {
                    dif4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                }
                static void idit4X8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3) noexcept
                {
                    idit4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                }

                static void difSplit28888(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    transform2(c0, c4);
                    transform2(c1, c5);
                    c5 = mulOmega8_1(c5);
                    transform2MulNegI(c2.real, c2.imag, c6.real, c6.imag);
                    transform2(c3, c7);
                    c7 = mulOmega8_3(c7);
                    dif4(c4.real, c4.imag, c5.real, c5.imag, c6.real, c6.imag, c7.real, c7.imag);
                }

                static void iditSplit28888(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    idit4(c4.real, c4.imag, c5.real, c5.imag, c6.real, c6.imag, c7.real, c7.imag);
                    transform2(c0, c4);
                    c5 = mulOmega8_7(c5);
                    transform2(c1, c5);
                    mulITransform2(c2.real, c2.imag, c6.real, c6.imag);
                    c7 = mulOmega8_5(c7);
                    transform2(c3, c7);
                }

                static void dif8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    transform2(c0, c4);
                    transform2(c1, c5);
                    c5 = mulOmega8_1(c5);
                    transform2MulNegI(c2.real, c2.imag, c6.real, c6.imag);
                    transform2(c3, c7);
                    c7 = mulOmega8_3(c7);
                    dif4X8(c0, c1, c2, c3);
                    dif4X8(c4, c5, c6, c7);
                }

                static void idit8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    idit4X8(c0, c1, c2, c3);
                    idit4X8(c4, c5, c6, c7);
                    transform2(c0, c4);
                    c5 = mulOmega8_7(c5);
                    transform2(c1, c5);
                    mulITransform2(c2.real, c2.imag, c6.real, c6.imag);
                    c7 = mulOmega8_5(c7);
                    transform2(c3, c7);
                }

                static void dif8X8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    transpose64_8X8(c0.real, c1.real, c2.real, c3.real, c4.real, c5.real, c6.real, c7.real);
                    transpose64_8X8(c0.imag, c1.imag, c2.imag, c3.imag, c4.imag, c5.imag, c6.imag, c7.imag);
                    dif8(c0, c1, c2, c3, c4, c5, c6, c7);
                    transpose64_8X8(c0.real, c1.real, c2.real, c3.real, c4.real, c5.real, c6.real, c7.real);
                    transpose64_8X8(c0.imag, c1.imag, c2.imag, c3.imag, c4.imag, c5.imag, c6.imag, c7.imag);
                }

                static void idit8X8(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    transpose64_8X8(c0.real, c1.real, c2.real, c3.real, c4.real, c5.real, c6.real, c7.real);
                    transpose64_8X8(c0.imag, c1.imag, c2.imag, c3.imag, c4.imag, c5.imag, c6.imag, c7.imag);
                    idit8(c0, c1, c2, c3, c4, c5, c6, c7);
                    transpose64_8X8(c0.real, c1.real, c2.real, c3.real, c4.real, c5.real, c6.real, c7.real);
                    transpose64_8X8(c0.imag, c1.imag, c2.imag, c3.imag, c4.imag, c5.imag, c6.imag, c7.imag);
                }

                static void dif16X4(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    C64X8 omega;
                    omega.load(tb16_1);
                    transform2(c0, c1);
                    c1 = c1.mul(omega);
                    transform2(c2, c3);
                    c3 = c3.mul(omega);
                    transform2(c4, c5);
                    c5 = c5.mul(omega);
                    transform2(c6, c7);
                    c7 = c7.mul(omega);
                    dif8X8(c0, c1, c2, c3, c4, c5, c6, c7);
                }
                static void idit16X4(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    C64X8 omega;
                    idit8X8(c0, c1, c2, c3, c4, c5, c6, c7);
                    omega.load(tb16_1);
                    c1 = c1.mulConj(omega);
                    transform2(c0, c1);
                    c3 = c3.mulConj(omega);
                    transform2(c2, c3);
                    c5 = c5.mulConj(omega);
                    transform2(c4, c5);
                    c7 = c7.mulConj(omega);
                    transform2(c6, c7);
                }

                static void dif32X2(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    C64X8 omega;
                    dif4X8(c0, c1, c2, c3);
                    dif4X8(c4, c5, c6, c7);
                    omega.load(tb32_2);
                    c1 = c1.mul(omega);
                    c5 = c5.mul(omega);
                    omega.load(tb32_1);
                    c2 = c2.mul(omega);
                    c6 = c6.mul(omega);
                    omega.load(tb32_3);
                    c3 = c3.mul(omega);
                    c7 = c7.mul(omega);
                    dif8X8(c0, c1, c2, c3, c4, c5, c6, c7);
                }
                static void idit32X2(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    C64X8 omega;
                    idit8X8(c0, c1, c2, c3, c4, c5, c6, c7);
                    omega.load(tb32_2);
                    c1 = c1.mulConj(omega);
                    c5 = c5.mulConj(omega);
                    omega.load(tb32_1);
                    c2 = c2.mulConj(omega);
                    c6 = c6.mulConj(omega);
                    omega.load(tb32_3);
                    c3 = c3.mulConj(omega);
                    c7 = c7.mulConj(omega);
                    idit4X8(c0, c1, c2, c3);
                    idit4X8(c4, c5, c6, c7);
                }
                static void dif32X2(F64 inout[]) noexcept
                {
                    auto p = reinterpret_cast<C64X8 *>(inout);
                    dif32X2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                }
                static void idit32X2(F64 inout[]) noexcept
                {
                    auto p = reinterpret_cast<C64X8 *>(inout);
                    idit32X2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                }
                static void dif64(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    C64X8 omega;
                    omega.load(tb64_1);
                    transform2(c0, c4);
                    c4 = c4.mul(omega);
                    omega.load(tb64_1 + 16);
                    transform2(c1, c5);
                    c5 = c5.mul(omega);
                    omega.load(tb64_1 + 32);
                    transform2(c2, c6);
                    c6 = c6.mul(omega);
                    omega.load(tb64_1 + 48);
                    transform2(c3, c7);
                    c7 = c7.mul(omega);
                    dif32X2(c0, c1, c2, c3, c4, c5, c6, c7);
                }
                static void idit64(C64X8 &c0, C64X8 &c1, C64X8 &c2, C64X8 &c3, C64X8 &c4, C64X8 &c5, C64X8 &c6, C64X8 &c7) noexcept
                {
                    idit32X2(c0, c1, c2, c3, c4, c5, c6, c7);
                    C64X8 omega;
                    omega.load(tb64_1);
                    c4 = c4.mulConj(omega);
                    transform2(c0, c4);
                    omega.load(tb64_1 + 16);
                    c5 = c5.mulConj(omega);
                    transform2(c1, c5);
                    omega.load(tb64_1 + 32);
                    c6 = c6.mulConj(omega);
                    transform2(c2, c6);
                    omega.load(tb64_1 + 48);
                    c7 = c7.mulConj(omega);
                    transform2(c3, c7);
                }

                static void dif64(F64 inout[]) noexcept
                {
                    auto p = reinterpret_cast<C64X8 *>(inout);
                    dif64(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                }
                static void idit64(F64 inout[]) noexcept
                {
                    auto p = reinterpret_cast<C64X8 *>(inout);
                    idit64(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                }

                static void dif128(F64 inout[]) noexcept
                {
                    C64X8 c0, c1, c2, c3, c4, omega1, omega3;
                    auto tb1 = tb128_1, tb3 = tb128_3;
                    for (auto it = inout; it < inout + 64; it += 16, tb1 += 16, tb3 += 16)
                    {
                        c0.load(it), c1.load(it + 64), c2.load(it + 128), c3.load(it + 192);
                        difSplitX8(c0, c1, c2, c3);
                        omega1.load(tb1), omega3.load(tb3);
                        c2 = c2.mul(omega1), c3 = c3.mul(omega3);
                        c0.store(it), c1.store(it + 64), c2.store(it + 128), c3.store(it + 192);
                    }
                    dif64(inout);
                    dif32X2(inout + 128);
                }
                static void idit128(F64 inout[]) noexcept
                {
                    idit64(inout);
                    idit32X2(inout + 128);
                    C64X8 c0, c1, c2, c3, c4, omega1, omega3;
                    auto tb1 = tb128_1, tb3 = tb128_3;
                    for (auto it = inout; it < inout + 64; it += 16, tb1 += 16, tb3 += 16)
                    {
                        c0.load(it), c1.load(it + 64), c2.load(it + 128), c3.load(it + 192);
                        omega1.load(tb1), omega3.load(tb3);
                        c2 = c2.mulConj(omega1), c3 = c3.mulConj(omega3);
                        iditSplitX8(c0, c1, c2, c3);
                        c0.store(it), c1.store(it + 64), c2.store(it + 128), c3.store(it + 192);
                    }
                }

                static void dif256(F64 inout[]) noexcept
                {
                    C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega;
                    auto tb1 = tb256_1, tb3 = tb256_3, tb5 = tb256_5, tb7 = tb256_7;
                    for (auto it = inout; it < inout + 64; it += 16, tb1 += 16, tb3 += 16, tb5 += 16, tb7 += 16)
                    {
                        c0.load(it), c1.load(it + 64), c2.load(it + 128), c3.load(it + 192);
                        c4.load(it + 256), c5.load(it + 320), c6.load(it + 384), c7.load(it + 448);
                        difSplit28888(c0, c1, c2, c3, c4, c5, c6, c7);
                        omega.load(tb1), c4 = c4.mul(omega);
                        omega.load(tb5), c5 = c5.mul(omega);
                        omega.load(tb3), c6 = c6.mul(omega);
                        omega.load(tb7), c7 = c7.mul(omega);
                        c0.store(it), c1.store(it + 64), c2.store(it + 128), c3.store(it + 192);
                        c4.store(it + 256), c5.store(it + 320), c6.store(it + 384), c7.store(it + 448);
                    }
                    dif128(inout);
                    dif32X2(inout + 256);
                    dif32X2(inout + 384);
                }
                static void idit256(F64 inout[]) noexcept
                {
                    idit128(inout);
                    idit32X2(inout + 256);
                    idit32X2(inout + 384);
                    C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega;
                    auto tb1 = tb256_1, tb3 = tb256_3, tb5 = tb256_5, tb7 = tb256_7;
                    for (auto it = inout; it < inout + 64; it += 16, tb1 += 16, tb3 += 16, tb5 += 16, tb7 += 16)
                    {
                        c0.load(it), c1.load(it + 64), c2.load(it + 128), c3.load(it + 192);
                        c4.load(it + 256), c5.load(it + 320), c6.load(it + 384), c7.load(it + 448);
                        omega.load(tb1), c4 = c4.mulConj(omega);
                        omega.load(tb5), c5 = c5.mulConj(omega);
                        omega.load(tb3), c6 = c6.mulConj(omega);
                        omega.load(tb7), c7 = c7.mulConj(omega);
                        iditSplit28888(c0, c1, c2, c3, c4, c5, c6, c7);
                        c0.store(it), c1.store(it + 64), c2.store(it + 128), c3.store(it + 192);
                        c4.store(it + 256), c5.store(it + 320), c6.store(it + 384), c7.store(it + 448);
                    }
                }

                template <bool FROM_RIRI = false>
                static void difRecSplit28888(Float64 in_out[], size_t float_len) noexcept
                {
                    using FromRIRI = std::integral_constant<bool, FROM_RIRI>;
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= 256)
                    {
                        assert(!FROM_RIRI);
                        if (fft_len == 256)
                        {
                            dif256(in_out);
                        }
                        else if (fft_len == 128)
                        {
                            dif128(in_out);
                        }
                        else if (fft_len == 64)
                        {
                            dif64(in_out);
                        }
                        return;
                    }
                    const size_t stride = float_len / 8, half = float_len / 2;
                    auto tb1 = multi_table_1.getBegin(fft_len);
                    auto tb3 = multi_table_3.getBegin(fft_len);
                    auto it0 = in_out, it1 = in_out + stride, it2 = in_out + stride * 2, it3 = in_out + stride * 3;
                    for (auto end = in_out + stride; it0 < end;
                         it0 += 16, it1 += 16, it2 += 16, it3 += 16,
                              tb1 += 16, tb3 += 16)
                    {
                        C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega1, omega3;
                        c0.load(it0, FromRIRI{}), c1.load(it1, FromRIRI{}), c2.load(it2, FromRIRI{}), c3.load(it3, FromRIRI{});
                        c4.load(it0 + half, FromRIRI{}), c5.load(it1 + half, FromRIRI{}), c6.load(it2 + half, FromRIRI{}), c7.load(it3 + half, FromRIRI{});
                        difSplit28888(c0, c1, c2, c3, c4, c5, c6, c7);
                        omega1.load(tb1), c4 = c4.mul(omega1);
                        omega3.load(tb3), c6 = c6.mul(omega3);
                        c5 = c5.mul(omega3.mul(omega1.square()));
                        c7 = c7.mul(omega1.mul(omega3.square()));
                        c0.store(it0), c1.store(it1), c2.store(it2), c3.store(it3);
                        c4.store(it0 + half), c5.store(it1 + half), c6.store(it2 + half), c7.store(it3 + half);
                    }
                    difRecSplit28888(in_out, half), in_out += half;
                    difRecSplit28888(in_out, stride), in_out += stride;
                    difRecSplit28888(in_out, stride), in_out += stride;
                    difRecSplit28888(in_out, stride), in_out += stride;
                    difRecSplit28888(in_out, stride);
                }

                template <bool TO_RIRI = false>
                static void iditRecSplit28888(Float64 in_out[], size_t float_len) noexcept
                {
                    using ToRIRI = std::integral_constant<bool, TO_RIRI>;
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= 256)
                    {
                        assert(!TO_RIRI);
                        if (fft_len == 256)
                        {
                            idit256(in_out);
                        }
                        else if (fft_len == 128)
                        {
                            idit128(in_out);
                        }
                        else if (fft_len == 64)
                        {
                            idit64(in_out);
                        }
                        return;
                    }
                    const size_t stride = float_len / 8, half = float_len / 2;
                    auto it0 = in_out, it1 = in_out + stride, it2 = in_out + stride * 2, it3 = in_out + stride * 3;
                    iditRecSplit28888(it0, half);
                    iditRecSplit28888(it0 + half, stride);
                    iditRecSplit28888(it1 + half, stride);
                    iditRecSplit28888(it2 + half, stride);
                    iditRecSplit28888(it3 + half, stride);
                    auto tb1 = multi_table_1.getBegin(fft_len);
                    auto tb3 = multi_table_3.getBegin(fft_len);
                    for (auto end = in_out + stride; it0 < end;
                         it0 += 16, it1 += 16, it2 += 16, it3 += 16,
                              tb1 += 16, tb3 += 16)
                    {
                        C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega1, omega3;
                        c0.load(it0), c1.load(it1), c2.load(it2), c3.load(it3);
                        c4.load(it0 + half), c5.load(it1 + half), c6.load(it2 + half), c7.load(it3 + half);
                        omega1.load(tb1), c4 = c4.mulConj(omega1);
                        omega3.load(tb3), c6 = c6.mulConj(omega3);
                        c5 = c5.mulConj(omega3.mul(omega1.square()));
                        c7 = c7.mulConj(omega1.mul(omega3.square()));
                        iditSplit28888(c0, c1, c2, c3, c4, c5, c6, c7);
                        c0.store(it0, ToRIRI{}), c1.store(it1, ToRIRI{}), c2.store(it2, ToRIRI{}), c3.store(it3, ToRIRI{});
                        c4.store(it0 + half, ToRIRI{}), c5.store(it1 + half, ToRIRI{}), c6.store(it2 + half, ToRIRI{}), c7.store(it3 + half, ToRIRI{});
                    }
                }

                template <bool FROM_RIRI = false>
                static void difRecRadix8(Float64 in_out[], size_t float_len) noexcept
                {
                    using FromRIRI = std::integral_constant<bool, FROM_RIRI>;
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= 256)
                    {
                        assert(!FROM_RIRI);
                        if (fft_len == 256)
                        {
                            dif256(in_out);
                        }
                        else if (fft_len == 128)
                        {
                            dif128(in_out);
                        }
                        else if (fft_len == 64)
                        {
                            dif64(in_out);
                        }
                        return;
                    }
                    const size_t stride = float_len / 8, half = float_len / 2;
                    auto tb1 = multi_table_1.getBegin(fft_len);
                    auto tb2 = multi_table_2.getBegin(fft_len);
                    auto tb3 = multi_table_3.getBegin(fft_len);
                    auto it0 = in_out, it1 = in_out + stride, it2 = in_out + stride * 2, it3 = in_out + stride * 3;
                    for (auto end = in_out + stride; it0 < end;
                         it0 += 16, it1 += 16, it2 += 16, it3 += 16,
                              tb1 += 16, tb2 += 16, tb3 += 16)
                    {
                        C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega1, omega2, omega3;
                        c0.load(it0, FromRIRI{}), c1.load(it1, FromRIRI{}), c2.load(it2, FromRIRI{}), c3.load(it3, FromRIRI{});
                        c4.load(it0 + half, FromRIRI{}), c5.load(it1 + half, FromRIRI{}), c6.load(it2 + half, FromRIRI{}), c7.load(it3 + half, FromRIRI{});
                        dif8(c0, c1, c2, c3, c4, c5, c6, c7);
                        omega1.load(tb1), c4 = c4.mul(omega1);
                        omega2.load(tb2), c2 = c2.mul(omega2);
                        omega3.load(tb3), c6 = c6.mul(omega3);
                        omega1 = omega2.square();
                        c1 = c1.mul(omega1);
                        c5 = c5.mul(omega2.mul(omega3));
                        c3 = c3.mul(omega3.square());
                        c7 = c7.mul(omega1.mul(omega3));
                        c0.store(it0), c1.store(it1), c2.store(it2), c3.store(it3);
                        c4.store(it0 + half), c5.store(it1 + half), c6.store(it2 + half), c7.store(it3 + half);
                    }
                    difRecRadix8(in_out, stride);
                    difRecRadix8(it0, stride);
                    difRecRadix8(it1, stride);
                    difRecRadix8(it2, stride);
                    difRecRadix8(in_out + half, stride);
                    difRecRadix8(it0 + half, stride);
                    difRecRadix8(it1 + half, stride);
                    difRecRadix8(it2 + half, stride);
                }

                template <bool TO_RIRI = false, bool TO_INT64 = false>
                static void iditRecRadix8(Float64 in_out[], size_t float_len) noexcept
                {
                    using ToRIRI = std::integral_constant<bool, TO_RIRI>;
                    using ToI64 = std::integral_constant<bool, TO_INT64>;
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= 256)
                    {
                        assert(!TO_RIRI);
                        if (fft_len == 256)
                        {
                            idit256(in_out);
                        }
                        else if (fft_len == 128)
                        {
                            idit128(in_out);
                        }
                        else if (fft_len == 64)
                        {
                            idit64(in_out);
                        }
                        return;
                    }
                    const size_t stride = float_len / 8, half = float_len / 2;
                    auto it0 = in_out, it1 = in_out + stride, it2 = in_out + stride * 2, it3 = in_out + stride * 3;
                    iditRecRadix8(it0, stride);
                    iditRecRadix8(it1, stride);
                    iditRecRadix8(it2, stride);
                    iditRecRadix8(it3, stride);
                    iditRecRadix8(it0 + half, stride);
                    iditRecRadix8(it1 + half, stride);
                    iditRecRadix8(it2 + half, stride);
                    iditRecRadix8(it3 + half, stride);
                    auto tb1 = multi_table_1.getBegin(fft_len);
                    auto tb2 = multi_table_2.getBegin(fft_len);
                    auto tb3 = multi_table_3.getBegin(fft_len);
                    for (auto end = in_out + stride; it0 < end;
                         it0 += 16, it1 += 16, it2 += 16, it3 += 16,
                              tb1 += 16, tb2 += 16, tb3 += 16)
                    {
                        C64X8 c0, c1, c2, c3, c4, c5, c6, c7, omega1, omega2, omega3;
                        c0.load(it0), c1.load(it1), c2.load(it2), c3.load(it3);
                        c4.load(it0 + half), c5.load(it1 + half), c6.load(it2 + half), c7.load(it3 + half);
                        omega1.load(tb1), c4 = c4.mulConj(omega1);
                        omega2.load(tb2), c2 = c2.mulConj(omega2);
                        omega3.load(tb3), c6 = c6.mulConj(omega3);
                        omega1 = omega2.square();
                        c1 = c1.mulConj(omega1);
                        c5 = c5.mulConj(omega2.mul(omega3));
                        c3 = c3.mulConj(omega3.square());
                        c7 = c7.mulConj(omega1.mul(omega3));
                        idit8(c0, c1, c2, c3, c4, c5, c6, c7);
                        c0.transToI64(ToI64{}).store(it0, ToRIRI{}), c1.transToI64(ToI64{}).store(it1, ToRIRI{}), c2.transToI64(ToI64{}).store(it2, ToRIRI{}), c3.transToI64(ToI64{}).store(it3, ToRIRI{});
                        c4.transToI64(ToI64{}).store(it0 + half, ToRIRI{}), c5.transToI64(ToI64{}).store(it1 + half, ToRIRI{}), c6.transToI64(ToI64{}).store(it2 + half, ToRIRI{}), c7.transToI64(ToI64{}).store(it3 + half, ToRIRI{});
                    }
                }
            };

            constexpr size_t FFT512::LOG_SHORT;
            constexpr size_t FFT512::LOG_MAX;
            constexpr size_t FFT512::SHORT_LEN;
            constexpr size_t FFT512::MAX_LEN;

            const TableFix<Float64, 8> FFT512::table_16_1(16, 1, 8);
            const TableFix<Float64, 8> FFT512::table_32_1(32, 1, 8);
            const TableFix<Float64, 8> FFT512::table_32_2(32, 2, 8);
            const TableFix<Float64, 8> FFT512::table_32_3(32, 3, 8);
            const TableFix<Float64, 32> FFT512::table_64_1(64, 1, 8);

            const TableFix<Float64, 32> FFT512::table_128_1(128, 1, 8);
            const TableFix<Float64, 32> FFT512::table_128_3(128, 3, 8);
            const TableFix<Float64, 32> FFT512::table_256_1(256, 1, 8);
            const TableFix<Float64, 32> FFT512::table_256_3(256, 3, 8);
            const TableFix<Float64, 32> FFT512::table_256_5(256, 5, 8);
            const TableFix<Float64, 32> FFT512::table_256_7(256, 7, 8);
            const TableFixMulti<Float64, 9, FFT512::LOG_MAX, 8> FFT512::multi_table_1(1);
            const TableFixMulti<Float64, 9, FFT512::LOG_MAX, 8> FFT512::multi_table_2(2);
            const TableFixMulti<Float64, 9, FFT512::LOG_MAX, 8> FFT512::multi_table_3(3);

            class BinRevTableC64X8HP
            {
            public:
                using F64 = double;
                using C64 = std::complex<F64>;
                using C64X8 = hint_simd::Complex64X8;
                static constexpr int MAX_LOG_LEN = 32, LOG_BLOCK = 3, BLOCK = 1 << LOG_BLOCK;
                static constexpr size_t MAX_LEN = size_t(1) << MAX_LOG_LEN;

                BinRevTableC64X8HP(int log_max_iter_in, int log_fft_len_in)
                    : index(0), pop(0), log_max_iter(log_max_iter_in), log_fft_len(log_fft_len_in)
                {
                    assert(log_max_iter <= log_fft_len);
                    assert(log_fft_len <= MAX_LOG_LEN);
                    const F64 factor = F64(1) / (size_t(1) << (log_fft_len - log_max_iter));
                    for (int i = 0; i < MAX_LOG_LEN; i++)
                    {
                        units[i] = getOmega(size_t(1) << (i + 1), 1, factor);
                    }
                    auto fp = reinterpret_cast<F64 *>(table);
                    fp[0] = 1, fp[BLOCK] = 0;
                    for (int i = 1; i < BLOCK; i++)
                    {
                        C64 omega = getOmega(BLOCK, bitrev(i, LOG_BLOCK), factor);
                        fp[i] = omega.real(), fp[i + BLOCK] = omega.imag();
                    }
                }

                // Only for power of 2
                void reset(size_t i = 0)
                {
                    if (i == 0)
                    {
                        pop = 0, index = i;
                        return;
                    }
                    assert((i & (i - 1)) == 0);
                    assert(i % BLOCK == 0);
                    pop = 1, index = i / BLOCK;
                    int zero = hint_ctz(index);
                    auto fp = reinterpret_cast<F64 *>(&units[zero + 3]);
                    table[1].load1(fp, fp + 1);
                    table[1] = table[1].mul(table[0]);
                }
                C64X8 iterate()
                {
                    C64X8 res = table[pop], unitx;
                    index++;
                    int zero = hint_ctz(index);
                    auto fp = reinterpret_cast<F64 *>(&units[zero + 3]);
                    unitx.load1(fp, fp + 1);
                    pop -= zero;
                    table[pop + 1] = table[pop].mul(unitx);
                    pop++;
                    return res;
                }

                static C64 getOmega(size_t n, size_t index, F64 factor = 1)
                {
                    F64 theta = -HINT_2PI * index / n;
                    return std::polar<F64>(1, theta * factor);
                }

            private:
                C64 units[MAX_LOG_LEN]{};
                C64X8 table[MAX_LOG_LEN]{};
                size_t index;
                int pop;
                int log_max_iter, log_fft_len;
            };
            template <size_t RI_DIFF = 1, typename FloatTy>
            inline void dot_rfft(FloatTy *inout0, FloatTy *inout1, const FloatTy *in0, const FloatTy *in1,
                                 const std::complex<FloatTy> &omega0, const FloatTy factor = 1)
            {
                using Complex = std::complex<FloatTy>;
                auto mul1 = [](Complex c0, Complex c1)
                {
                    return Complex(c0.imag() * c1.real() + c0.real() * c1.imag(),
                                   c0.imag() * c1.imag() - c0.real() * c1.real());
                };
                auto mul2 = [](Complex c0, Complex c1)
                {
                    return Complex(c0.real() * c1.imag() - c0.imag() * c1.real(),
                                   c0.real() * c1.real() + c0.imag() * c1.imag());
                };
                auto compute2 = [&omega0](Complex in0, Complex in1, Complex &out0, Complex &out1, auto Func)
                {
                    in1 = std::conj(in1);
                    transform2(in0, in1);
                    in1 = Func(in1, omega0);
                    out0 = in0 + in1;
                    out1 = std::conj(in0 - in1);
                };
                Complex c0, c1;
                {
                    Complex x0, x1, x2, x3;
                    c0.real(inout0[0]), c0.imag(inout0[RI_DIFF]), c1.real(inout1[0]), c1.imag(inout1[RI_DIFF]);
                    compute2(c0, c1, x0, x1, mul1);
                    c0.real(in0[0]), c0.imag(in0[RI_DIFF]), c1.real(in1[0]), c1.imag(in1[RI_DIFF]);
                    compute2(c0, c1, x2, x3, mul1);
                    x0 *= x2 * factor;
                    x1 *= x3 * factor;
                    compute2(x0, x1, c0, c1, mul2);
                }
                inout0[0] = c0.real(), inout0[RI_DIFF] = c0.imag();
                inout1[0] = c1.real(), inout1[RI_DIFF] = c1.imag();
            }
            inline void dot_rfftX8(F64 *inout0, F64 *inout1, const F64 *in0, const F64 *in1, const C64X8 &omega0, const F64X8 &inv)
            {
                auto mul1 = [](C64X8 c0, C64X8 c1)
                {
                    return C64X8(F64X8::fmadd(c0.imag, c1.real, c0.real * c1.imag),
                                 F64X8::fmsub(c0.imag, c1.imag, c0.real * c1.real));
                };
                auto mul2 = [](C64X8 c0, C64X8 c1)
                {
                    return C64X8(F64X8::fmsub(c0.real, c1.imag, c0.imag * c1.real),
                                 F64X8::fmadd(c0.real, c1.real, c0.imag * c1.imag));
                };
                auto compute2 = [&omega0](C64X8 c0, C64X8 c1, C64X8 &out0, C64X8 &out1, auto Func)
                {
                    C64X8 t0(c0.real + c1.real, c0.imag - c1.imag), t1(c0.real - c1.real, c0.imag + c1.imag);
                    t1 = Func(t1, omega0);
                    out0 = t0 + t1;
                    out1.real = t0.real - t1.real;
                    out1.imag = t1.imag - t0.imag;
                };
                C64X8 c0, c1;
                {
                    C64X8 x0, x1, x2, x3;
                    c0.load(inout0), c1.load(inout1);
                    compute2(c0, c1.reverse(), x0, x1, mul1);

                    c0.load(in0), c1.load(in1);
                    compute2(c0, c1.reverse(), x2, x3, mul1);
                    c0 = x0.mul(x2) * inv;
                    c1 = x1.mul(x3) * inv;
                    compute2(c0, c1, c0, c1, mul2);
                }
                c0.store(inout0), c1.reverse().store(inout1);
            }

            // inv = 1 / float_len in AVX function
            template <size_t RI_DIFF = 1, typename Float>
            inline void real_dot_binrev32(Float in_out[], const Float in[], size_t float_len, Float inv = -1)
            {
                constexpr size_t MAX_LEN = 32;
                constexpr int LOG_LEN = hint_log2(MAX_LEN);
                static_assert(is_2pow(RI_DIFF));
                static_assert(RI_DIFF <= 8);
                assert(is_2pow(float_len));
                assert(float_len <= MAX_LEN);
                if (float_len < 2)
                {
                    return;
                }
                assert(float_len >= RI_DIFF * 2);
                auto idx_trans = [](size_t idx)
                {
                    return (idx / RI_DIFF) * RI_DIFF * 2 + idx % RI_DIFF;
                };
                auto get_omega = [](size_t idx, size_t rank)
                {
                    return std::polar<Float>(1, -HINT_PI * Float(idx) / rank);
                };
                using Complex = std::complex<Float>;
                static const Complex table[]{
                    get_omega(bitrev(4, LOG_LEN), MAX_LEN),
                    get_omega(bitrev(5, LOG_LEN), MAX_LEN),
                    get_omega(bitrev(8, LOG_LEN), MAX_LEN),
                    get_omega(bitrev(9, LOG_LEN), MAX_LEN),
                    get_omega(bitrev(10, LOG_LEN), MAX_LEN),
                    get_omega(bitrev(11, LOG_LEN), MAX_LEN),
                };
                inv = inv < 0 ? Float(2) / float_len : inv * Float(2);
                auto r0 = in_out[0], i0 = in_out[RI_DIFF], r1 = in[0], i1 = in[RI_DIFF];
                transform2(r0, i0);
                transform2(r1, i1);
                r0 *= r1, i0 *= i1;
                transform2(r0, i0);
                in_out[0] = r0 * 0.5 * inv, in_out[RI_DIFF] = i0 * 0.5 * inv;
                if (float_len >= 4)
                {
                    Complex temp(in_out[idx_trans(1)], in_out[idx_trans(1) + RI_DIFF]);
                    temp *= Complex(in[idx_trans(1)], in[idx_trans(1) + RI_DIFF]) * inv;
                    in_out[idx_trans(1)] = temp.real(), in_out[idx_trans(1) + RI_DIFF] = temp.imag();
                }
                if (float_len >= 8)
                {
                    inv *= Float(0.125);
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(2)], &in_out[idx_trans(3)],
                                      &in[idx_trans(2)], &in[idx_trans(3)], Complex(COS_PI_8, -COS_PI_8), inv);
                }

                if (float_len >= 16)
                {
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(4)], &in_out[idx_trans(7)],
                                      &in[idx_trans(4)], &in[idx_trans(7)], table[0], inv);
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(5)], &in_out[idx_trans(6)],
                                      &in[idx_trans(5)], &in[idx_trans(6)], table[1], inv);
                }
                if (float_len >= 32)
                {
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(8)], &in_out[idx_trans(15)],
                                      &in[idx_trans(8)], &in[idx_trans(15)], table[2], inv);
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(9)], &in_out[idx_trans(14)],
                                      &in[idx_trans(9)], &in[idx_trans(14)], table[3], inv);
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(10)], &in_out[idx_trans(13)],
                                      &in[idx_trans(10)], &in[idx_trans(13)], table[4], inv);
                    dot_rfft<RI_DIFF>(&in_out[idx_trans(11)], &in_out[idx_trans(12)],
                                      &in[idx_trans(11)], &in[idx_trans(12)], table[5], inv);
                }
            }

            inline void real_dot_binrev8(Float64 in_out[], Float64 in[], size_t float_len)
            {
                using Complex = std::complex<Float64>;
                Float64 inv = 1.0 / float_len;
                real_dot_binrev32<8>(in_out, in, 32, inv);
                inv = 0.25 / float_len;
                const F64X8 inv8 = F64X8(inv);
                BinRevTableC64X8HP table(31, 32);
                for (size_t begin = 32; begin < float_len; begin *= 2)
                {
                    table.reset(begin / 2);
                    auto it0 = in_out + begin, it1 = it0 + begin - 16;
                    auto it2 = in + begin, it3 = it2 + begin - 16;
                    for (; it0 < it1; it0 += 16, it1 -= 16, it2 += 16, it3 -= 16)
                    {
                        dot_rfftX8(it0, it1, it2, it3, table.iterate(), inv8);
                    }
                }
            }
            template <bool TO_INT>
            inline void real_conv_avx512(F64 *in_out1, F64 *in2, size_t float_len)
            {
                assert(is_2pow(float_len));
                FFT512::difRecRadix8<true>(in_out1, float_len);
                FFT512::difRecRadix8<true>(in2, float_len);
                real_dot_binrev8(in_out1, in2, float_len);
                FFT512::iditRecRadix8<true, TO_INT>(in_out1, float_len);
            }
        }
    }
}
#endif // FFT_AVX_HPP

void permuteRIRIToRRII(const double *in, double *out, int len)
{
    hint_simd::Complex64X8 c;
    for (size_t i = 0; i < len; i += 16)
    {
        c.loadu(in + i);
        c = c.toRRIIPermu();
        c.storeu(out + i);
    }
}

void permuteRRIIToRIRI(const double *in, double *out, int len)
{
    hint_simd::Complex64X8 c;
    for (size_t i = 0; i < len; i += 16)
    {
        c.loadu(in + i);
        c = c.toRIRIPermu();
        c.storeu(out + i);
    }
}
using Complex = std::complex<double>;

inline void idit(Complex in_out[], size_t len, bool norm = false)
{
    for (size_t rank = 2; rank <= len; rank *= 2)
    {
        size_t stride = rank / 2;
        auto it0 = in_out, it1 = it0 + stride;
        for (size_t begin = 0; begin < len; begin += rank, it0 += rank, it1 += rank)
        {
            for (size_t i = 0; i < stride; i++)
            {
                Complex x = it0[i];
                Complex y = it1[i] * std::polar(1.0, hint::HINT_2PI * i / rank);
                it0[i] = x + y;
                it1[i] = x - y;
            }
        }
    }
    if (norm)
    {
        const double len_inv = 1.0 / len;
        for (size_t i = 0; i < len; i++)
        {
            in_out[i] *= len_inv;
        }
    }
}

inline void dif(Complex in_out[], size_t len)
{
    for (size_t rank = len; rank >= 2; rank /= 2)
    {
        size_t stride = rank / 2;
        auto it0 = in_out, it1 = it0 + stride;
        for (size_t begin = 0; begin < len; begin += rank, it0 += rank, it1 += rank)
        {
            for (size_t i = 0; i < stride; i++)
            {
                Complex x = it0[i];
                Complex y = it1[i];
                it0[i] = x + y;
                it1[i] = (x - y) * std::polar(1.0, -hint::HINT_2PI * i / rank);
            }
        }
    }
}
#include <chrono>
#include <vector>
#include "../bind_cpu.hpp"

template <typename T>
std::vector<T> poly_multiply(const std::vector<T> &in1, const std::vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size();
    size_t conv_len = len1 + len2;
    size_t float_len = hint::int_ceil2(conv_len);
    size_t fft_len = float_len / 2;
    auto p1 = (double *)_mm_malloc(float_len * sizeof(double), 64);
    auto p2 = (double *)_mm_malloc(float_len * sizeof(double), 64);
    std::copy(in1.begin(), in1.end(), p1);
    std::copy(in2.begin(), in2.end(), p2);
    std::fill(p1 + len1, p1 + float_len, 0);
    std::fill(p2 + len2, p2 + float_len, 0);
    hint::transform::fft::real_conv_avx512<true>(p1, p2, float_len);
    auto i64_p = reinterpret_cast<uint64_t *>(p1);
    std::vector<T> res(conv_len);
    for (size_t i = 0; i < conv_len; i++)
    {
        res[i] = i64_p[i];
    }
    _mm_free(p1);
    _mm_free(p2);
    return res;
}

template <typename T>
void result_test(const std::vector<T> &res, uint64_t ele1, uint64_t ele2)
{
    size_t len = res.size();
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele1 * ele2;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele1 * ele2;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    std::cout << "success\n";
}
void perf_conv()
{
    int n = 13;
    std::cin >> n;
    size_t len = size_t(1) << n; // 变换长度
    std::cout << "conv len:" << len << "\n";
    int MAX = 1 << 8;
    uint64_t ele1 = MAX - 1, ele2 = MAX - 1;
    std::vector<uint64_t> in1(len / 2, 0);
    std::vector<uint64_t> in2(len / 2, 0); // 计算两个长度为len/2，每个元素为ele的卷积
    for (size_t i = 0; i < len / 2; i++)
    {
        in1[i] = ele1;
        in2[i] = ele2;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> res = poly_multiply(in1, in2);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (auto i : res)
    {
        // std::cout << i << " ";
    }
    std::cout << "\n";
    result_test<uint64_t>(res, ele1, ele2); // 结果校验
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}

void test_dif()
{
    bind_cpu(0);
    constexpr size_t MAX = 1 << 23;
    alignas(64) static double arr[MAX]{}, arr2[MAX]{}, arr3[MAX]{}, arr4[MAX]{};
    size_t len = 1 << 23;
    for (int i = 0; i < len; i += 2)
    {
        arr[i] = (i / 2) % 10000;
        arr[i + 1] = i % 10000;
        arr2[i] = arr[i];
        arr2[i + 1] = arr[i + 1];
        arr3[i] = arr4[i] = rand();
    }
    permuteRIRIToRRII(arr2, arr2, len);
    permuteRIRIToRRII(arr4, arr4, len);
    hint_simd::Complex64X8 c0, c1, c2, c3, c4, c5, c6, c7;
    auto t1 = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < 1000; i++)
    {
        // hint::transform::fft::real_dot_binrev8(arr2, arr4, len);
        // hint::transform::fft::real_dot_binrev<8>(arr2, arr4, len);
        hint::transform::fft::FFT512::iditRecRadix8<false>(arr2, len);
        // hint::transform::fft::FFT512::dif64(arr2);
        // hint::transform::fft::FFT512::dif4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
    }
    // c0.store(arr2);
    // c1.store(arr2 + 16);
    // c2.store(arr2 + 32);
    // c3.store(arr2 + 48);
    // c4.store(arr2 + 64);
    // c5.store(arr2 + 80);
    // c6.store(arr2 + 96);
    // c7.store(arr2 + 112);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    permuteRRIIToRIRI(arr2, arr2, len);
    auto p = reinterpret_cast<Complex *>(arr);
    // for (size_t i = 0; i < 32; i += 4)
    {
        idit(p, len / 2);
        // dif(p , 32);
        // dif(p + 32, 32);
    }
    for (size_t i = 0; i < len; i++)
    {
        if (std::abs(arr[i] - arr2[i]) > 1e-3)
        {
            std::cout << i << "\t" << arr[i] << " " << arr2[i] << std::endl;
            // break;
        }
    }
}

int main()
{
    perf_conv();
    // test_dif();
}