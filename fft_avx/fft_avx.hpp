#ifndef FFT_AVX_HPP
#define FFT_AVX_HPP

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
            using F64X4 = hint_simd::Float64X4;
            using C64X4 = hint_simd::Complex64X4;
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
                TableFixMulti(Float factor_in, size_t stride_in = 4, bool preload = true)
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
                    assert(stride == 4); // TODO: support other stride
                    assert(log_rank <= LOG_END);
                    for (int log_len = log_current + 1; log_len <= log_rank; log_len++)
                    {
                        size_t len = size_t(1) << log_len, cpx_len = len / DIV;
                        Float theta = -HINT_2PI * factor / len;
                        auto it = getBeginLog(log_len), it_last = getBeginLog(log_len - 1);
                        C64X4 unit(std::cos(theta), std::sin(theta));
                        for (auto end = it + cpx_len * 2; it < end; it += 16, it_last += 8)
                        {
                            C64X4 omega0, omega1;
                            omega0.load(it_last);
                            omega1 = omega0.mul(unit);
                            transpose64_2X4(omega0.real, omega1.real);
                            transpose64_2X4(omega0.imag, omega1.imag);
                            omega0.store(it), omega1.store(it + 8);
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
                template <typename Float>
                static void dif4(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    difSplit(r0, i0, r1, i1, r2, i2, r3, i3);
                    transform2(r0, r1);
                    transform2(i0, i1);
                }
                template <typename Float>
                static void idit4(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r0, r1);
                    transform2(i0, i1);
                    iditSplit(r0, i0, r1, i1, r2, i2, r3, i3);
                }
                template <typename Float>
                static void difSplit(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r0, r2);
                    transform2(i0, i2);
                    transform2(r1, r3);
                    transform2(i1, i3);

                    transform2(r2, i3);
                    transform2(i2, r3, r3, i2);
                    std::swap(i3, r3);
                }
                template <typename Float>
                static void iditSplit(Float &r0, Float &i0, Float &r1, Float &i1, Float &r2, Float &i2, Float &r3, Float &i3)
                {
                    transform2(r2, r3);
                    transform2(i2, i3);

                    transform2(r0, r2);
                    transform2(i0, i2);
                    transform2(r1, i3, i3, r1);
                    transform2(i1, r3);
                    std::swap(i3, r3);
                }
            };

            struct FFTAVX : public FFT
            {
                static constexpr size_t LOG_SHORT = 10;
                static constexpr size_t LOG_MAX = 27;
                static constexpr size_t SHORT_LEN = size_t(1) << LOG_SHORT;
                static constexpr size_t MAX_LEN = size_t(1) << LOG_MAX;
                static const TableFix<Float64, 4> table_8;
                static const TableFix<Float64, 4> table_16_1;
                static const TableFix<Float64, 4> table_16_3;
                static const TableFix<Float64, 8> table_32_1;
                static const TableFix<Float64, 8> table_32_3;
                static const TableFixMulti<Float64, 6, LOG_SHORT, 4> multi_table_3;
                static const TableFixMulti<Float64, 6, LOG_SHORT, 4> multi_table_2;
                static TableFixMulti<Float64, 6, LOG_MAX, 4> multi_table_1;

                static constexpr const Float64 *it8 = &table_8[0];
                static constexpr const Float64 *it16_1 = &table_16_1[0];
                static constexpr const Float64 *it16_3 = &table_16_3[0];
                static constexpr const Float64 *it32_1 = &table_32_1[0];
                static constexpr const Float64 *it32_3 = &table_32_3[0];

                static void dif4x4(F64X4 &r0, F64X4 &i0, F64X4 &r1, F64X4 &i1, F64X4 &r2, F64X4 &i2, F64X4 &r3, F64X4 &i3)
                {
                    transpose64_4X4(r0, r1, r2, r3);
                    transpose64_4X4(i0, i1, i2, i3);

                    dif4(r0, i0, r1, i1, r2, i2, r3, i3);

                    transpose64_4X4(r0, r1, r2, r3);
                    transpose64_4X4(i0, i1, i2, i3);
                }
                static void idit4x4(F64X4 &r0, F64X4 &i0, F64X4 &r1, F64X4 &i1, F64X4 &r2, F64X4 &i2, F64X4 &r3, F64X4 &i3)
                {
                    transpose64_4X4(r0, r1, r2, r3);
                    transpose64_4X4(i0, i1, i2, i3);

                    idit4(r0, i0, r1, i1, r2, i2, r3, i3);

                    transpose64_4X4(r0, r1, r2, r3);
                    transpose64_4X4(i0, i1, i2, i3);
                }
                static void dif8x2(C64X4 &c0, C64X4 &c1, C64X4 &c2, C64X4 &c3, const C64X4 &omega)
                {
                    transform2(c0, c1);
                    transform2(c2, c3);
                    c1 = c1.mul(omega);
                    c3 = c3.mul(omega);
                    FFTAVX::dif4x4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                }
                static void idit8x2(C64X4 &c0, C64X4 &c1, C64X4 &c2, C64X4 &c3, const C64X4 &omega)
                {
                    idit4x4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    c1 = c1.mulConj(omega);
                    c3 = c3.mulConj(omega);
                    transform2(c0, c1);
                    transform2(c2, c3);
                }
                static void dif8x2(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    c0.load(in_out), c1.load(in_out + 8), c2.load(in_out + 16), c3.load(in_out + 24), omega.load(it8);
                    dif8x2(c0, c1, c2, c3, omega);
                    c0.store(in_out), c1.store(in_out + 8), c2.store(in_out + 16), c3.store(in_out + 24);
                }
                static void idit8x2(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    c0.load(in_out), c1.load(in_out + 8), c2.load(in_out + 16), c3.load(in_out + 24), omega.load(it8);
                    idit8x2(c0, c1, c2, c3, omega);
                    c0.store(in_out), c1.store(in_out + 8), c2.store(in_out + 16), c3.store(in_out + 24);
                }
                static void dif16(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    c0.load(in_out), c1.load(in_out + 8), c2.load(in_out + 16), c3.load(in_out + 24);
                    dif4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    omega.load(it8), c1 = c1.mul(omega);
                    omega.load(it16_1), c2 = c2.mul(omega);
                    omega.load(it16_3), c3 = c3.mul(omega);
                    dif4x4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    c0.store(in_out), c1.store(in_out + 8), c2.store(in_out + 16), c3.store(in_out + 24);
                }
                static void idit16(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    c0.load(in_out), c1.load(in_out + 8), c2.load(in_out + 16), c3.load(in_out + 24);
                    idit4x4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    omega.load(it8), c1 = c1.mulConj(omega);
                    omega.load(it16_1), c2 = c2.mulConj(omega);
                    omega.load(it16_3), c3 = c3.mulConj(omega);
                    idit4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    c0.store(in_out), c1.store(in_out + 8), c2.store(in_out + 16), c3.store(in_out + 24);
                }
                static void dif32(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    c0.load(in_out), c1.load(in_out + 16), c2.load(in_out + 32), c3.load(in_out + 48);
                    difSplit(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    omega.load(it32_1), c2 = c2.mul(omega);
                    omega.load(it32_3), c3 = c3.mul(omega);
                    c0.store(in_out), c1.store(in_out + 16), c2.store(in_out + 32), c3.store(in_out + 48);

                    c0.load(in_out + 8), c1.load(in_out + 24), c2.load(in_out + 40), c3.load(in_out + 56); // 1,3,5,7
                    difSplit(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    omega.load(it32_1 + 8), c2 = c2.mul(omega);
                    omega.load(it32_3 + 8), c3 = c3.mul(omega);
                    c0.store(in_out + 8), c1.store(in_out + 24);
                    c0.load(in_out + 32), c1.load(in_out + 48), omega.load(it8); // 4,6
                    dif8x2(c0, c2, c1, c3, omega);
                    c0.store(in_out + 32), c2.store(in_out + 40), c1.store(in_out + 48), c3.store(in_out + 56);
                    dif16(in_out);
                }
                static void idit32(Float64 in_out[])
                {
                    C64X4 c0, c1, c2, c3, omega;
                    idit16(in_out);
                    c0.load(in_out + 32), c1.load(in_out + 40), c2.load(in_out + 48), c3.load(in_out + 56), omega.load(it8); // 4,5,6,7
                    idit8x2(c0, c1, c2, c3, omega);
                    c1.store(in_out + 40), c3.store(in_out + 56);

                    c1.load(in_out), c3.load(in_out + 16);
                    omega.load(it32_1), c0 = c0.mulConj(omega);
                    omega.load(it32_3), c2 = c2.mulConj(omega);
                    iditSplit(c1.real, c1.imag, c3.real, c3.imag, c0.real, c0.imag, c2.real, c2.imag);
                    c1.store(in_out), c3.store(in_out + 16), c0.store(in_out + 32), c2.store(in_out + 48);

                    c0.load(in_out + 8), c1.load(in_out + 24), c2.load(in_out + 40), c3.load(in_out + 56);
                    omega.load(it32_1 + 8), c2 = c2.mulConj(omega);
                    omega.load(it32_3 + 8), c3 = c3.mulConj(omega);
                    iditSplit(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                    c0.store(in_out + 8), c1.store(in_out + 24), c2.store(in_out + 40), c3.store(in_out + 56);
                }
                static void dif16(Float64 in_out[], size_t float_len)
                {
                    assert(float_len >= 32);
                    for (auto end = in_out + float_len; in_out < end; in_out += 32)
                    {
                        dif16(in_out);
                    }
                }
                static void idit16(Float64 in_out[], size_t float_len)
                {
                    assert(float_len >= 32);
                    for (auto end = in_out + float_len; in_out < end; in_out += 32)
                    {
                        idit16(in_out);
                    }
                }
                static void dif32(Float64 in_out[], size_t float_len)
                {
                    assert(float_len >= 64);
                    for (auto end = in_out + float_len; in_out < end; in_out += 64)
                    {
                        dif32(in_out);
                    }
                }
                static void idit32(Float64 in_out[], size_t float_len)
                {
                    assert(float_len >= 64);
                    for (auto end = in_out + float_len; in_out < end; in_out += 64)
                    {
                        idit32(in_out);
                    }
                }
                static void difIter(Float64 in_out[], size_t float_len)
                {
                    size_t fft_len = float_len / 2;
                    multi_table_1.initBottomUp(hint_log2(fft_len));
                    assert(fft_len <= SHORT_LEN);
                    for (size_t rank = fft_len; rank >= 64; rank /= 4)
                    {
                        const size_t stride1 = rank / 2, stride2 = stride1 * 2, stride3 = stride1 * 3;
                        for (auto begin = in_out, end = in_out + float_len; begin < end; begin += rank * 2)
                        {
                            auto table1 = multi_table_1.getBegin(rank);
                            auto table2 = multi_table_2.getBegin(rank), table3 = multi_table_3.getBegin(rank);
                            auto it0 = begin, it1 = begin + stride1, it2 = begin + stride2, it3 = begin + stride3;
                            for (; it0 < begin + stride1; it0 += 8, it1 += 8, it2 += 8, it3 += 8, table1 += 8, table2 += 8, table3 += 8)
                            {
                                C64X4 c0, c1, c2, c3, omega;
                                c0.load(it0), c1.load(it1), c2.load(it2), c3.load(it3);
                                dif4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                                omega.load(table2), c1 = c1.mul(omega);
                                omega.load(table1), c2 = c2.mul(omega);
                                omega.load(table3), c3 = c3.mul(omega);
                                c0.store(it0), c1.store(it1), c2.store(it2), c3.store(it3);
                            }
                        }
                    }
                    if (hint_log2(fft_len) % 2 == 0)
                    {
                        dif16(in_out, float_len);
                    }
                    else
                    {
                        dif32(in_out, float_len);
                    }
                }
                static void iditIter(Float64 in_out[], size_t float_len)
                {
                    size_t fft_len = float_len / 2;
                    multi_table_1.initBottomUp(hint_log2(fft_len));
                    assert(fft_len <= SHORT_LEN);
                    size_t rank = 0;
                    if (hint_log2(fft_len) % 2 == 0)
                    {
                        idit16(in_out, float_len);
                        rank = 64;
                    }
                    else
                    {
                        idit32(in_out, float_len);
                        rank = 128;
                    }
                    for (; rank <= fft_len; rank *= 4)
                    {
                        const size_t stride1 = rank / 2, stride2 = stride1 * 2, stride3 = stride1 * 3;
                        for (auto begin = in_out, end = in_out + float_len; begin < end; begin += rank * 2)
                        {
                            auto table1 = multi_table_1.getBegin(rank);
                            auto table2 = multi_table_2.getBegin(rank), table3 = multi_table_3.getBegin(rank);
                            auto it0 = begin, it1 = begin + stride1, it2 = begin + stride2, it3 = begin + stride3;
                            for (; it0 < begin + stride1; it0 += 8, it1 += 8, it2 += 8, it3 += 8, table1 += 8, table2 += 8, table3 += 8)
                            {
                                C64X4 c0, c1, c2, c3, omega;
                                c0.load(it0), c1.load(it1), c2.load(it2), c3.load(it3);
                                omega.load(table2), c1 = c1.mulConj(omega);
                                omega.load(table1), c2 = c2.mulConj(omega);
                                omega.load(table3), c3 = c3.mulConj(omega);
                                idit4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                                c0.store(it0), c1.store(it1), c2.store(it2), c3.store(it3);
                            }
                        }
                    }
                }

                template <bool FROM_RIRI_PERM = false>
                static void difRecRadix4(Float64 in_out[], size_t float_len)
                {
                    using FromRIRI = std::integral_constant<bool, FROM_RIRI_PERM>;
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= SHORT_LEN)
                    {
                        difIter(in_out, float_len);
                        assert(!FROM_RIRI_PERM);
                        return;
                    }
                    multi_table_1.initBottomUp(hint_log2(fft_len));
                    const size_t stride1 = float_len / 4, stride2 = stride1 * 2, stride3 = stride1 * 3;
                    auto table1 = multi_table_1.getBegin(fft_len);
                    for (auto end = in_out + stride1, it = in_out; it < end; it += 8, table1 += 8)
                    {
                        C64X4 c0, c1, c2, c3, omega, omega2;
                        c0.load(it, FromRIRI{}), c1.load(it + stride1, FromRIRI{}), c2.load(it + stride2, FromRIRI{}), c3.load(it + stride3, FromRIRI{});
                        c2.load(it + stride2, FromRIRI{}), c3.load(it + stride3, FromRIRI{});
                        dif4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                        omega.load(table1), c2 = c2.mul(omega);
                        omega2 = omega.square(), c1 = c1.mul(omega2);
                        c3 = c3.mul(omega2.mul(omega));
                        c0.store(it), c1.store(it + stride1), c2.store(it + stride2), c3.store(it + stride3);
                    }
                    difRecRadix4(in_out, stride1);
                    difRecRadix4(in_out + stride1, stride1);
                    difRecRadix4(in_out + stride2, stride1);
                    difRecRadix4(in_out + stride3, stride1);
                }
                template <bool TO_RIRI_PERM = false, bool TO_INT64 = false>
                static void iditRecRadix4(Float64 in_out[], size_t float_len)
                {
                    const size_t fft_len = float_len / 2;
                    assert(fft_len <= MAX_LEN);
                    if (fft_len <= SHORT_LEN)
                    {
                        iditIter(in_out, float_len);
                        assert(!TO_RIRI_PERM);
                        assert(!TO_INT64);
                        return;
                    }
                    multi_table_1.initBottomUp(hint_log2(fft_len));
                    using ToRIRI = std::integral_constant<bool, TO_RIRI_PERM>;
                    using ToI64 = std::integral_constant<bool, TO_INT64>;
                    const size_t stride1 = float_len / 4, stride2 = stride1 * 2, stride3 = stride1 * 3;
                    iditRecRadix4(in_out, stride1);
                    iditRecRadix4(in_out + stride1, stride1);
                    iditRecRadix4(in_out + stride2, stride1);
                    iditRecRadix4(in_out + stride3, stride1);
                    auto table1 = multi_table_1.getBegin(fft_len);
                    for (auto end = in_out + stride1, it = in_out; it < end; it += 8, table1 += 8)
                    {
                        C64X4 c0, c1, c2, c3, omega, omega2;
                        c0.load(it), c1.load(it + stride1), c2.load(it + stride2), c3.load(it + stride3);
                        omega.load(table1), c2 = c2.mulConj(omega);
                        omega2 = omega.square(), c1 = c1.mulConj(omega2);
                        c3 = c3.mulConj(omega2.mul(omega));
                        idit4(c0.real, c0.imag, c1.real, c1.imag, c2.real, c2.imag, c3.real, c3.imag);
                        c0 = c0.transToI64(ToI64{}), c1 = c1.transToI64(ToI64{}), c2 = c2.transToI64(ToI64{}), c3 = c3.transToI64(ToI64{});
                        c0.store(it, ToRIRI{}), c1.store(it + stride1, ToRIRI{}), c2.store(it + stride2, ToRIRI{}), c3.store(it + stride3, ToRIRI{});
                    }
                }
            };
            constexpr size_t FFTAVX::LOG_SHORT;
            constexpr size_t FFTAVX::LOG_MAX;
            constexpr size_t FFTAVX::SHORT_LEN;
            constexpr size_t FFTAVX::MAX_LEN;

            const TableFix<Float64, 4> FFTAVX::table_8(8, 1, 4);
            const TableFix<Float64, 4> FFTAVX::table_16_1(16, 1, 4);
            const TableFix<Float64, 4> FFTAVX::table_16_3(16, 3, 4);
            const TableFix<Float64, 8> FFTAVX::table_32_1(32, 1, 4);
            const TableFix<Float64, 8> FFTAVX::table_32_3(32, 3, 4);
            const TableFixMulti<Float64, 6, FFTAVX::LOG_SHORT, 4> FFTAVX::multi_table_3(3);
            const TableFixMulti<Float64, 6, FFTAVX::LOG_SHORT, 4> FFTAVX::multi_table_2(2);
            TableFixMulti<Float64, 6, FFTAVX::LOG_MAX, 4> FFTAVX::multi_table_1(1, 4, false);

            constexpr const Float64 *FFTAVX::it8;
            constexpr const Float64 *FFTAVX::it16_1;
            constexpr const Float64 *FFTAVX::it16_3;
            constexpr const Float64 *FFTAVX::it32_1;
            constexpr const Float64 *FFTAVX::it32_3;

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

            class BinRevTableC64X4HP
            {
            public:
                using F64 = double;
                using C64 = std::complex<F64>;
                using C64X4 = hint_simd::Complex64X4;
                static constexpr int MAX_LOG_LEN = 32, LOG_BLOCK = 2, BLOCK = 1 << LOG_BLOCK;
                static constexpr size_t MAX_LEN = size_t(1) << MAX_LOG_LEN;

                BinRevTableC64X4HP(int log_max_iter_in, int log_fft_len_in)
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
                    auto fp = reinterpret_cast<F64 *>(&units[zero + 2]);
                    table[1].load1(fp, fp + 1);
                    table[1] = table[1].mul(table[0]);
                }
                C64X4 iterate()
                {
                    C64X4 res = table[pop], unit4;
                    index++;
                    int zero = hint_ctz(index);
                    auto fp = reinterpret_cast<F64 *>(&units[zero + 2]);
                    unit4.load1(fp, fp + 1);
                    pop -= zero;
                    table[pop + 1] = table[pop].mul(unit4);
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
                C64X4 table[MAX_LOG_LEN]{};
                size_t index;
                int pop;
                int log_max_iter, log_fft_len;
            };

            template <typename FloatTy>
            class BinRevTableComplexIterHP
            {
            public:
                using Complex = std::complex<FloatTy>;
                static constexpr int MAX_LOG_LEN = 32;
                static constexpr size_t MAX_LEN = size_t(1) << MAX_LOG_LEN;

                BinRevTableComplexIterHP(int log_max_iter_in, int log_fft_len_in)
                    : index(0), pop(0), log_max_iter(log_max_iter_in), log_fft_len(log_fft_len_in)
                {
                    assert(log_max_iter <= log_fft_len);
                    assert(log_fft_len <= MAX_LOG_LEN);
                    const FloatTy factor = FloatTy(1) / (size_t(1) << (log_fft_len - log_max_iter));
                    for (int i = 0; i < MAX_LOG_LEN; i++)
                    {
                        units[i] = getOmega(size_t(1) << (i + 1), 1, factor);
                    }
                    table[0] = Complex(1, 0);
                }
                void reset(size_t i = 0)
                {
                    index = i;
                    if (i == 0)
                    {
                        pop = 0;
                        return;
                    }
                    pop = hint_popcnt(i);
                    const size_t len = size_t(1) << log_fft_len;
                    for (int p = pop; p > 0; p--)
                    {
                        table[p] = getOmega(len, bitrev(i, log_max_iter));
                        i &= (i - 1);
                    }
                }
                Complex iterate()
                {
                    Complex res = table[pop];
                    index++;
                    int zero = hint_ctz(index);
                    pop -= zero;
                    table[pop + 1] = table[pop] * units[zero];
                    pop++;
                    return res;
                }

                static Complex getOmega(size_t n, size_t index, FloatTy factor = 1)
                {
                    FloatTy theta = -HINT_2PI * index / n;
                    return std::polar<FloatTy>(1, theta * factor);
                }

            private:
                Complex units[MAX_LOG_LEN]{};
                Complex table[MAX_LOG_LEN]{};
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
            inline void dot_rfftX4(F64 *inout0, F64 *inout1, const F64 *in0, const F64 *in1, const C64X4 &omega0, const F64X4 &inv)
            {
                auto mul1 = [](C64X4 c0, C64X4 c1)
                {
                    return C64X4(F64X4::fmadd(c0.imag, c1.real, c0.real * c1.imag),
                                 F64X4::fmsub(c0.imag, c1.imag, c0.real * c1.real));
                };
                auto mul2 = [](C64X4 c0, C64X4 c1)
                {
                    return C64X4(F64X4::fmsub(c0.real, c1.imag, c0.imag * c1.real),
                                 F64X4::fmadd(c0.real, c1.real, c0.imag * c1.imag));
                };
                auto compute2 = [&omega0](C64X4 c0, C64X4 c1, C64X4 &out0, C64X4 &out1, auto Func)
                {
                    C64X4 t0(c0.real + c1.real, c0.imag - c1.imag), t1(c0.real - c1.real, c0.imag + c1.imag);
                    t1 = Func(t1, omega0);
                    out0 = t0 + t1;
                    out1.real = t0.real - t1.real;
                    out1.imag = t1.imag - t0.imag;
                };
                C64X4 c0, c1;
                {
                    C64X4 x0, x1, x2, x3;
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
            inline void real_dot_binrev(Float in_out[], Float in[], size_t float_len, Float inv = -1)
            {
                static_assert(is_2pow(RI_DIFF));
                assert(is_2pow(float_len));
                assert(float_len >= RI_DIFF * 2);
                auto transToRIRI = [](Float arr[], size_t len)
                {
                    if (RI_DIFF == 1)
                    {
                        return;
                    }
                    Float temp[RI_DIFF * 2]{};
                    for (auto it = arr, end = arr + len; it < end; it += RI_DIFF * 2)
                    {
                        std::copy(it, it + RI_DIFF * 2, temp);
                        for (size_t i = 0; i < RI_DIFF; i++)
                        {
                            it[i * 2] = temp[i];
                            it[i * 2 + 1] = temp[i + RI_DIFF];
                        }
                    }
                };
                auto transToRRII = [](Float arr[], size_t len)
                {
                    if (RI_DIFF == 1)
                    {
                        return;
                    }
                    Float temp[RI_DIFF * 2]{};
                    for (auto it = arr, end = arr + len; it < end; it += RI_DIFF * 2)
                    {
                        for (size_t i = 0; i < RI_DIFF; i++)
                        {
                            temp[i] = it[i * 2];
                            temp[i + RI_DIFF] = it[i * 2 + 1];
                        }
                        std::copy(temp, temp + RI_DIFF * 2, it);
                    }
                };
                transToRIRI(in_out, float_len);
                transToRIRI(in, float_len);
                using Complex = std::complex<Float>;
                inv = inv < 0 ? Float(2) / float_len : inv * Float(2);
                {
                    auto r0 = in_out[0], i0 = in_out[1], r1 = in[0], i1 = in[1];
                    transform2(r0, i0);
                    transform2(r1, i1);
                    r0 *= r1, i0 *= i1;
                    transform2(r0, i0);
                    in_out[0] = r0 * 0.5 * inv, in_out[1] = i0 * 0.5 * inv;
                    auto temp = Complex(in_out[2], in_out[3]) * Complex(in[2], in[3]) * inv;
                    in_out[2] = temp.real(), in_out[3] = temp.imag();
                    inv /= Float(8);
                    dot_rfft(&in_out[4], &in_out[6], &in[4], &in[6], Complex(COS_PI_8, -COS_PI_8), inv);
                }
                BinRevTableComplexIterHP<Float> table(31, 32);
                for (size_t begin = 8; begin < float_len; begin *= 2)
                {
                    table.reset(begin / 2);
                    auto it0 = in_out + begin, it1 = it0 + begin - 2, it2 = in + begin, it3 = it2 + begin - 2;
                    for (; it0 < it1; it0 += 2, it1 -= 2, it2 += 2, it3 -= 2)
                    {
                        auto omega = table.iterate();
                        dot_rfft(it0, it1, it2, it3, omega, inv);
                    }
                }
                transToRRII(in_out, float_len);
            }

            inline void real_dot_binrev4(Float64 in_out[], Float64 in[], size_t float_len)
            {
                using Complex = std::complex<Float64>;
                Float64 inv = 1.0 / float_len;
                real_dot_binrev<4>(in_out, in, 16, inv);
                inv = 0.25 / float_len;
                const Float64X4 inv4 = F64X4(inv);
                BinRevTableC64X4HP table(31, 32);
                for (size_t begin = 16; begin < float_len; begin *= 2)
                {
                    table.reset(begin / 2);
                    auto it0 = in_out + begin, it1 = it0 + begin - 8, it2 = in + begin, it3 = it2 + begin - 8;
                    for (; it0 < it1; it0 += 8, it1 -= 8, it2 += 8, it3 -= 8)
                    {
                        dot_rfftX4(it0, it1, it2, it3, table.iterate(), inv4);
                    }
                }
            }
            template <bool TO_INT = false>
            inline void real_conv_avx(F64 *in_out1, F64 *in2, size_t float_len)
            {
                assert(is_2pow(float_len));
                FFTAVX::difRecRadix4<true>(in_out1, float_len);
                FFTAVX::difRecRadix4<true>(in2, float_len);
                real_dot_binrev4(in_out1, in2, float_len);
                FFTAVX::iditRecRadix4<true, TO_INT>(in_out1, float_len);
            }
        }
    }
}
#endif // FFT_AVX_HPP