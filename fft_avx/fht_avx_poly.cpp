// Lightning Fast
// Fast Hartley Transform
// https://github.com/With-Sky/HintFFT
#include <tuple>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <complex>
#include <immintrin.h>
#ifndef HINT_SIMD_HPP
#define HINT_SIMD_HPP

#pragma GCC target("avx2")
#pragma GCC target("fma")

namespace hint_simd
{
    template <typename T, size_t LEN>
    class AlignAry
    {
    private:
        alignas(4096) T ary[LEN];

    public:
        constexpr AlignAry() {}
        constexpr T &operator[](size_t index)
        {
            return ary[index];
        }
        constexpr const T &operator[](size_t index) const
        {
            return ary[index];
        }
        constexpr size_t size() const
        {
            return LEN;
        }
        T *data()
        {
            return reinterpret_cast<T *>(ary);
        }
        const T *data() const
        {
            return reinterpret_cast<const T *>(ary);
        }
        template <typename Ty>
        Ty *cast_ptr()
        {
            return reinterpret_cast<Ty *>(ary);
        }
        template <typename Ty>
        const Ty *cast_ptr() const
        {
            return reinterpret_cast<const Ty *>(ary);
        }
    };
    // Use AVX
    // 256bit simd
    // 4个Double并行
    struct DoubleX4
    {
        __m256d data;
        DoubleX4()
        {
            data = _mm256_setzero_pd();
        }
        DoubleX4(double input)
        {
            data = _mm256_set1_pd(input);
        }
        DoubleX4(__m256d input)
        {
            data = input;
        }
        DoubleX4(const DoubleX4 &input)
        {
            data = input.data;
        }
        // 从连续的数组构造
        DoubleX4(double const *ptr)
        {
            loadu(ptr);
        }
        // 用4个数构造
        DoubleX4(double a3, double a2, double a1, double a0)
        {
            data = _mm256_set_pd(a3, a2, a1, a0);
        }
        void clr()
        {
            data = _mm256_setzero_pd();
        }
        void load(double const *ptr)
        {
            data = _mm256_load_pd(ptr);
        }
        void loadu(double const *ptr)
        {
            data = _mm256_loadu_pd(ptr);
        }
        void store(double *ptr) const
        {
            _mm256_store_pd(ptr, data);
        }
        void storeu(double *ptr) const
        {
            _mm256_storeu_pd(ptr, data);
        }
        void operator<<(double *ptr)
        {
            data = _mm256_loadu_pd(ptr);
        }
        void print() const
        {
            double ary[4];
            store(ary);
            printf("(%lf,%lf,%lf,%lf)\n",
                   ary[0], ary[1], ary[2], ary[3]);
        }
        template <int N>
        DoubleX4 permute4x64() const
        {
            return _mm256_permute4x64_pd(data, N);
        }
        template <int N>
        DoubleX4 permute() const
        {
            return _mm256_permute_pd(data, N);
        }
        DoubleX4 reverse() const
        {
            return permute4x64<0b00011011>();
        }
        DoubleX4 addsub(DoubleX4 input) const
        {
            return _mm256_addsub_pd(data, input.data);
        }
        DoubleX4 fmadd(DoubleX4 mul1, DoubleX4 mul2) const
        {
            return _mm256_fmadd_pd(mul1.data, mul2.data, data);
        }
        DoubleX4 fmsub(DoubleX4 mul1, DoubleX4 mul2) const
        {
            return _mm256_fmsub_pd(mul1.data, mul2.data, data);
        }
        DoubleX4 fnmadd(DoubleX4 mul1, DoubleX4 mul2) const
        {
            return _mm256_fnmadd_pd(mul1.data, mul2.data, data);
        }
        DoubleX4 fnmsub(DoubleX4 mul1, DoubleX4 mul2) const
        {
            return _mm256_fnmsub_pd(mul1.data, mul2.data, data);
        }
        DoubleX4 operator+(DoubleX4 input) const
        {
            return _mm256_add_pd(data, input.data);
        }
        DoubleX4 operator-(DoubleX4 input) const
        {
            return _mm256_sub_pd(data, input.data);
        }
        DoubleX4 operator*(DoubleX4 input) const
        {
            return _mm256_mul_pd(data, input.data);
        }
        DoubleX4 operator/(DoubleX4 input) const
        {
            return _mm256_div_pd(data, input.data);
        }
        DoubleX4 operator-() const
        {
            return _mm256_sub_pd(_mm256_setzero_pd(), data);
        }
    };
}

#endif

#include <array>
#include <vector>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>

namespace hint
{
    using namespace hint_simd;
    using Float32 = float;
    using Float64 = double;

    constexpr Float64 HINT_PI = 3.141592653589793238462643;
    constexpr Float64 HINT_2PI = HINT_PI * 2;
    constexpr size_t FHT_MAX_LEN = size_t(1) << 21;

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
    template <typename T>
    constexpr std::pair<T, T> div_mod(T dividend, T divisor)
    {
        return std::make_pair(dividend / divisor, dividend % divisor);
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
        // 返回单位圆上辐角为theta的点
        template <typename FloatTy>
        inline auto unit_root(FloatTy theta)
        {
            return std::polar<FloatTy>(1.0, theta);
        }
        namespace hint_fht
        {
            namespace split_radix
            {
                template <size_t LEN, typename FloatTy>
                struct FHT
                {
                    static constexpr size_t fht_len = LEN;
                    static constexpr size_t half_len = fht_len / 2;
                    static constexpr size_t quarter_len = fht_len / 4;
                    static constexpr size_t oct_len = fht_len / 8;
                    static constexpr int log_len = hint_log2(fht_len);

                    using HalfFHT = FHT<half_len, FloatTy>;
                    using QuarterFHT = FHT<quarter_len, FloatTy>;

                    static FloatTy cos_omega(size_t i)
                    {
                        return std::cos(i * HINT_2PI / fht_len);
                    }
                    static FloatTy sin_omega(size_t i)
                    {
                        return std::sin(i * HINT_2PI / fht_len);
                    }
                    static std::complex<FloatTy> comp_omega(size_t i)
                    {
                        return std::polar<FloatTy>(1.0, i * HINT_2PI / fht_len);
                    }

                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        using value_type = typename std::iterator_traits<FloatIt>::value_type;
                        static_assert(std::is_same<value_type, FloatTy>::value, "Must be same as the FHT template float type");

                        QuarterFHT::dit(in_out + half_len + quarter_len);
                        QuarterFHT::dit(in_out + half_len);
                        HalfFHT::dit(in_out);
                        transform2(in_out[half_len], in_out[half_len + quarter_len]);
                        static constexpr value_type SQRT_2 = 1.4142135623730950488016887242097;
                        in_out[half_len + oct_len] *= SQRT_2, in_out[half_len + oct_len + quarter_len] *= SQRT_2;

                        auto it0 = in_out + 1, it1 = in_out + quarter_len - 1;
                        auto it2 = it0 + quarter_len, it3 = it1 + quarter_len;
                        static FloatTy cos_arr1[8] = {1, cos_omega(1), cos_omega(2), cos_omega(3), cos_omega(4), cos_omega(5), cos_omega(6), cos_omega(7)};
                        static FloatTy sin_arr1[8] = {0, sin_omega(1), sin_omega(2), sin_omega(3), sin_omega(4), sin_omega(5), sin_omega(6), sin_omega(7)};
                        static FloatTy cos_arr3[8] = {1, cos_omega(3), cos_omega(6), cos_omega(9), cos_omega(12), cos_omega(15), cos_omega(18), cos_omega(21)};
                        static FloatTy sin_arr3[8] = {0, sin_omega(3), sin_omega(6), sin_omega(9), sin_omega(12), sin_omega(15), sin_omega(18), sin_omega(21)};
                        for (; it0 < in_out + 4; it0++, it1--, it2++, it3--)
                        {
                            auto omega1 = std::complex<FloatTy>(cos_arr1[it0 - in_out], sin_arr1[it0 - in_out]);
                            auto omega3 = std::complex<FloatTy>(cos_arr3[it0 - in_out], sin_arr3[it0 - in_out]);
                            auto temp4 = it0[half_len], temp5 = it1[half_len];
                            auto temp0 = temp4 * omega1.real() + temp5 * omega1.imag();
                            auto temp2 = temp4 * omega1.imag() - temp5 * omega1.real();
                            temp4 = it2[half_len], temp5 = it3[half_len];
                            auto temp1 = temp4 * omega3.real() + temp5 * omega3.imag();
                            auto temp3 = temp4 * omega3.imag() - temp5 * omega3.real();

                            transform2(temp0, temp1);
                            transform2(temp3, temp2);

                            temp4 = it0[0], temp5 = it1[0];
                            it0[0] = temp4 + temp0, it1[0] = temp5 + temp1;
                            it0[half_len] = temp4 - temp0, it1[half_len] = temp5 - temp1;

                            temp4 = it2[0], temp5 = it3[0];
                            it2[0] = temp4 + temp2, it3[0] = temp5 + temp3;
                            it2[half_len] = temp4 - temp2, it3[half_len] = temp5 - temp3;
                        }
                        it1 -= 3, it3 -= 3;
                        static const DoubleX4 cos_unit1(cos_arr1[4]), sin_unit1(sin_arr1[4]);
                        static const DoubleX4 cos_unit3 = DoubleX4(cos_arr3[4]).reverse();
                        static const DoubleX4 sin_unit3 = DoubleX4(sin_arr3[4]).reverse();
                        DoubleX4 cos1_4(cos_arr1 + 4), sin1_4(sin_arr1 + 4);
                        DoubleX4 cos3_4 = DoubleX4(cos_arr3 + 4).reverse();
                        DoubleX4 sin3_4 = DoubleX4(sin_arr3 + 4).reverse();
                        for (; it0 < in_out + oct_len; it0 += 4, it1 -= 4, it2 += 4, it3 -= 4)
                        {
                            DoubleX4 temp0, temp1, temp2, temp3, temp4, temp5;
                            temp4.load(&it0[half_len]), temp5.loadu(&it1[half_len]);
                            temp5 = temp5.reverse();
                            temp0 = (temp4 * cos1_4).fmadd(temp5, sin1_4);
                            temp2 = (temp4 * sin1_4).fnmadd(temp5, cos1_4);
                            temp4.load(&it2[half_len]), temp5.loadu(&it3[half_len]);
                            temp4 = temp4.reverse();
                            temp1 = (temp5 * sin3_4).fmadd(temp4, cos3_4);
                            temp3 = (temp5 * cos3_4).fmsub(temp4, sin3_4);

                            temp4 = temp0, temp5 = temp1;
                            temp0 = temp4 + temp5.reverse();
                            temp1 = temp4.reverse() - temp5;
                            temp4 = temp2, temp5 = temp3;
                            temp2 = temp5.reverse() - temp4;
                            temp3 = temp5 + temp4.reverse();

                            temp4.load(&it0[0]), temp5.loadu(&it1[0]);
                            (temp4 + temp0).store(&it0[0]);
                            (temp5 + temp1).storeu(&it1[0]);
                            (temp4 - temp0).store(&it0[half_len]);
                            (temp5 - temp1).storeu(&it1[half_len]);

                            temp4.load(&it2[0]), temp5.loadu(&it3[0]);
                            (temp4 + temp2).store(&it2[0]);
                            (temp5 + temp3).storeu(&it3[0]);
                            (temp4 - temp2).store(&it2[half_len]);
                            (temp5 - temp3).storeu(&it3[half_len]);

                            temp0 = cos1_4, temp1 = sin1_4;
                            cos1_4 = temp0 * cos_unit1 - temp1 * sin_unit1;
                            sin1_4 = temp0 * sin_unit1 + temp1 * cos_unit1;
                            temp0 = cos3_4, temp1 = sin3_4;
                            cos3_4 = temp0 * cos_unit3 - temp1 * sin_unit3;
                            sin3_4 = temp0 * sin_unit3 + temp1 * cos_unit3;
                        }
                        transform2(in_out[0], in_out[half_len]);
                        transform2(in_out[oct_len], in_out[half_len + oct_len]);
                        transform2(in_out[oct_len * 2], in_out[half_len + oct_len * 2]);
                        transform2(in_out[oct_len * 3], in_out[half_len + oct_len * 3]);
                    }

                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        using value_type = typename std::iterator_traits<FloatIt>::value_type;
                        static_assert(std::is_same<value_type, FloatTy>::value, "Must be same as the FHT template float type");

                        transform2(in_out[0], in_out[half_len]);
                        transform2(in_out[oct_len], in_out[half_len + oct_len]);
                        transform2(in_out[oct_len * 2], in_out[half_len + oct_len * 2]);
                        transform2(in_out[oct_len * 3], in_out[half_len + oct_len * 3]);
                        auto it0 = in_out + 1, it1 = in_out + quarter_len - 1;
                        auto it2 = it0 + quarter_len, it3 = it1 + quarter_len;
                        static FloatTy cos_arr1[8] = {1, cos_omega(1), cos_omega(2), cos_omega(3), cos_omega(4), cos_omega(5), cos_omega(6), cos_omega(7)};
                        static FloatTy sin_arr1[8] = {0, sin_omega(1), sin_omega(2), sin_omega(3), sin_omega(4), sin_omega(5), sin_omega(6), sin_omega(7)};
                        static FloatTy cos_arr3[8] = {1, cos_omega(3), cos_omega(6), cos_omega(9), cos_omega(12), cos_omega(15), cos_omega(18), cos_omega(21)};
                        static FloatTy sin_arr3[8] = {0, sin_omega(3), sin_omega(6), sin_omega(9), sin_omega(12), sin_omega(15), sin_omega(18), sin_omega(21)};
                        for (; it0 < in_out + 4; it0++, it1--, it2++, it3--)
                        {
                            auto omega1 = std::complex<FloatTy>(cos_arr1[it0 - in_out], sin_arr1[it0 - in_out]);
                            auto omega3 = std::complex<FloatTy>(cos_arr3[it0 - in_out], sin_arr3[it0 - in_out]);
                            auto temp0 = it0[half_len], temp1 = it1[half_len];
                            auto temp2 = it2[half_len], temp3 = it3[half_len];

                            auto temp4 = it0[0], temp5 = it1[0];
                            it0[0] = temp4 + temp0, it1[0] = temp5 + temp1;
                            temp0 = temp4 - temp0, temp1 = temp5 - temp1;

                            temp4 = it2[0], temp5 = it3[0];
                            it2[0] = temp4 + temp2, it3[0] = temp5 + temp3;
                            temp2 = temp4 - temp2, temp3 = temp5 - temp3;

                            transform2(temp0, temp1);
                            transform2(temp3, temp2);

                            it0[half_len] = temp0 * omega1.real() + temp2 * omega1.imag();
                            it1[half_len] = temp0 * omega1.imag() - temp2 * omega1.real();
                            it2[half_len] = temp1 * omega3.real() + temp3 * omega3.imag();
                            it3[half_len] = temp1 * omega3.imag() - temp3 * omega3.real();
                        }
                        it1 -= 3, it3 -= 3;
                        static const DoubleX4 cos_unit1(cos_arr1[4]), sin_unit1(sin_arr1[4]);
                        static const DoubleX4 cos_unit3 = DoubleX4(cos_arr3[4]).reverse();
                        static const DoubleX4 sin_unit3 = DoubleX4(sin_arr3[4]).reverse();
                        DoubleX4 cos1_4(cos_arr1 + 4), sin1_4(sin_arr1 + 4);
                        DoubleX4 cos3_4 = DoubleX4(cos_arr3 + 4).reverse();
                        DoubleX4 sin3_4 = DoubleX4(sin_arr3 + 4).reverse();
                        for (; it0 < in_out + oct_len; it0 += 4, it1 -= 4, it2 += 4, it3 -= 4)
                        {
                            DoubleX4 temp0, temp1, temp2, temp3, temp4, temp5;
                            temp0.load(&it0[half_len]), temp1.loadu(&it1[half_len]);
                            temp2.load(&it2[half_len]), temp3.loadu(&it3[half_len]);

                            temp4.load(&it0[0]), temp5.loadu(&it1[0]);
                            (temp4 + temp0).store(&it0[0]);
                            (temp5 + temp1).storeu(&it1[0]);
                            temp0 = temp4 - temp0, temp1 = temp5 - temp1;

                            temp4.load(&it2[0]), temp5.loadu(&it3[0]);
                            (temp4 + temp2).store(&it2[0]);
                            (temp5 + temp3).storeu(&it3[0]);
                            temp2 = temp4 - temp2, temp3 = temp5 - temp3;

                            temp4 = temp0, temp5 = temp1;
                            temp0 = temp4 + temp5.reverse();
                            temp1 = temp4.reverse() - temp5;
                            temp4 = temp2, temp5 = temp3;
                            temp2 = temp5.reverse() - temp4;
                            temp3 = temp5 + temp4.reverse();

                            (temp0 * cos1_4 + temp2 * sin1_4).store(&it0[half_len]);
                            (temp0 * sin1_4 - temp2 * cos1_4).reverse().storeu(&it1[half_len]);
                            (temp1 * cos3_4 + temp3 * sin3_4).reverse().store(&it2[half_len]);
                            (temp1 * sin3_4 - temp3 * cos3_4).storeu(&it3[half_len]);

                            temp0 = cos1_4, temp1 = sin1_4;
                            cos1_4 = temp0 * cos_unit1 - temp1 * sin_unit1;
                            sin1_4 = temp0 * sin_unit1 + temp1 * cos_unit1;
                            temp0 = cos3_4, temp1 = sin3_4;
                            cos3_4 = temp0 * cos_unit3 - temp1 * sin_unit3;
                            sin3_4 = temp0 * sin_unit3 + temp1 * cos_unit3;
                        }

                        transform2(in_out[half_len], in_out[half_len + quarter_len]);
                        static constexpr value_type SQRT_2 = 1.4142135623730950488016887242097;
                        in_out[half_len + oct_len] *= SQRT_2, in_out[half_len + oct_len + quarter_len] *= SQRT_2;
                        HalfFHT::dif(in_out);
                        QuarterFHT::dif(in_out + half_len);
                        QuarterFHT::dif(in_out + half_len + quarter_len);
                    }
                };

                template <typename FloatTy>
                struct FHT<0, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out) {}
                    template <typename FloatIt>
                    static void dif(FloatIt in_out) {}
                };

                template <typename FloatTy>
                struct FHT<1, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out) {}
                    template <typename FloatIt>
                    static void dif(FloatIt in_out) {}
                };

                template <typename FloatTy>
                struct FHT<2, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                };

                template <typename FloatTy>
                struct FHT<4, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        auto temp0 = in_out[0], temp1 = in_out[1];
                        auto temp2 = in_out[2], temp3 = in_out[3];
                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        in_out[0] = temp0 + temp2;
                        in_out[1] = temp1 + temp3;
                        in_out[2] = temp0 - temp2;
                        in_out[3] = temp1 - temp3;
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        auto temp0 = in_out[0], temp1 = in_out[1];
                        auto temp2 = in_out[2], temp3 = in_out[3];
                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                    }
                };

                template <typename FloatTy>
                struct FHT<8, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        auto temp0 = in_out[0], temp1 = in_out[1];
                        auto temp2 = in_out[2], temp3 = in_out[3];
                        auto temp4 = in_out[4], temp5 = in_out[5];
                        auto temp6 = in_out[6], temp7 = in_out[7];
                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        transform2(temp4, temp5);
                        transform2(temp6, temp7);
                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        static constexpr decltype(temp0) SQRT_2 = 1.4142135623730950488016887242097;
                        temp5 *= SQRT_2, temp7 *= SQRT_2;
                        in_out[0] = temp0 + temp4;
                        in_out[1] = temp1 + temp5;
                        in_out[2] = temp2 + temp6;
                        in_out[3] = temp3 + temp7;
                        in_out[4] = temp0 - temp4;
                        in_out[5] = temp1 - temp5;
                        in_out[6] = temp2 - temp6;
                        in_out[7] = temp3 - temp7;
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        auto temp0 = in_out[0], temp1 = in_out[1];
                        auto temp2 = in_out[2], temp3 = in_out[3];
                        auto temp4 = in_out[4], temp5 = in_out[5];
                        auto temp6 = in_out[6], temp7 = in_out[7];
                        transform2(temp0, temp4);
                        transform2(temp1, temp5);
                        transform2(temp2, temp6);
                        transform2(temp3, temp7);
                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        static constexpr decltype(temp0) SQRT_2 = 1.4142135623730950488016887242097;
                        temp5 *= SQRT_2, temp7 *= SQRT_2;
                        transform2(temp4, temp6);
                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                        in_out[4] = temp4 + temp5;
                        in_out[5] = temp4 - temp5;
                        in_out[6] = temp6 + temp7;
                        in_out[7] = temp6 - temp7;
                    }
                };

                template <typename FloatTy>
                struct FHT<16, FloatTy>
                {
                    static void init() {}
                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        using value_type = typename std::iterator_traits<FloatIt>::value_type;
                        FHT<4, FloatTy>::dit(in_out + 12);
                        FHT<4, FloatTy>::dit(in_out + 8);
                        FHT<8, FloatTy>::dit(in_out);
                        static constexpr value_type SQRT_2 = 1.4142135623730950488016887242097;
                        static constexpr value_type COS_16 = 0.9238795325112867561281831893967; // cos(2PI/16);
                        static constexpr value_type SIN_16 = 0.3826834323650897717284599840304; // sin(2PI/16);
                        auto temp4 = in_out[9], temp5 = in_out[11];
                        auto temp0 = temp4 * COS_16 + temp5 * SIN_16;
                        auto temp2 = temp4 * SIN_16 - temp5 * COS_16;

                        temp4 = in_out[13], temp5 = in_out[15];
                        auto temp1 = temp4 * SIN_16 + temp5 * COS_16;
                        auto temp3 = temp4 * COS_16 - temp5 * SIN_16;

                        transform2(temp0, temp1);
                        transform2(temp3, temp2);

                        temp4 = in_out[1], temp5 = in_out[3];
                        in_out[1] = temp4 + temp0, in_out[3] = temp5 + temp1;
                        in_out[9] = temp4 - temp0, in_out[11] = temp5 - temp1;

                        temp4 = in_out[5], temp5 = in_out[7];
                        in_out[5] = temp4 + temp2, in_out[7] = temp5 + temp3;
                        in_out[13] = temp4 - temp2, in_out[15] = temp5 - temp3;

                        in_out[10] *= SQRT_2, in_out[14] *= SQRT_2;
                        transform2(in_out[8], in_out[12]);
                        transform2(in_out[0], in_out[8]);
                        transform2(in_out[2], in_out[10]);
                        transform2(in_out[4], in_out[12]);
                        transform2(in_out[6], in_out[14]);
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        using value_type = typename std::iterator_traits<FloatIt>::value_type;
                        static constexpr value_type SQRT_2 = 1.4142135623730950488016887242097;
                        static constexpr value_type COS_16 = 0.9238795325112867561281831893967; // cos(2PI/16);
                        static constexpr value_type SIN_16 = 0.3826834323650897717284599840304; // sin(2PI/16);
                        transform2(in_out[0], in_out[8]);
                        transform2(in_out[2], in_out[10]);
                        transform2(in_out[4], in_out[12]);
                        transform2(in_out[6], in_out[14]);
                        transform2(in_out[8], in_out[12]);
                        in_out[10] *= SQRT_2, in_out[14] *= SQRT_2;

                        auto temp0 = in_out[9], temp1 = in_out[11];
                        auto temp2 = in_out[13], temp3 = in_out[15];

                        auto temp4 = in_out[1], temp5 = in_out[3];
                        in_out[1] = temp4 + temp0;
                        in_out[3] = temp5 + temp1;
                        temp0 = temp4 - temp0;
                        temp1 = temp5 - temp1;

                        temp4 = in_out[5], temp5 = in_out[7];
                        in_out[5] = temp4 + temp2;
                        in_out[7] = temp5 + temp3;
                        temp2 = temp4 - temp2;
                        temp3 = temp5 - temp3;

                        transform2(temp0, temp1);
                        transform2(temp3, temp2);

                        in_out[9] = temp0 * COS_16 + temp2 * SIN_16;
                        in_out[11] = temp0 * SIN_16 - temp2 * COS_16;
                        in_out[13] = temp1 * SIN_16 + temp3 * COS_16;
                        in_out[15] = temp1 * COS_16 - temp3 * SIN_16;

                        FHT<8, FloatTy>::dif(in_out);
                        FHT<4, FloatTy>::dif(in_out + 8);
                        FHT<4, FloatTy>::dif(in_out + 12);
                    }
                };
            }

            // 默认FHT为分裂基
            template <size_t len, typename FloatTy>
            using FHTDefault = split_radix::FHT<len, FloatTy>;

            /// @brief 初始化所有FHT查找表
            /// @tparam FloatTy
            template <typename FloatTy>
            inline void fht_init()
            {
                FHTDefault<FHT_MAX_LEN, FloatTy>::init();
            }

            // 辅助选择函数
            template <size_t LEN = 1>
            inline void fht_dit_template_alt(Float64 *in_out, size_t fht_len)
            {
                if (fht_len > LEN)
                {
                    fht_dit_template_alt<LEN * 2>(in_out, fht_len);
                    return;
                }
                FHTDefault<LEN, Float64>::dit(in_out);
            }
            template <>
            inline void fht_dit_template_alt<FHT_MAX_LEN * 2>(Float64 *in_out, size_t fht_len)
            {
                throw("Length of FHT can't be larger than FHT_MAX_LEN");
            }

            // 辅助选择函数
            template <size_t LEN = 1>
            inline void fht_dif_template_alt(Float64 *in_out, size_t fht_len)
            {
                if (fht_len > LEN)
                {
                    fht_dif_template_alt<LEN * 2>(in_out, fht_len);
                    return;
                }
                FHTDefault<LEN, Float64>::dif(in_out);
            }
            template <>
            inline void fht_dif_template_alt<FHT_MAX_LEN * 2>(Float64 *in_out, size_t fht_len)
            {
                throw("Length of FHT can't be larger than FHT_MAX_LEN");
            }

            // 时间抽取快速哈特莱变换
            inline void fht_dit(Float64 *in_out, size_t fht_len)
            {
                fht_dit_template_alt<1>(in_out, fht_len);
            }
            // 频率抽取快速哈特莱变换
            inline void fht_dif(Float64 *in_out, size_t fht_len)
            {
                fht_dif_template_alt<1>(in_out, fht_len);
            }

            // FHT加速卷积
            inline void fht_convolution(Float64 fht_ary1[], Float64 fht_ary2[], Float64 out[], size_t fht_len)
            {
                if (fht_len == 0)
                {
                    return;
                }
                if (fht_len == 1)
                {
                    out[0] = fht_ary1[0] * fht_ary2[0];
                    return;
                }
                fht_len = int_floor2(fht_len);
                if (fht_len > FHT_MAX_LEN)
                {
                    throw("FHT len cannot be larger than FHT_MAX_LEN");
                }
                fht_dif(fht_ary1, fht_len);
                // 两个输入相同时只进行一次计算，提升平方速度
                if (fht_ary1 != fht_ary2)
                {
                    fht_dif(fht_ary2, fht_len);
                }
                const double inv = 0.5 / fht_len;
                out[0] = fht_ary1[0] * fht_ary2[0] / fht_len;
                out[1] = fht_ary1[1] * fht_ary2[1] / fht_len;
                if (fht_len == 2)
                {
                    return;
                }
                // DHT的卷积定理
                auto temp0 = fht_ary1[2], temp1 = fht_ary1[3];
                auto temp2 = fht_ary2[2], temp3 = fht_ary2[3];
                transform2(temp0, temp1);
                out[2] = (temp2 * temp0 + temp3 * temp1) * inv;
                out[3] = (temp3 * temp0 - temp2 * temp1) * inv;
                for (size_t i = 4; i < fht_len; i *= 2)
                {
                    auto it0 = fht_ary1 + i, it1 = it0 + i - 1;
                    auto it2 = fht_ary2 + i, it3 = it2 + i - 1;
                    auto it4 = out + i, it5 = it4 + i - 1;
                    for (; it0 < it1; it0 += 2, it1 -= 2, it2 += 2, it3 -= 2, it4 += 2, it5 -= 2)
                    {
                        temp0 = *it0, temp1 = *it1, temp2 = *it2, temp3 = *it3;
                        transform2(temp0, temp1);
                        *it4 = (temp2 * temp0 + temp3 * temp1) * inv;
                        *it5 = (temp3 * temp0 - temp2 * temp1) * inv;
                        temp0 = *(it1 - 1), temp1 = *(it0 + 1), temp2 = *(it3 - 1), temp3 = *(it2 + 1);
                        transform2(temp0, temp1);
                        *(it5 - 1) = (temp2 * temp0 + temp3 * temp1) * inv;
                        *(it4 + 1) = (temp3 * temp0 - temp2 * temp1) * inv;
                    }
                }
                fht_dit(out, fht_len);
            }
        }
    }
}

using namespace std;
using namespace hint;
using namespace hint_simd;
using namespace hint_transform;
using namespace hint_fht;

void poly_multiply(unsigned *a, int n, unsigned *b, int m, unsigned *c)
{
    size_t conv_len = m + n + 1, fht_len = int_ceil2(conv_len);
    static AlignAry<Float64, FHT_MAX_LEN> buffer1, buffer2;
    std::copy(a, a + n + 1, buffer1.data());
    std::copy(b, b + n + 1, buffer2.data());
    fht_convolution(buffer1.data(), buffer2.data(), buffer1.data(), fht_len);
    size_t i = 0, rem = conv_len % 4;
    for (; i < conv_len - rem; i += 4)
    {
        c[i] = buffer1[i] + 0.5;
        c[i + 1] = buffer1[i + 1] + 0.5;
        c[i + 2] = buffer1[i + 2] + 0.5;
        c[i + 3] = buffer1[i + 3] + 0.5;
    }
    for (; i < conv_len; i++)
    {
        c[i] = buffer1[i] + 0.5;
    }
}

int main()
{
    std::ios::sync_with_stdio(false);
    cout.tie(nullptr);
    cin.tie(nullptr);
    constexpr size_t len = 1 << 10;
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
    poly_multiply(a, len - 1, b, len - 1, c);
    auto t2 = chrono::steady_clock::now();
    for (auto i : c)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
}