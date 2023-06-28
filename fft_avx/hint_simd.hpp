#include <tuple>
#include <iostream>
#include <complex>
#include <immintrin.h>
#ifndef HINT_SIMD_HPP
#define HINT_SIMD_HPP

#pragma GCC target("fma")
#pragma GCC target("avx2")
namespace hint_simd
{
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
            memcpy(&data, ptr, 32);
        }
        // 用4个数构造
        DoubleX4(double a7, double a6, double a5, double a4)
        {
            data = _mm256_set_pd(a7, a6, a5, a4);
        }
        void clr()
        {
            data = _mm256_setzero_pd();
        }
        void store(void *ptr) const
        {
            memcpy(ptr, &data, 32);
        }
        void print() const
        {
            double ary[4];
            store(ary);
            printf("(%lf,%lf,%lf,%lf)\n",
                   ary[0], ary[1], ary[2], ary[3]);
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
    };

    using Complex = std::complex<double>;
    // 2个复数并行
    struct Complex2
    {
        __m256d data;
        Complex2()
        {
            data = _mm256_setzero_pd();
        }
        Complex2(double input)
        {
            data = _mm256_set1_pd(input);
        }
        Complex2(__m256d input)
        {
            data = input;
        }
        Complex2(const Complex2 &input)
        {
            data = input.data;
        }
        // 从连续的数组构造
        Complex2(double const *ptr)
        {
            data = _mm256_loadu_pd(ptr);
        }
        Complex2(Complex a)
        {
            data = _mm256_broadcast_pd((__m128d *)&a);
        }
        Complex2(Complex a, Complex b)
        {
            data = _mm256_set_m128d(*(__m128d *)&b, *(__m128d *)&a);
        }
        Complex2(const Complex *ptr)
        {
            data = _mm256_loadu_pd((const double *)ptr);
        }
        void clr()
        {
            data = _mm256_setzero_pd();
        }
        void store(Complex *a) const
        {
            _mm256_storeu_pd((double *)a, data);
        }
        void print() const
        {
            double ary[4];
            _mm256_storeu_pd(ary, data);
            printf("(%lf,%lf) (%lf,%lf)\n", ary[0], ary[1], ary[2], ary[3]);
        }
        template <int M>
        Complex2 element_mask_neg() const
        {
            static const __m256d neg_mask = _mm256_castsi256_pd(
                _mm256_set_epi64x((M & 8ull) << 60, (M & 4ull) << 61, (M & 2ull) << 62, (M & 1ull) << 63));
            return _mm256_xor_pd(data, neg_mask);
        }
        template <int M>
        Complex2 element_permute() const
        {
            return _mm256_permute_pd(data, M);
        }
        template <int M>
        Complex2 element_permute64() const
        {
            return _mm256_permute4x64_pd(data, M);
        }
        Complex2 all_real() const
        {
            return _mm256_unpacklo_pd(data, data);
            // return _mm256_shuffle_pd(data, data, 0);
            // return _mm256_movedup_pd(data);
        }
        Complex2 all_imag() const
        {
            return _mm256_unpackhi_pd(data, data);
            // return _mm256_shuffle_pd(data, data, 15);
            // return element_permute<0XF>();
        }
        Complex2 swap() const
        {
            return _mm256_shuffle_pd(data, data, 5);
            // return element_permute<0X5>();
        }
        Complex2 mul_neg_i() const
        {
            static const __m256d subber{};
            return Complex2(_mm256_addsub_pd(subber, data)).swap();
            // return swap().conj();
        }
        Complex2 conj() const
        {
            return element_mask_neg<10>();
        }
        Complex2 linear_mul(Complex2 input) const
        {
            return _mm256_mul_pd(data, input.data);
        }
        Complex2 operator+(Complex2 input) const
        {
            return _mm256_add_pd(data, input.data);
        }
        Complex2 operator-(Complex2 input) const
        {
            return _mm256_sub_pd(data, input.data);
        }
        Complex2 operator*(Complex2 input) const
        {
            const __m256d a_rr = all_real().data;
            const __m256d a_ii = all_imag().data;
            const __m256d b_ir = input.swap().data;
            return _mm256_addsub_pd(_mm256_mul_pd(a_rr, input.data), _mm256_mul_pd(a_ii, b_ir));
            // auto imag = _mm256_mul_pd(all_imag().data, input.swap().data);
            // return _mm256_fmaddsub_pd(all_real().data, input.data, imag);
        }
        Complex2 operator/(Complex2 input) const
        {
            return _mm256_div_pd(data, input.data);
        }
    };
}
#endif