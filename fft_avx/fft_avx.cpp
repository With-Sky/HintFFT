// TSKY 2024/2/14
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <chrono>
#include <string>
#include <bitset>
#include <type_traits>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <cstring>
#include <tuple>
#include <immintrin.h>
#ifndef HINT_SIMD_HPP
#define HINT_SIMD_HPP

#pragma GCC target("fma")
#pragma GCC target("avx2")
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
    struct Float64X4
    {
        __m256d data;
        Float64X4()
        {
            data = _mm256_setzero_pd();
        }
        Float64X4(double in_out)
        {
            data = _mm256_set1_pd(in_out);
        }
        Float64X4(__m256d in_out)
        {
            data = in_out;
        }
        Float64X4(const Float64X4 &in_out)
        {
            data = in_out.data;
        }
        // 从连续的数组构造
        template <typename T>
        Float64X4(const T *ptr)
        {
            load(ptr);
        }
        // 用4个数构造
        Float64X4(double a7, double a6, double a5, double a4)
        {
            data = _mm256_set_pd(a7, a6, a5, a4);
        }
        void clr()
        {
            data = _mm256_setzero_pd();
        }
        template <typename T>
        void load(const T *ptr)
        {
            data = _mm256_load_pd((double *)ptr);
        }
        template <typename T>
        void loadu(const T *ptr)
        {
            data = _mm256_loadu_pd((double *)ptr);
        }
        template <typename T>
        void store(T *ptr) const
        {
            _mm256_store_pd((double *)ptr, data);
        }
        template <typename T>
        void storeu(T *ptr) const
        {
            _mm256_storeu_pd((double *)ptr, data);
        }
        void print() const
        {
            double ary[4];
            store(ary);
            printf("(%lf,%lf,%lf,%lf)\n",
                   ary[0], ary[1], ary[2], ary[3]);
        }
        template <int N>
        Float64X4 permute4x64() const
        {
            return _mm256_permute4x64_pd(data, N);
        }
        template <int N>
        Float64X4 permute() const
        {
            return _mm256_permute_pd(data, N);
        }
        Float64X4 reverse() const
        {
            return permute4x64<0b00011011>();
        }
        Float64X4 addsub(Float64X4 in_out) const
        {
            return _mm256_addsub_pd(data, in_out.data);
        }
        Float64X4 fmadd(Float64X4 mul1, Float64X4 mul2) const
        {
            return _mm256_fmadd_pd(mul1.data, mul2.data, data);
        }
        Float64X4 fmsub(Float64X4 mul1, Float64X4 mul2) const
        {
            return _mm256_fmsub_pd(mul1.data, mul2.data, data);
        }
        Float64X4 fnmadd(Float64X4 mul1, Float64X4 mul2) const
        {
            return _mm256_fnmadd_pd(mul1.data, mul2.data, data);
        }
        Float64X4 fnmsub(Float64X4 mul1, Float64X4 mul2) const
        {
            return _mm256_fnmsub_pd(mul1.data, mul2.data, data);
        }
        Float64X4 operator+(Float64X4 in_out) const
        {
            return _mm256_add_pd(data, in_out.data);
        }
        Float64X4 operator-(Float64X4 in_out) const
        {
            return _mm256_sub_pd(data, in_out.data);
        }
        Float64X4 operator*(Float64X4 in_out) const
        {
            return _mm256_mul_pd(data, in_out.data);
        }
        Float64X4 operator/(Float64X4 in_out) const
        {
            return _mm256_div_pd(data, in_out.data);
        }
        Float64X4 operator-() const
        {
            return _mm256_sub_pd(_mm256_setzero_pd(), data);
        }
    };
    // Use AVX
    // 256bit simd
    using Complex64 = std::complex<double>;
    // 2个复数并行
    struct Comple64x2 : public Float64X4
    {
        Comple64x2()
        {
            data = _mm256_setzero_pd();
        }
        Comple64x2(double in_out)
        {
            data = _mm256_set1_pd(in_out);
        }
        Comple64x2(const __m256d &in_out)
        {
            data = in_out;
        }
        Comple64x2(const Comple64x2 &in_out)
        {
            data = in_out.data;
        }
        Comple64x2(const Float64X4 &in_out)
        {
            data = in_out.data;
        }
        // 从连续的数组构造
        Comple64x2(double const *ptr)
        {
            data = _mm256_load_pd(ptr);
        }
        Comple64x2(const Complex64 &a, const Complex64 &b)
        {
            data = _mm256_loadu2_m128d((const double *)&b, (const double *)&a);
        }
        Comple64x2(const Complex64 *ptr)
        {
            load((const double *)ptr);
        }
        void print() const
        {
            double ary[4];
            storeu(ary);
            printf("(%lf,%lf) (%lf,%lf)\n", ary[0], ary[1], ary[2], ary[3]);
        }
        Comple64x2 all_real() const
        {
            return _mm256_unpacklo_pd(data, data);
            // return _mm256_shuffle_pd(data, data, 0);
            // return _mm256_movedup_pd(data);
        }
        Comple64x2 all_imag() const
        {
            return _mm256_unpackhi_pd(data, data);
            // return _mm256_shuffle_pd(data, data, 15);
            // return element_permute<0XF>();
        }
        template <int M>
        Comple64x2 element_mask_neg() const
        {
            static const __m256d neg_mask = _mm256_castsi256_pd(
                _mm256_set_epi64x((M & 8ull) << 60, (M & 4ull) << 61, (M & 2ull) << 62, (M & 1ull) << 63));
            return _mm256_xor_pd(data, neg_mask);
        }
        Comple64x2 swap() const
        {
            return permute<0X5>();
        }
        Comple64x2 mul_neg_j() const
        {
            return Comple64x2(_mm256_addsub_pd(_mm256_setzero_pd(), data)).swap();
            // return swap().conj();
        }
        Comple64x2 conj() const
        {
            return element_mask_neg<10>();
        }
        Comple64x2 linear_mul(Comple64x2 in_out) const
        {
            return _mm256_mul_pd(data, in_out.data);
        }
        Comple64x2 operator*(const Comple64x2 &in_out) const
        {
            // const __m256d a_rr = all_real().data;
            // const __m256d a_ii = all_imag().data;
            // const __m256d b_ir = in_out.swap().data;
            // return _mm256_addsub_pd(_mm256_mul_pd(a_rr, in_out.data), _mm256_mul_pd(a_ii, b_ir));
            auto imag = _mm256_mul_pd(all_imag().data, in_out.swap().data);
            return _mm256_fmaddsub_pd(all_real().data, in_out.data, imag);
        }
    };
}
#endif

namespace hint
{
    using Float32 = float;
    using Float64 = double;
    using Complex32 = std::complex<Float32>;
    using Complex64 = std::complex<Float64>;

    constexpr Float64 HINT_PI = 3.141592653589793238462643;
    constexpr Float64 HINT_2PI = HINT_PI * 2;
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
        return (n & (n - 1)) == 0;
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

    // FFT与类FFT变换的命名空间
    namespace hint_transform
    {
        using namespace hint_simd;

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

        template <typename FloatTy, int log_len, int len_div>
        class ComplexTableStatic
        {
        public:
            static constexpr size_t len = size_t(1) << log_len;
            static constexpr size_t table_len = len / len_div;
            static constexpr FloatTy unit = HINT_2PI / len;
            using Ty = std::complex<FloatTy>;
            using TableTy = AlignAry<Ty, table_len>;
            ComplexTableStatic() {}
            ComplexTableStatic(int factor) { init(factor); }
            void allocate() {}
            void init(int factor)
            {
                for (size_t i = 0; i < table.size(); i++)
                {
                    table[i] = unit_root(factor * unit * FloatTy(i));
                }
            }
            const auto &operator[](size_t n) const { return table[n]; }
            auto &operator[](size_t n) { return table[n]; }
            auto get_it(size_t n = 0) const { return &table[n]; }

        private:
            TableTy table;
        };

        static constexpr Float64 COS_8 = 0.70710678118654752440084436210485; // cos(2PI/8)
        static constexpr Float64 SQRT_2 = COS_8 * 2;                         // sqrt(2)
        static constexpr Float64 COS_16 = 0.9238795325112867561281831893968; // cos(2PI/16)
        static constexpr Float64 SIN_16 = 0.3826834323650897717284599840304; // sin(2PI/16)
        namespace hint_fft
        {
            // x * (-j)
            template <typename FloatTy>
            inline std::complex<FloatTy> mul_neg_j(const std::complex<FloatTy> &x)
            {
                return std::complex<FloatTy>(x.imag(), -x.real());
            }

            // x * (j)
            template <typename FloatTy>
            inline std::complex<FloatTy> mul_j(const std::complex<FloatTy> &x)
            {
                return std::complex<FloatTy>(-x.imag(), x.real());
            }

            //(a,bj)->(b,aj)
            template <typename FloatTy>
            inline std::complex<FloatTy> swap_ri(const std::complex<FloatTy> &x)
            {
                return std::complex<FloatTy>(x.imag(), x.real());
            }

            // (a,bj)->(-a,bj)
            template <typename FloatTy>
            inline std::complex<FloatTy> conj_real(const std::complex<FloatTy> &x)
            {
                return std::complex<FloatTy>(-x.real(), x.imag());
            }

            // x*conj(y)
            template <typename FloatTy>
            inline std::complex<FloatTy> mul_conj(const std::complex<FloatTy> &x, const std::complex<FloatTy> &y)
            {
                FloatTy r = x.real() * y.real() + x.imag() * y.imag();
                FloatTy i = x.imag() * y.real() - x.real() * y.imag();
                return std::complex<FloatTy>(r, i);
            }

            namespace split_radix_avx
            {
                template <typename FloatTy, int log_len>
                class FFTTableSplitRadix
                {
                public:
                    using HalfTable = FFTTableSplitRadix<FloatTy, log_len - 1>;
                    using TableTy = ComplexTableStatic<FloatTy, log_len, 4>;

                    static constexpr int factor1 = HalfTable::factor1;
                    static constexpr int factor3 = HalfTable::factor3;
                    FFTTableSplitRadix()
                    {
                        // init();
                    }
                    static void init()
                    {
                        if (has_init)
                        {
                            return;
                        }
                        HalfTable::init();
                        constexpr size_t table_len = table1.table_len;
                        table1.allocate();
                        table3.allocate();
                        for (size_t i = 0; i < table_len; i += 2)
                        {
                            table1[i] = HalfTable::table1[i / 2];
                            table3[i] = HalfTable::table3[i / 2];
                            table1[i + 1] = unit_root(FloatTy(i + 1) * factor1 * table1.unit);
                            table3[i + 1] = unit_root(FloatTy(i + 1) * factor3 * table3.unit);
                        }
                        has_init = true;
                    }
                    static auto get_it1(size_t n = 0) { return table1.get_it(n); }
                    static auto get_it3(size_t n = 0) { return table3.get_it(n); }

                    static TableTy table1;
                    static TableTy table3;

                private:
                    static bool has_init;
                };

                template <typename FloatTy, int log_len>
                typename FFTTableSplitRadix<FloatTy, log_len>::TableTy
                    FFTTableSplitRadix<FloatTy, log_len>::table1;
                template <typename FloatTy, int log_len>
                typename FFTTableSplitRadix<FloatTy, log_len>::TableTy
                    FFTTableSplitRadix<FloatTy, log_len>::table3;
                template <typename FloatTy, int log_len>
                bool FFTTableSplitRadix<FloatTy, log_len>::has_init = false;

                template <typename FloatTy>
                class FFTTableSplitRadix<FloatTy, 4>
                {
                public:
                    static constexpr int factor1 = -1;
                    static constexpr int factor3 = -3;
                    using TableTy = ComplexTableStatic<FloatTy, 4, 4>;
                    FFTTableSplitRadix() { init(); }
                    static void init()
                    {
                        table1.init(factor1);
                        table3.init(factor3);
                    }
                    static auto get_it1(size_t n = 0) { return table1.get_it(n); }
                    static auto get_it3(size_t n = 0) { return table3.get_it(n); }

                    static TableTy table1;
                    static TableTy table3;
                };
                template <typename FloatTy>
                typename FFTTableSplitRadix<FloatTy, 4>::TableTy FFTTableSplitRadix<FloatTy, 4>::table1;
                template <typename FloatTy>
                typename FFTTableSplitRadix<FloatTy, 4>::TableTy FFTTableSplitRadix<FloatTy, 4>::table3;

                template <size_t rank, typename It, typename OmegaIt>
                inline void dit_butterfly_avx(It it, OmegaIt omega1_it, OmegaIt omega3_it)
                {
                    // Comple64x2 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                    Comple64x2 omega0, omega1, omega2, omega3;
                    Comple64x2 temp2(&it[rank * 2]);
                    Comple64x2 temp6(&it[rank * 2 + 2]);
                    Comple64x2 temp3(&it[rank * 3]);
                    Comple64x2 temp7(&it[rank * 3 + 2]);
                    omega0.load(&omega1_it[0]);
                    omega1.load(&omega1_it[2]);
                    omega2.load(&omega3_it[0]);
                    omega3.load(&omega3_it[2]);
                    temp2 = temp2 * omega0;
                    temp3 = temp3 * omega2;
                    temp6 = temp6 * omega1;
                    temp7 = temp7 * omega3;

                    transform2(temp2, temp3);
                    transform2(temp6, temp7);
                    temp3 = temp3.mul_neg_j();
                    temp7 = temp7.mul_neg_j();

                    Comple64x2 temp0(&it[0]);
                    Comple64x2 temp4(&it[2]);
                    Comple64x2 temp1(&it[rank]);
                    Comple64x2 temp5(&it[rank + 2]);
                    (temp0 + temp2).storeu(&it[0]);
                    (temp4 + temp6).storeu(&it[2]);
                    (temp1 + temp3).storeu(&it[rank]);
                    (temp5 + temp7).storeu(&it[rank + 2]);
                    (temp0 - temp2).storeu(&it[rank * 2]);
                    (temp4 - temp6).storeu(&it[rank * 2 + 2]);
                    (temp1 - temp3).storeu(&it[rank * 3]);
                    (temp5 - temp7).storeu(&it[rank * 3 + 2]);
                }
                template <size_t rank, typename It, typename OmegaIt>
                inline void dif_butterfly_avx(It it, OmegaIt omega1_it, OmegaIt omega3_it)
                {
                    // Comple64x2 temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
                    Comple64x2 omega0, omega1, omega2, omega3;
                    Comple64x2 temp0(&it[0]);
                    Comple64x2 temp4(&it[2]);
                    Comple64x2 temp1(&it[rank]);
                    Comple64x2 temp5(&it[rank + 2]);
                    Comple64x2 temp2(&it[rank * 2]);
                    Comple64x2 temp6(&it[rank * 2 + 2]);
                    Comple64x2 temp3(&it[rank * 3]);
                    Comple64x2 temp7(&it[rank * 3 + 2]);
                    (temp0 + temp2).storeu(&it[0]);
                    (temp4 + temp6).storeu(&it[2]);
                    (temp1 + temp3).storeu(&it[rank]);
                    (temp5 + temp7).storeu(&it[rank + 2]);

                    temp2 = temp0 - temp2;
                    temp3 = temp1 - temp3;
                    temp6 = temp4 - temp6;
                    temp7 = temp5 - temp7;
                    temp3 = temp3.mul_neg_j();
                    temp7 = temp7.mul_neg_j();
                    transform2(temp2, temp3);
                    transform2(temp6, temp7);

                    omega0.load(&omega1_it[0]);
                    omega1.load(&omega1_it[2]);
                    omega2.load(&omega3_it[0]);
                    omega3.load(&omega3_it[2]);
                    (temp2 * omega0).storeu(&it[rank * 2]);
                    (temp3 * omega2).storeu(&it[rank * 3]);
                    (temp6 * omega1).storeu(&it[rank * 2 + 2]);
                    (temp7 * omega3).storeu(&it[rank * 3 + 2]);
                }

                inline void fft_dit_4point_avx(std::complex<Float64> *input)
                {
                    static const __m256d neg_mask = _mm256_castsi256_pd(
                        _mm256_set_epi64x(INT64_MIN, 0, 0, 0));
                    __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
                    __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3

                    __m256d tmp2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0,c2
                    __m256d tmp3 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c1,c3

                    tmp0 = _mm256_add_pd(tmp2, tmp3); // c0+c1,c2+c3
                    tmp1 = _mm256_sub_pd(tmp2, tmp3); // c0-c1,c2-c3

                    tmp2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0+c1,c0-c1;(A,B)
                    tmp3 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c2+c3,c2-c3

                    tmp3 = _mm256_permute_pd(tmp3, 0b0110);
                    tmp3 = _mm256_xor_pd(tmp3, neg_mask); // (C,D)

                    tmp0 = _mm256_add_pd(tmp2, tmp3); // A+C,B+D
                    tmp1 = _mm256_sub_pd(tmp2, tmp3); // A-C,B-D

                    _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp0);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp1);
                }
                inline void fft_dit_8point_avx(std::complex<Float64> *input)
                {
                    static const __m256d neg_mask = _mm256_castsi256_pd(_mm256_set_epi64x(INT64_MIN, 0, 0, 0));
                    static const __m256d mul1 = _mm256_set_pd(0.70710678118654752440084436210485, 0.70710678118654752440084436210485, 0, 0);
                    static const __m256d mul2 = _mm256_set_pd(-0.70710678118654752440084436210485, -0.70710678118654752440084436210485, -1, 1);
                    __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
                    __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3
                    __m256d tmp2 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 4)); // c0,c1
                    __m256d tmp3 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 6)); // c2,c3

                    __m256d tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0,c2
                    __m256d tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c1,c3
                    __m256d tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // c0,c2
                    __m256d tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // c1,c3

                    tmp0 = _mm256_add_pd(tmp4, tmp5); // c0+c1,c2+c3
                    tmp1 = _mm256_sub_pd(tmp4, tmp5); // c0-c1,c2-c3
                    tmp2 = _mm256_add_pd(tmp6, tmp7); // c0+c1,c2+c3
                    tmp3 = _mm256_sub_pd(tmp6, tmp7); // c0-c1,c2-c3

                    tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0+c1,c0-c1;(A,B)
                    tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c2+c3,c2-c3
                    tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // c0+c1,c0-c1;(A,B)
                    tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // c2+c3,c2-c3

                    tmp5 = _mm256_permute_pd(tmp5, 0b0110);
                    tmp5 = _mm256_xor_pd(tmp5, neg_mask); // (C,D)
                    tmp7 = _mm256_permute_pd(tmp7, 0b0110);
                    tmp7 = _mm256_xor_pd(tmp7, neg_mask); // (C,D)

                    tmp0 = _mm256_add_pd(tmp4, tmp5); // A+C,B+D
                    tmp1 = _mm256_sub_pd(tmp4, tmp5); // A-C,B-D
                    tmp2 = _mm256_add_pd(tmp6, tmp7); // A+C,B+D
                    tmp3 = _mm256_sub_pd(tmp6, tmp7); // A-C,B-D

                    // 2X4point-done
                    tmp6 = _mm256_permute_pd(tmp2, 0b0110);
                    tmp6 = _mm256_addsub_pd(tmp6, tmp2);
                    tmp6 = _mm256_permute_pd(tmp6, 0b0110);
                    tmp6 = _mm256_mul_pd(tmp6, mul1);
                    tmp2 = _mm256_blend_pd(tmp2, tmp6, 0b1100);

                    tmp7 = _mm256_permute_pd(tmp3, 0b0101);
                    tmp3 = _mm256_addsub_pd(tmp3, tmp7);
                    tmp3 = _mm256_blend_pd(tmp7, tmp3, 0b1100);
                    tmp3 = _mm256_mul_pd(tmp3, mul2);

                    tmp4 = _mm256_add_pd(tmp0, tmp2);
                    tmp5 = _mm256_add_pd(tmp1, tmp3);
                    tmp6 = _mm256_sub_pd(tmp0, tmp2);
                    tmp7 = _mm256_sub_pd(tmp1, tmp3);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp4);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp5);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 4), tmp6);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 6), tmp7);
                }
                inline void fft_dif_4point_avx(std::complex<Float64> *input)
                {
                    __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
                    __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3

                    __m256d tmp2 = _mm256_add_pd(tmp0, tmp1); // c0+c2,c1+c3;
                    __m256d tmp3 = _mm256_sub_pd(tmp0, tmp1); // c0-c2,c1-c3;
                    tmp3 = _mm256_permute_pd(tmp3, 0b0110);   // c0-c2,r(c1-c3);

                    static const __m256d neg_mask = _mm256_castsi256_pd(
                        _mm256_set_epi64x(INT64_MIN, 0, 0, 0));
                    tmp3 = _mm256_xor_pd(tmp3, neg_mask);

                    tmp0 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // A,C
                    tmp1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // B,D

                    tmp2 = _mm256_add_pd(tmp0, tmp1); // A+B,C+D
                    tmp3 = _mm256_sub_pd(tmp0, tmp1); // A-B,C-D

                    tmp0 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
                    tmp1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

                    _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp0);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp1);
                }
                inline void fft_dif_8point_avx(std::complex<Float64> *input)
                {
                    static const __m256d neg_mask = _mm256_castsi256_pd(_mm256_set_epi64x(INT64_MIN, 0, 0, 0));
                    static const __m256d mul1 = _mm256_set_pd(0.70710678118654752440084436210485, 0.70710678118654752440084436210485, 0, 0);
                    static const __m256d mul2 = _mm256_set_pd(-0.70710678118654752440084436210485, -0.70710678118654752440084436210485, -1, 1);
                    __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
                    __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3
                    __m256d tmp2 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 4)); // c4,c5
                    __m256d tmp3 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 6)); // c6,c7

                    __m256d tmp4 = _mm256_add_pd(tmp0, tmp2);
                    __m256d tmp5 = _mm256_add_pd(tmp1, tmp3);
                    __m256d tmp6 = _mm256_sub_pd(tmp0, tmp2);
                    __m256d tmp7 = _mm256_sub_pd(tmp1, tmp3);

                    tmp2 = _mm256_permute_pd(tmp6, 0b0110);
                    tmp2 = _mm256_addsub_pd(tmp2, tmp6);
                    tmp2 = _mm256_permute_pd(tmp2, 0b0110);
                    tmp2 = _mm256_mul_pd(tmp2, mul1);
                    tmp6 = _mm256_blend_pd(tmp6, tmp2, 0b1100);

                    tmp3 = _mm256_permute_pd(tmp7, 0b0101);
                    tmp7 = _mm256_addsub_pd(tmp7, tmp3);
                    tmp7 = _mm256_blend_pd(tmp3, tmp7, 0b1100);
                    tmp7 = _mm256_mul_pd(tmp7, mul2);

                    // 2X4point
                    tmp0 = _mm256_add_pd(tmp4, tmp5);
                    tmp1 = _mm256_sub_pd(tmp4, tmp5);
                    tmp1 = _mm256_permute_pd(tmp1, 0b0110);
                    tmp1 = _mm256_xor_pd(tmp1, neg_mask);

                    tmp2 = _mm256_add_pd(tmp6, tmp7);
                    tmp3 = _mm256_sub_pd(tmp6, tmp7);
                    tmp3 = _mm256_permute_pd(tmp3, 0b0110);
                    tmp3 = _mm256_xor_pd(tmp3, neg_mask);

                    tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
                    tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
                    tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
                    tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

                    tmp0 = _mm256_add_pd(tmp4, tmp5);
                    tmp1 = _mm256_sub_pd(tmp4, tmp5);
                    tmp2 = _mm256_add_pd(tmp6, tmp7);
                    tmp3 = _mm256_sub_pd(tmp6, tmp7);

                    tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
                    tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
                    tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
                    tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

                    _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp4);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp5);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 4), tmp6);
                    _mm256_storeu_pd(reinterpret_cast<double *>(input + 6), tmp7);
                }
                template <size_t LEN, typename FloatTy>
                struct FFT
                {
                    static_assert(std::is_same<Float64, FloatTy>::value, "AVX Only for double");
                    static constexpr size_t fft_len = LEN;
                    static constexpr size_t half_len = fft_len / 2;
                    static constexpr size_t quarter_len = fft_len / 4;
                    static constexpr size_t oct_len = fft_len / 8;
                    static constexpr int log_len = hint_log2(fft_len);

                    using HalfFFT = FFT<half_len, FloatTy>;
                    using QuarterFFT = FFT<quarter_len, FloatTy>;
                    using TableTy = FFTTableSplitRadix<FloatTy, log_len>;
                    using DataTy = std::complex<FloatTy>;
                    static void init()
                    {
                        TableTy::init();
                    }

                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        using value_type = typename std::iterator_traits<ComplexIt>::value_type;
                        static_assert(std::is_same<value_type, DataTy>::value, "Must be same as the FFT template float type");

                        QuarterFFT::dit(in_out + half_len + quarter_len);
                        QuarterFFT::dit(in_out + half_len);
                        HalfFFT::dit(in_out);

                        static auto const omega1_it = reinterpret_cast<const DataTy *>(TableTy::get_it1());
                        static auto const omega3_it = reinterpret_cast<const DataTy *>(TableTy::get_it3());
                        for (size_t i = 0; i < quarter_len; i += 4)
                        {
                            dit_butterfly_avx<quarter_len>(in_out + i, omega1_it + i, omega3_it + i);
                        }
                    }

                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        using value_type = typename std::iterator_traits<ComplexIt>::value_type;
                        static_assert(std::is_same<value_type, DataTy>::value, "Must be same as the FFT template float type");

                        static auto const omega1_it = reinterpret_cast<const DataTy *>(TableTy::get_it1());
                        static auto const omega3_it = reinterpret_cast<const DataTy *>(TableTy::get_it3());
                        for (size_t i = 0; i < quarter_len; i += 4)
                        {
                            dif_butterfly_avx<quarter_len>(in_out + i, omega1_it + i, omega3_it + i);
                        }
                        HalfFFT::dif(in_out);
                        QuarterFFT::dif(in_out + half_len);
                        QuarterFFT::dif(in_out + half_len + quarter_len);
                    }
                };

                template <typename FloatTy>
                struct FFT<0, FloatTy>
                {
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out) {}
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out) {}
                };

                template <typename FloatTy>
                struct FFT<1, FloatTy>
                {
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out) {}
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out) {}
                };

                template <typename FloatTy>
                struct FFT<2, FloatTy>
                {
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        transform2(in_out[0], in_out[1]);
                    }
                };

                template <typename FloatTy>
                struct FFT<4, FloatTy>
                {
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        fft_dit_4point_avx(&in_out[0]);
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        fft_dif_4point_avx(&in_out[0]);
                    }
                };

                template <typename FloatTy>
                struct FFT<8, FloatTy>
                {
                    using Complex = std::complex<FloatTy>;

                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        fft_dit_8point_avx(&in_out[0]);
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        fft_dif_8point_avx(&in_out[0]);
                    }
                };

                template <typename FloatTy>
                struct FFT<16, FloatTy>
                {
                    using Complex = std::complex<FloatTy>;
                    static constexpr Complex w1{COS_16, -SIN_16};
                    static constexpr Complex w3{SIN_16, -COS_16};
                    static constexpr Complex w9{-COS_16, SIN_16};
                    alignas(64) static constexpr Complex omega1_table[4] = {Complex(1), w1, Complex(COS_8, -COS_8), w3};
                    alignas(64) static constexpr Complex omega3_table[4] = {Complex(1), w3, Complex(-COS_8, -COS_8), w9};
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        FFT<8, FloatTy>::dit(in_out);
                        FFT<4, FloatTy>::dit(in_out + 8);
                        FFT<4, FloatTy>::dit(in_out + 12);
                        dit_butterfly_avx<4>(in_out, omega1_table, omega3_table);
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        dif_butterfly_avx<4>(in_out, omega1_table, omega3_table);
                        FFT<8, FloatTy>::dif(in_out);
                        FFT<4, FloatTy>::dif(in_out + 8);
                        FFT<4, FloatTy>::dif(in_out + 12);
                    }
                };
                template <typename FloatTy>
                constexpr std::complex<FloatTy> FFT<16, FloatTy>::w1;
                template <typename FloatTy>
                constexpr std::complex<FloatTy> FFT<16, FloatTy>::w3;
                template <typename FloatTy>
                constexpr std::complex<FloatTy> FFT<16, FloatTy>::w9;
                template <typename FloatTy>
                alignas(64) constexpr std::complex<FloatTy> FFT<16, FloatTy>::omega1_table[4];
                template <typename FloatTy>
                alignas(64) constexpr std::complex<FloatTy> FFT<16, FloatTy>::omega3_table[4];
            }

            // 默认FFT为分裂基
            template <size_t len, typename FloatTy>
            using FFTDefault = split_radix_avx::FFT<len, FloatTy>;

            /// @brief 初始化所有FFT查找表
            /// @tparam FloatTy
            template <typename FloatTy>
            inline void fft_init()
            {
                FFTDefault<FFT_MAX_LEN, FloatTy>::init();
            }

            // 获取FFT查找表
            template <size_t LEN, typename FloatTy>
            struct FFTLUT
            {
                using TABLE = typename FFTDefault<LEN, FloatTy>::TableTy;
                using It = decltype(TABLE::get_it1());
                static It get_fft_table(size_t len)
                {
                    if (len > LEN)
                    {
                        return FFTLUT<LEN * 2, FloatTy>::get_fft_table(len);
                    }
                    FFTDefault<LEN, FloatTy>::init();
                    return TABLE::get_it1();
                }
            };
            template <typename FloatTy>
            struct FFTLUT<1, FloatTy>
            {
                static auto get_fft_table(size_t len) { return nullptr; }
            };
            template <typename FloatTy>
            struct FFTLUT<2, FloatTy>
            {
                static auto get_fft_table(size_t len) { return nullptr; }
            };
            template <typename FloatTy>
            struct FFTLUT<4, FloatTy>
            {
                static auto get_fft_table(size_t len) { return nullptr; }
            };
            template <typename FloatTy>
            struct FFTLUT<8, FloatTy>
            {
                static auto get_fft_table(size_t len) { return nullptr; }
            };
            template <typename FloatTy>
            struct FFTLUT<16, FloatTy>
            {
                static auto get_fft_table(size_t len) { return nullptr; }
            };
            template <typename FloatTy>
            struct FFTLUT<FFT_MAX_LEN * 2, FloatTy>
            {
                static auto get_fft_table(size_t len)
                {
                    throw("Length of FFT can't be larger than FFT_MAX_LEN");
                    return nullptr;
                }
            };

            template <typename FloatTy>
            inline auto get_fft_table(size_t len)
            {
                return FFTLUT<size_t(1) << 5, FloatTy>::get_fft_table(len);
            }

            // 辅助选择类
            template <size_t LEN, typename FloatTy>
            struct FFTAlt
            {
                static void dit(std::complex<FloatTy> in_out[], size_t fft_len)
                {
                    if (fft_len > LEN)
                    {
                        FFTAlt<LEN * 2, FloatTy>::dit(in_out, fft_len);
                        return;
                    }
                    FFTDefault<LEN, Float64>::init();
                    FFTDefault<LEN, Float64>::dit(in_out);
                }
                static void dif(std::complex<FloatTy> in_out[], size_t fft_len)
                {
                    if (fft_len > LEN)
                    {
                        FFTAlt<LEN * 2, FloatTy>::dif(in_out, fft_len);
                        return;
                    }
                    FFTDefault<LEN, Float64>::init();
                    FFTDefault<LEN, Float64>::dif(in_out);
                }
            };
            template <typename FloatTy>
            struct FFTAlt<FFT_MAX_LEN * 2, FloatTy>
            {
                static void dit(std::complex<FloatTy> in_out[], size_t len)
                {
                    throw("Length of FFT can't be larger than FFT_MAX_LEN");
                }
                static void dif(std::complex<FloatTy> in_out[], size_t len)
                {
                    throw("Length of FFT can't be larger than FFT_MAX_LEN");
                }
            };

            // 时间抽取快速傅里叶变换
            template <typename FloatTy>
            inline void fft_dit(std::complex<FloatTy> in_out[], size_t fft_len)
            {
                FFTAlt<1, FloatTy>::dit(in_out, fft_len);
            }
            // 时间抽取快速傅里叶变换
            template <typename FloatTy>
            inline void fft_dif(std::complex<FloatTy> in_out[], size_t fft_len)
            {
                FFTAlt<1, FloatTy>::dif(in_out, fft_len);
            }

            // 离散傅里叶变换
            template <typename FloatTy>
            inline void dft(std::complex<FloatTy> in_out[], size_t len)
            {
                fft_dif(in_out, len);
                binary_reverse_swap(in_out, len);
            }
            // 离散傅里叶逆变换
            template <typename FloatTy>
            inline void idft(std::complex<FloatTy> in_out[], size_t len)
            {
                for (size_t i = 0; i < len; i++)
                {
                    in_out[i] = std::conj(in_out[i]);
                }
                dft(in_out, len);
                Float64 inv = Float64(1) / len;
                for (size_t i = 0; i < len; i++)
                {
                    in_out[i] = std::conj(in_out[i]) * inv;
                }
            }

            // 实数快速傅里叶变换
            template <typename FloatTy>
            inline void dft_real(const FloatTy in[], std::complex<FloatTy> out[], size_t len)
            {
                using Complex = std::complex<FloatTy>;
                len = int_floor2(len);
                if (len <= 16)
                {
                    size_t i = len;
                    while (i > 0)
                    {
                        i--;
                        out[i] = in[i];
                    }
                    dft(out, len);
                    return;
                }
                std::copy(in, in + len, reinterpret_cast<FloatTy *>(out));
                size_t fft_len = len / 2;
                dft(out, fft_len);

                Complex temp = out[0];
                out[0] = temp.real() + temp.imag();
                out[fft_len] = temp.real() - temp.imag();

                temp = out[fft_len / 2];
                out[fft_len / 2] = std::conj(temp);
                out[fft_len + fft_len / 2] = temp;
                auto omega_it = get_fft_table<FloatTy>(len);
                for (auto it0 = out + 1, it1 = out + fft_len - 1; it0 < out + fft_len / 2; it0++, it1--)
                {
                    Complex temp0 = it0[0] * 0.5, temp1 = std::conj(it1[0]) * 0.5;
                    Complex temp2 = temp0 + temp1, temp3 = temp0 - temp1;
                    temp1 = temp3;
                    temp3 = mul_neg_j(temp3);
                    temp0 = std::conj(temp2);

                    temp3 *= omega_it[it0 - out];
                    it0[0] = temp2 + temp3;
                    it0[fft_len] = temp2 - temp3;

                    temp1 = mul_conj(omega_it[it1 - out - fft_len / 2], temp1);
                    it1[0] = temp0 + temp1;
                    it1[fft_len] = temp0 - temp1;
                }
            }

            // 实数快速傅里逆叶变换
            template <typename FloatTy>
            inline void idft_real(const std::complex<FloatTy> in[], FloatTy out[], size_t len)
            {
                using Complex = std::complex<FloatTy>;
                len = int_floor2(len);
                if (len <= 16)
                {
                    std::vector<Complex> temp(in, in + len);
                    idft(temp.data(), len);
                    for (size_t i = 0; i < len; i++)
                    {
                        out[i] = temp[i].real();
                    }
                    return;
                }
                size_t fft_len = len / 2;
                auto omega_it = get_fft_table<FloatTy>(len);
                auto fft_ary = reinterpret_cast<Complex *>(out);
                for (size_t i = 1; i < fft_len / 2; i++)
                {
                    Complex temp0 = in[i] * FloatTy(0.5), temp1 = in[i + fft_len] * FloatTy(0.5);
                    Complex temp2 = temp0 + temp1, temp3 = temp0 - temp1;
                    temp3 = mul_conj(temp3, omega_it[i]);
                    fft_ary[i] = temp2 + mul_j(temp3);
                    fft_ary[fft_len - i] = std::conj(temp2) + mul_j(std::conj(temp3));
                }
                Complex temp0 = in[0].real(), temp1 = in[fft_len].real();
                transform2(temp0, temp1);
                fft_ary[0] = (temp0 + mul_j(temp1)) * 0.5;
                temp0 = in[fft_len / 2], temp1 = in[fft_len * 3 / 2];
                transform2(temp0, temp1);
                fft_ary[fft_len / 2] = (temp0 - temp1) * 0.5;
                idft(fft_ary, fft_len);
            }

            // FFT加速复数卷积
            template <typename FloatTy>
            inline void fft_convolution_complex(std::complex<FloatTy> fft_ary1[], std::complex<FloatTy> fft_ary2[], std::complex<FloatTy> out[], size_t fft_len)
            {
                if (fft_len == 0)
                {
                    return;
                }
                if (fft_len == 1)
                {
                    out[0] = fft_ary1[0] * fft_ary2[0];
                    return;
                }
                fft_len = int_floor2(fft_len);
                if (fft_len > FFT_MAX_LEN)
                {
                    throw("FFT len cannot be larger than FFT_MAX_LEN");
                }
                fft_dif(fft_ary1, fft_len);
                // 两个输入相同时只进行一次计算，提升平方速度
                if (fft_ary1 != fft_ary2)
                {
                    fft_dif(fft_ary2, fft_len);
                }
                const Float64 inv = 1.0 / fft_len;
                for (size_t i = 0; i < fft_len; i++)
                {
                    out[i] = std::conj(fft_ary1[i] * fft_ary2[i]) * inv;
                }
                fft_dit(out, fft_len);
                for (size_t i = 0; i < fft_len; i++)
                {
                    out[i] = std::conj(out[i]);
                }
            }

            // FFT加速实数卷积
            template <typename FloatTy>
            inline void fft_convolution_real(FloatTy ary1[], size_t len1, FloatTy ary2[], size_t len2, FloatTy out[])
            {
                using Complex = std::complex<FloatTy>;
                size_t conv_len = len1 + len2 - 1, fft_len = int_ceil2(conv_len);
                std::vector<Complex> fft_ary1_c(fft_len), fft_ary2_c(fft_len);
                std::copy(ary1, ary1 + len1, fft_ary1_c.begin());
                std::copy(ary2, ary2 + len2, fft_ary2_c.begin());
                fft_convolution_complex(fft_ary1_c.data(), fft_ary2_c.data(), fft_ary1_c.data(), fft_len);
                for (size_t i = 0; i < conv_len; i++)
                {
                    out[i] = fft_ary1_c[i].real();
                }
            }
        }
    }
}

using namespace std;
using namespace hint;
using namespace hint_transform;
using namespace hint_fft;

template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    vector<T> result(out_len);
    size_t fft_len = int_floor2(out_len);
    vector<Complex64> fft_ary1(fft_len);
    for (size_t i = 0; i < len1; i++)
    {
        fft_ary1[i].real(in1[i]);
    }
    for (size_t i = 0; i < len2; i++)
    {
        fft_ary1[i].imag(in2[i]);
    }
    fft_dif(fft_ary1.data(), fft_len);
    for (size_t i = 0; i < fft_len; i += 2)
    {
        Comple64x2 t(fft_ary1.data() + i);
        t = (t * t).conj();
        t.storeu(fft_ary1.data() + i);
    }
    fft_dit(fft_ary1.data(), fft_len);
    const Float64 inv = -0.5 / fft_len;
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = T(inv * fft_ary1[i].imag() + 0.5);
    }
    return result;
}

template <typename T>
void result_test(const vector<T> &res, uint64_t ele)
{
    size_t len = res.size();
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << (i + 1) * ele * ele << "\t" << y << "\n";
            return;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    cout << "success\n";
}

// FFT convolution
int main()
{
    fft_init<Float64>();
    int n = 18;
    cin >> n;
    size_t len = 1 << n; // 变换长度
    uint64_t ele = 5;
    vector<uint32_t> in1(len / 2, ele);
    vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    auto t1 = chrono::steady_clock::now();
    vector<uint32_t> res = poly_multiply(in1, in2);
    auto t2 = chrono::steady_clock::now();
    result_test<uint32_t>(res, ele); // 结果校验
    cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us\n";
}