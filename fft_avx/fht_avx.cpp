// TSKY 2024/2/12
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
#ifdef __AVX512F__
#pragma GCC target("avx512f")
    struct DoubleX8
    {
        __m512d data;
        DoubleX8()
        {
            data = _mm512_setzero_pd();
        }
        DoubleX8(double input)
        {
            data = _mm512_set1_pd(input);
        }
        DoubleX8(__m512d input)
        {
            data = input;
        }
        DoubleX8(const DoubleX8 &input)
        {
            data = input.data;
        }
        // 从连续的数组构造
        DoubleX8(double const *ptr)
        {
            load(ptr);
        }
        // 用4个数构造
        DoubleX8(double a7, double a6, double a5, double a4, double a3, double a2, double a1, double a0)
        {
            data = _mm512_set_pd(a7, a6, a5, a4, a3, a2, a1, a0);
        }
        void clr()
        {
            data = _mm512_setzero_pd();
        }
        void load(double const *ptr)
        {
            data = _mm512_load_pd(ptr);
        }
        void loadu(double const *ptr)
        {
            data = _mm512_loadu_pd(ptr);
        }
        void store(double *ptr) const
        {
            _mm512_store_pd(ptr, data);
        }
        void storeu(double *ptr) const
        {
            _mm512_storeu_pd(ptr, data);
        }
        void print() const
        {
            double ary[8];
            storeu(ary);
            printf("(%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf)\n",
                   ary[0], ary[1], ary[2], ary[3], ary[4], ary[5], ary[6], ary[7]);
        }
        template <int N>
        DoubleX8 permutex() const
        {
            return _mm512_permutex_pd(data, N);
        }
        template <int N>
        DoubleX8 permute() const
        {
            return _mm512_permute_pd(data, N);
        }
        template <int N>
        DoubleX8 shuffle(DoubleX8 in) const
        {
            return _mm512_shuffle_pd(data, in.data, N);
        }
        template <int N>
        DoubleX8 shuffle_f128(DoubleX8 in) const
        {
            return _mm512_shuffle_f64x2(data, in.data, N);
        }
        DoubleX8 swap_oe() const
        {
            return permute<0b01010101>();
        }
        DoubleX8 reverse() const
        {
            auto eo = swap_oe();
            return eo.shuffle_f128<0b00011011>(eo);
        }
        DoubleX8 fmadd(DoubleX8 mul1, DoubleX8 mul2) const
        {
            return _mm512_fmadd_pd(mul1.data, mul2.data, data);
        }
        DoubleX8 fmsub(DoubleX8 mul1, DoubleX8 mul2) const
        {
            return _mm512_fmsub_pd(mul1.data, mul2.data, data);
        }
        DoubleX8 operator+(DoubleX8 input) const
        {
            return _mm512_add_pd(data, input.data);
        }
        DoubleX8 operator-(DoubleX8 input) const
        {
            return _mm512_sub_pd(data, input.data);
        }
        DoubleX8 operator*(DoubleX8 input) const
        {
            return _mm512_mul_pd(data, input.data);
        }
        DoubleX8 operator/(DoubleX8 input) const
        {
            return _mm512_div_pd(data, input.data);
        }
        DoubleX8 operator-() const
        {
            return _mm512_sub_pd(_mm512_setzero_pd(), data);
        }
    };
#else
    struct DoubleX8
    {
        DoubleX4 data0;
        DoubleX4 data1;
        DoubleX8()
        {
            data0 = data1 = _mm256_setzero_pd();
        }
        DoubleX8(DoubleX4 in0, DoubleX4 in1) : data0(in0), data1(in1) {}
        DoubleX8(double input)
        {
            data0 = data1 = _mm256_set1_pd(input);
        }
        DoubleX8(const DoubleX8 &input)
        {
            data0 = input.data0;
            data1 = input.data1;
        }
        // 从连续的数组构造
        DoubleX8(double const *ptr)
        {
            loadu(ptr);
        }
        // 用4个数构造
        DoubleX8(double a7, double a6, double a5, double a4, double a3, double a2, double a1, double a0)
        {
            data1 = _mm256_set_pd(a7, a6, a5, a4);
            data0 = _mm256_set_pd(a3, a2, a1, a0);
        }
        void clr()
        {
            data0 = data1 = _mm256_setzero_pd();
        }
        void load(double const *ptr)
        {
            data0.load(ptr);
            data1.load(ptr + 4);
        }
        void loadu(double const *ptr)
        {
            data0.loadu(ptr);
            data1.loadu(ptr + 4);
        }
        void store(double *ptr) const
        {
            data0.store(ptr);
            data1.store(ptr + 4);
        }
        void storeu(double *ptr) const
        {
            data0.storeu(ptr);
            data1.storeu(ptr + 4);
        }
        void print() const
        {
            double ary[8];
            storeu(ary);
            printf("(%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf)\n",
                   ary[0], ary[1], ary[2], ary[3], ary[4], ary[5], ary[6], ary[7]);
        }
        DoubleX8 reverse() const
        {
            return DoubleX8(data1.reverse(), data0.reverse());
        }
        DoubleX8 fmadd(DoubleX8 mul1, DoubleX8 mul2) const
        {
            return DoubleX8(data0.fmadd(mul1.data0, mul2.data0), data1.fmadd(mul1.data1, mul2.data1));
        }
        DoubleX8 fmsub(DoubleX8 mul1, DoubleX8 mul2) const
        {
            return DoubleX8(data0.fmsub(mul1.data0, mul2.data0), data1.fmsub(mul1.data1, mul2.data1));
        }
        DoubleX8 operator+(DoubleX8 input) const
        {
            return DoubleX8(data0 + input.data0, data1 + input.data1);
        }
        DoubleX8 operator-(DoubleX8 input) const
        {
            return DoubleX8(data0 - input.data0, data1 - input.data1);
        }
        DoubleX8 operator*(DoubleX8 input) const
        {
            return DoubleX8(data0 * input.data0, data1 * input.data1);
        }
        DoubleX8 operator/(DoubleX8 input) const
        {
            return DoubleX8(data0 / input.data0, data1 / input.data1);
        }
        DoubleX8 operator-() const
        {
            return DoubleX8(-data0, -data1);
        }
    };
#endif
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
    constexpr size_t FHT_MAX_LEN = size_t(1) << 23;

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

    // bits个二进制全为1的数,等于2^bits-1
    template <typename T>
    constexpr T all_one(int bits)
    {
        T tmp = T(1) << (bits - 1);
        return tmp - T(1) + tmp;
    }

    // 整数log2
    template <typename UintTy>
    constexpr int hint_log2(UintTy n)
    {
        constexpr int bits_2 = 4 * sizeof(UintTy);
        constexpr UintTy mask = all_one<UintTy>(bits_2) << bits_2;
        UintTy m = mask;
        int res = 0, shift = bits_2;
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

    // FFT与类FFT变换的命名空间
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
        class CosTableStatic
        {
        public:
            static constexpr size_t len = size_t(1) << log_len;
            static constexpr size_t table_len = len / len_div;
            static constexpr FloatTy unit = HINT_2PI / len;
            using Ty = FloatTy;
            using TableTy = hint_simd::AlignAry<Ty, table_len>;
            CosTableStatic() {}
            CosTableStatic(int factor) { init(factor); }
            void allocate() {}
            void init(int factor)
            {
                for (size_t i = 0; i < table.size(); i++)
                {
                    table[i] = std::cos(factor * i * unit);
                }
            }
            const auto &operator[](size_t n) const { return table[n]; }
            auto &operator[](size_t n) { return table[n]; }
            auto get_it(size_t n = 0) const { return &table[n]; }

        private:
            TableTy table;
        };

        namespace hint_fht
        {
            // 基2FHT AVX加速
            namespace radix2_avx
            {
                using hint_simd::DoubleX4;
                using hint_simd::DoubleX8;

                template <typename FloatTy, int log_len>
                class FHTTableRadix2
                {
                public:
                    using HalfTable = FHTTableRadix2<FloatTy, log_len - 1>;
                    using TableTy = CosTableStatic<FloatTy, log_len, 4>;

                    static constexpr int factor = HalfTable::factor;
                    FHTTableRadix2()
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
                        constexpr size_t table_len = table.table_len;
                        table.allocate();
                        for (size_t i = 0; i < table_len; i += 2)
                        {
                            table[i] = HalfTable::table[i / 2];
                            table[i + 1] = std::cos(FloatTy(i + 1) * factor * table.unit);
                        }
                        has_init = true;
                    }
                    static auto get_it(size_t n = 0) { return table.get_it(n); }

                    static TableTy table;

                private:
                    static bool has_init;
                };

                template <typename FloatTy, int log_len>
                typename FHTTableRadix2<FloatTy, log_len>::TableTy
                    FHTTableRadix2<FloatTy, log_len>::table;
                template <typename FloatTy, int log_len>
                bool FHTTableRadix2<FloatTy, log_len>::has_init = false;

                template <typename FloatTy>
                class FHTTableRadix2<FloatTy, 4>
                {
                public:
                    using TableTy = CosTableStatic<FloatTy, 4, 4>;
                    static constexpr int factor = 1;
                    FHTTableRadix2() { init(); }
                    static void init()
                    {
                        table.init(factor);
                    }
                    static auto get_it(size_t n = 0) { return table.get_it(n); }

                    static TableTy table;
                };
                template <typename FloatTy>
                typename FHTTableRadix2<FloatTy, 4>::TableTy FHTTableRadix2<FloatTy, 4>::table;

                template <size_t LEN, typename FloatTy>
                struct FHT
                {
                    static_assert(std::is_same<Float64, FloatTy>::value, "AVX Only for double");
                    static constexpr size_t fht_len = LEN;
                    static constexpr size_t half_len = fht_len / 2;
                    static constexpr size_t quarter_len = fht_len / 4;
                    static constexpr int log_len = hint_log2(fht_len);

                    using HalfFHT = FHT<half_len, FloatTy>;
                    using TableTy = FHTTableRadix2<FloatTy, log_len>;

                    static void init()
                    {
                        TableTy::init();
                    }

                    template <typename FloatIt>
                    static void dit(FloatIt in_out)
                    {
                        static_assert(std::is_same<typename std::iterator_traits<FloatIt>::value_type, FloatTy>::value, "Must be same as the FHT template float type");

                        HalfFHT::dit(in_out + half_len);
                        HalfFHT::dit(in_out);
                        transform2(in_out[0], in_out[half_len]);
                        transform2(in_out[quarter_len], in_out[half_len + quarter_len]);

                        auto it0 = in_out + 1, it1 = in_out + half_len - 1;
                        auto it2 = it0 + half_len, it3 = it1 + half_len;
                        auto cos_it = TableTy::get_it(1), sin_it = TableTy::get_it(TableTy::table.table_len - 1);
                        for (; it0 < in_out + 4; ++it0, --it1, ++it2, --it3, cos_it++, sin_it--)
                        {
                            auto c = cos_it[0], s = sin_it[0];
                            auto temp0 = it2[0], temp1 = it3[0];
                            auto temp2 = temp0 * c + temp1 * s; //+*+ -(-*-)
                            auto temp3 = temp0 * s - temp1 * c; //-(+)*- -*-(+)
                            temp0 = it0[0], temp1 = it1[0];
                            it0[0] = temp0 + temp2; //+
                            it1[0] = temp1 + temp3; //-
                            it2[0] = temp0 - temp2; //+
                            it3[0] = temp1 - temp3; //-
                        }
                        it1 -= 3, it3 -= 3, sin_it -= 3;
                        {
                            DoubleX4 c4, s4, temp0, temp1, temp2, temp3;
                            c4.loadu(&cos_it[0]), s4.loadu(&sin_it[0]);
                            temp0.loadu(&it2[0]), temp1.loadu(&it3[0]);
                            // temp2 = (temp1 * s4).reverse().fmadd(temp0, c4);
                            temp2 = (temp0 * c4).fmadd(temp1.reverse(), s4.reverse());
                            temp3 = (c4.reverse() * temp1).fmsub(temp0.reverse(), s4);
                            temp0.loadu(&it0[0]), temp1.loadu(&it1[0]);
                            (temp0 + temp2).storeu(&it0[0]);
                            (temp1 + temp3).storeu(&it1[0]);
                            (temp0 - temp2).storeu(&it2[0]);
                            (temp1 - temp3).storeu(&it3[0]);
                        }
                        it0 += 4, it2 += 4, cos_it += 4, it1 -= 8, it3 -= 8, sin_it -= 8;
                        for (; it0 < in_out + quarter_len; it0 += 8, it1 -= 8, it2 += 8, it3 -= 8, cos_it += 8, sin_it -= 8)
                        {
                            DoubleX8 c4, s4, temp0, temp1, temp2, temp3;
                            c4.loadu(&cos_it[0]), s4.loadu(&sin_it[0]);
                            temp0.loadu(&it2[0]), temp1.loadu(&it3[0]);
                            // temp2 = (temp1 * s4).reverse().fmadd(temp0, c4);
                            temp2 = (temp0 * c4).fmadd(temp1.reverse(), s4.reverse());
                            temp3 = (c4.reverse() * temp1).fmsub(temp0.reverse(), s4);
                            temp0.loadu(&it0[0]), temp1.loadu(&it1[0]);
                            (temp0 + temp2).storeu(&it0[0]);
                            (temp1 + temp3).storeu(&it1[0]);
                            (temp0 - temp2).storeu(&it2[0]);
                            (temp1 - temp3).storeu(&it3[0]);
                        }
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        static_assert(std::is_same<typename std::iterator_traits<FloatIt>::value_type, FloatTy>::value, "Must be same as the FHT template float type");

                        auto it0 = in_out + 1, it1 = in_out + half_len - 1;
                        auto it2 = it0 + half_len, it3 = it1 + half_len;
                        auto cos_it = TableTy::get_it(1), sin_it = TableTy::get_it(TableTy::table.table_len - 1);
                        for (; it0 < in_out + 4; ++it0, --it1, ++it2, --it3, cos_it++, sin_it--)
                        {
                            auto c = cos_it[0], s = sin_it[0];   //+,-
                            auto temp0 = it0[0], temp1 = it1[0]; //+,-
                            auto temp2 = it2[0], temp3 = it3[0]; //+,-
                            it0[0] = temp0 + temp2;              //+
                            it1[0] = temp1 + temp3;              //-
                            temp0 = temp0 - temp2;               //+
                            temp1 = temp1 - temp3;               //-
                            it2[0] = temp0 * c + temp1 * s;      //+*+  -(-*-)
                            it3[0] = temp0 * s - temp1 * c;      //-(+)*- -*-(+)
                        }
                        it1 -= 3, it3 -= 3, sin_it -= 3;
                        {
                            DoubleX4 c4, s4, temp0, temp1, temp2, temp3;
                            temp0.loadu(&it0[0]), temp1.loadu(&it1[0]);
                            temp2.loadu(&it2[0]), temp3.loadu(&it3[0]);
                            (temp0 + temp2).storeu(&it0[0]);
                            (temp1 + temp3).storeu(&it1[0]);
                            temp0 = temp0 - temp2;
                            temp1 = temp1 - temp3;
                            c4.loadu(&cos_it[0]), s4.loadu(&sin_it[0]);
                            // temp2 = (temp1 * s4).reverse().fmadd(temp0, c4);
                            temp2 = (temp0 * c4).fmadd(temp1.reverse(), s4.reverse());
                            temp3 = (c4.reverse() * temp1).fmsub(temp0.reverse(), s4);
                            temp2.storeu(&it2[0]), temp3.storeu(&it3[0]);
                        }
                        it0 += 4, it2 += 4, cos_it += 4, it1 -= 8, it3 -= 8, sin_it -= 8;
                        for (; it0 < in_out + quarter_len; it0 += 8, it1 -= 8, it2 += 8, it3 -= 8, cos_it += 8, sin_it -= 8)
                        {
                            DoubleX8 c4, s4, temp0, temp1, temp2, temp3;
                            temp0.loadu(&it0[0]), temp1.loadu(&it1[0]);
                            temp2.loadu(&it2[0]), temp3.loadu(&it3[0]);
                            (temp0 + temp2).storeu(&it0[0]);
                            (temp1 + temp3).storeu(&it1[0]);
                            temp0 = temp0 - temp2;
                            temp1 = temp1 - temp3;
                            c4.loadu(&cos_it[0]), s4.loadu(&sin_it[0]);
                            // temp2 = (temp1 * s4).reverse().fmadd(temp0, c4);
                            temp2 = (temp0 * c4).fmadd(temp1.reverse(), s4.reverse());
                            temp3 = (c4.reverse() * temp1).fmsub(temp0.reverse(), s4);
                            temp2.storeu(&it2[0]), temp3.storeu(&it3[0]);
                        }

                        transform2(in_out[0], in_out[half_len]);
                        transform2(in_out[quarter_len], in_out[half_len + quarter_len]);
                        HalfFHT::dif(in_out);
                        HalfFHT::dif(in_out + half_len);
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
            // 分裂基FHT AVX加速 无查找表
            namespace split_radix_avx
            {
                using hint_simd::DoubleX4;
                template <size_t LEN, typename FloatTy>
                struct FHT
                {
                    static_assert(std::is_same<Float64, FloatTy>::value, "AVX Only for double");
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
                    static void init() {}
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
                        static const DoubleX4 cos_unit3 = DoubleX4(cos_arr3[4]);
                        static const DoubleX4 sin_unit3 = DoubleX4(sin_arr3[4]);
                        DoubleX4 cos1_4(cos_arr1 + 4), sin1_4(sin_arr1 + 4);
                        DoubleX4 cos3_4 = DoubleX4(cos_arr3 + 4).reverse();
                        DoubleX4 sin3_4 = DoubleX4(sin_arr3 + 4).reverse();
                        for (; it0 < in_out + oct_len; it0 += 4, it1 -= 4, it2 += 4, it3 -= 4)
                        {
                            DoubleX4 temp0, temp1, temp2, temp3, temp4, temp5;
                            temp4.loadu(&it0[half_len]), temp5.loadu(&it1[half_len]);
                            temp5 = temp5.reverse();
                            temp0 = (temp4 * cos1_4) + (temp5 * sin1_4);
                            temp2 = (temp4 * sin1_4) - (temp5 * cos1_4);
                            temp4.loadu(&it2[half_len]), temp5.loadu(&it3[half_len]);
                            temp4 = temp4.reverse();
                            temp1 = (temp5 * sin3_4) + (temp4 * cos3_4);
                            temp3 = (temp4 * sin3_4) - (temp5 * cos3_4);

                            temp4 = temp0, temp5 = temp1;
                            temp0 = temp4 + temp5.reverse();
                            temp1 = temp4.reverse() - temp5;
                            temp4 = temp2, temp5 = temp3;
                            temp2 = temp5.reverse() - temp4;
                            temp3 = temp5 + temp4.reverse();

                            temp4.loadu(&it0[0]), temp5.loadu(&it1[0]);
                            (temp4 + temp0).storeu(&it0[0]);
                            (temp5 + temp1).storeu(&it1[0]);
                            (temp4 - temp0).storeu(&it0[half_len]);
                            (temp5 - temp1).storeu(&it1[half_len]);

                            temp4.loadu(&it2[0]), temp5.loadu(&it3[0]);
                            (temp4 + temp2).storeu(&it2[0]);
                            (temp5 + temp3).storeu(&it3[0]);
                            (temp4 - temp2).storeu(&it2[half_len]);
                            (temp5 - temp3).storeu(&it3[half_len]);

                            temp0 = cos1_4, temp1 = sin1_4;
                            cos1_4 = (temp0 * cos_unit1).fnmadd(temp1, sin_unit1);
                            sin1_4 = (temp0 * sin_unit1).fmadd(temp1, cos_unit1);
                            temp0 = cos3_4, temp1 = sin3_4;
                            cos3_4 = (temp0 * cos_unit3).fnmadd(temp1, sin_unit3);
                            sin3_4 = (temp0 * sin_unit3).fmadd(temp1, cos_unit3);
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
                        static const DoubleX4 cos_unit3 = DoubleX4(cos_arr3[4]);
                        static const DoubleX4 sin_unit3 = DoubleX4(sin_arr3[4]);
                        DoubleX4 cos1_4(cos_arr1 + 4), sin1_4(sin_arr1 + 4);
                        DoubleX4 cos3_4 = DoubleX4(cos_arr3 + 4).reverse();
                        DoubleX4 sin3_4 = DoubleX4(sin_arr3 + 4).reverse();
                        for (; it0 < in_out + oct_len; it0 += 4, it1 -= 4, it2 += 4, it3 -= 4)
                        {
                            DoubleX4 temp0, temp1, temp2, temp3, temp4, temp5;
                            temp0.loadu(&it0[half_len]), temp1.loadu(&it1[half_len]);
                            temp2.loadu(&it2[half_len]), temp3.loadu(&it3[half_len]);

                            temp4.loadu(&it0[0]), temp5.loadu(&it1[0]);
                            (temp4 + temp0).storeu(&it0[0]);
                            (temp5 + temp1).storeu(&it1[0]);
                            temp0 = temp4 - temp0, temp1 = temp5 - temp1;

                            temp4.loadu(&it2[0]), temp5.loadu(&it3[0]);
                            (temp4 + temp2).storeu(&it2[0]);
                            (temp5 + temp3).storeu(&it3[0]);
                            temp2 = temp4 - temp2, temp3 = temp5 - temp3;

                            temp4 = temp0, temp5 = temp1;
                            temp0 = temp4 + temp5.reverse();
                            temp1 = temp4.reverse() - temp5;
                            temp4 = temp2, temp5 = temp3;
                            temp2 = temp5.reverse() - temp4;
                            temp3 = temp5 + temp4.reverse();

                            (temp0 * cos1_4 + temp2 * sin1_4).storeu(&it0[half_len]);
                            (temp0 * sin1_4 - temp2 * cos1_4).reverse().storeu(&it1[half_len]);
                            (temp1 * cos3_4 + temp3 * sin3_4).reverse().storeu(&it2[half_len]);
                            (temp1 * sin3_4 - temp3 * cos3_4).storeu(&it3[half_len]);

                            temp0 = cos1_4, temp1 = sin1_4;
                            cos1_4 = (temp0 * cos_unit1).fnmadd(temp1, sin_unit1);
                            sin1_4 = (temp0 * sin_unit1).fmadd(temp1, cos_unit1);
                            temp0 = cos3_4, temp1 = sin3_4;
                            cos3_4 = (temp0 * cos_unit3).fnmadd(temp1, sin_unit3);
                            sin3_4 = (temp0 * sin_unit3).fmadd(temp1, cos_unit3);
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
            using FHTDefault = split_radix_avx::FHT<len, FloatTy>;

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
                FHTDefault<LEN, Float64>::init();
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
                FHTDefault<LEN, Float64>::init();
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

            // 离散哈特莱变换
            inline void dht(Float64 *in_out, size_t len)
            {
                binary_reverse_swap(in_out, len);
                fht_dit(in_out, len);
            }
            // 离散哈特莱逆变换
            inline void idht(Float64 *in_out, size_t len)
            {
                dht(in_out, len);
                Float64 inv = Float64(1) / len;
                for (size_t i = 0; i < len; i++)
                {
                    in_out[i] *= inv;
                }
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
template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    vector<T> result(out_len);
    size_t fht_len = hint::int_floor2(out_len);
    std::vector<double> buffer1(fht_len), buffer2(fht_len);
    std::copy(in1.begin(), in1.end(), buffer1.begin());
    std::copy(in2.begin(), in2.end(), buffer2.begin());
    hint::hint_transform::hint_fht::fht_convolution(buffer1.data(), buffer2.data(), buffer1.data(), fht_len);
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(buffer1[i] + 0.5);
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

int main()
{
    hint::hint_transform::hint_fht::fht_init<double>();
    int n = 10;
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