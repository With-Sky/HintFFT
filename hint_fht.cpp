// TSKY 2024/2/13
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
        return tmp - 1 + tmp;
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
        class CosTableDynamic
        {
        public:
            static constexpr size_t len = size_t(1) << log_len;
            static constexpr size_t table_len = len / len_div;
            static constexpr FloatTy unit = HINT_2PI / len;
            using Ty = FloatTy;
            using TableTy = std::vector<Ty>;
            CosTableDynamic() {}
            CosTableDynamic(int factor) { init(factor); }
            void allocate() { table.resize(table_len); }
            void init(int factor)
            {
                allocate();
                for (size_t i = 0; i < table.size(); i++)
                {
                    table[i] = std::cos(factor * unit * FloatTy(i));
                }
            }
            const auto &operator[](size_t n) const { return table[n]; }
            auto &operator[](size_t n) { return table[n]; }
            auto get_it(size_t n = 0) const { return &table[n]; }

        private:
            TableTy table;
        };

        template <typename FloatTy, int log_len, int len_div>
        class ComplexTableDynamic
        {
        public:
            static constexpr size_t len = size_t(1) << log_len;
            static constexpr size_t table_len = len / len_div;
            static constexpr FloatTy unit = HINT_2PI / len;
            using Ty = std::complex<FloatTy>;
            using TableTy = std::vector<Ty>;
            ComplexTableDynamic() {}
            ComplexTableDynamic(int factor) { init(factor); }
            void allocate() { table.resize(table_len); }
            void init(int factor)
            {
                allocate();
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

        namespace hint_fht
        {
            // 基2FHT
            namespace radix2
            {
                template <typename FloatTy, int log_len>
                class FHTTableRadix2
                {
                public:
                    using HalfTable = FHTTableRadix2<FloatTy, log_len - 1>;
                    using TableTy = CosTableDynamic<FloatTy, log_len, 4>;

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
                    using TableTy = CosTableDynamic<FloatTy, 4, 4>;
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
                    static constexpr size_t fht_len = LEN;
                    static constexpr size_t half_len = fht_len / 2;
                    static constexpr size_t quarter_len = fht_len / 4;
                    static constexpr int log_len = hint_log2(fht_len);

                    using HalfFHT = FHT<half_len, FloatTy>;
                    using TableTy = FHTTableRadix2<FloatTy, log_len>;
                    // static TableTy TABLE;
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
                        for (; it0 < it1; ++it0, --it1, ++it2, --it3, cos_it++, sin_it--)
                        {
                            auto c = cos_it[0], s = sin_it[0];
                            auto temp0 = it2[0], temp1 = it3[0];
                            auto temp2 = temp0 * c + temp1 * s;
                            auto temp3 = temp0 * s - temp1 * c;
                            temp0 = it0[0], temp1 = it1[0];
                            it0[0] = temp0 + temp2;
                            it1[0] = temp1 + temp3;
                            it2[0] = temp0 - temp2;
                            it3[0] = temp1 - temp3;
                        }
                    }
                    template <typename FloatIt>
                    static void dif(FloatIt in_out)
                    {
                        static_assert(std::is_same<typename std::iterator_traits<FloatIt>::value_type, FloatTy>::value, "Must be same as the FHT template float type");

                        auto it0 = in_out + 1, it1 = in_out + half_len - 1;
                        auto it2 = it0 + half_len, it3 = it1 + half_len;
                        auto cos_it = TableTy::get_it(1), sin_it = TableTy::get_it(TableTy::table.table_len - 1);
                        for (; it0 < it1; ++it0, --it1, ++it2, --it3, cos_it++, sin_it--)
                        {
                            auto c = cos_it[0], s = sin_it[0];
                            auto temp0 = it0[0], temp1 = it1[0]; //+,-
                            auto temp2 = it2[0], temp3 = it3[0]; //+,-
                            it0[0] = temp0 + temp2;              //+
                            it1[0] = temp1 + temp3;              //-
                            temp0 = temp0 - temp2;
                            temp1 = temp1 - temp3;
                            it2[0] = temp0 * c + temp1 * s;
                            it3[0] = temp0 * s - temp1 * c;
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
                        transform2(temp5, temp7);

                        in_out[0] = temp0 + temp4;
                        in_out[2] = temp2 + temp6;
                        in_out[4] = temp0 - temp4;
                        in_out[6] = temp2 - temp6;
                        static constexpr decltype(temp0) SQRT_2_2 = 0.70710678118654757;
                        temp0 = (temp5 + temp7) * SQRT_2_2;
                        temp2 = (temp5 - temp7) * SQRT_2_2;
                        in_out[1] = temp1 + temp0;
                        in_out[3] = temp3 + temp2;
                        in_out[5] = temp1 - temp0;
                        in_out[7] = temp3 - temp2;
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
                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        in_out[0] = temp0 + temp2;
                        in_out[1] = temp1 + temp3;
                        in_out[2] = temp0 - temp2;
                        in_out[3] = temp1 - temp3;
                        static constexpr decltype(temp0) SQRT_2_2 = 0.70710678118654757;
                        temp0 = (temp5 + temp7) * SQRT_2_2;
                        temp2 = (temp5 - temp7) * SQRT_2_2;
                        transform2(temp4, temp6);
                        transform2(temp0, temp2);
                        in_out[4] = temp4 + temp0;
                        in_out[5] = temp4 - temp0;
                        in_out[6] = temp6 + temp2;
                        in_out[7] = temp6 - temp2;
                    }
                };
            };

            namespace split_radix
            {
                template <typename FloatTy, int log_len>
                class FHTTableSplitRadix
                {
                public:
                    using HalfTable = FHTTableSplitRadix<FloatTy, log_len - 1>;
                    using TableTy = ComplexTableDynamic<FloatTy, log_len, 8>;

                    static constexpr int factor1 = HalfTable::factor1;
                    static constexpr int factor3 = HalfTable::factor3;

                    FHTTableSplitRadix()
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
                typename FHTTableSplitRadix<FloatTy, log_len>::TableTy
                    FHTTableSplitRadix<FloatTy, log_len>::table1;
                template <typename FloatTy, int log_len>
                typename FHTTableSplitRadix<FloatTy, log_len>::TableTy
                    FHTTableSplitRadix<FloatTy, log_len>::table3;
                template <typename FloatTy, int log_len>
                bool FHTTableSplitRadix<FloatTy, log_len>::has_init = false;

                template <typename FloatTy>
                class FHTTableSplitRadix<FloatTy, 5>
                {
                public:
                    using TableTy = ComplexTableDynamic<FloatTy, 5, 8>;

                    static constexpr int factor1 = 1;
                    static constexpr int factor3 = 3;

                    FHTTableSplitRadix() { init(); }
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
                typename FHTTableSplitRadix<FloatTy, 5>::TableTy FHTTableSplitRadix<FloatTy, 5>::table1;
                template <typename FloatTy>
                typename FHTTableSplitRadix<FloatTy, 5>::TableTy FHTTableSplitRadix<FloatTy, 5>::table3;

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
                    using TableTy = FHTTableSplitRadix<FloatTy, log_len>;
                    // static TableTy TABLE;
                    static void init()
                    {
                        TableTy::init();
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
                        auto omega1_it = TableTy::get_it1() + 1, omega3_it = TableTy::get_it3() + 1;
                        for (; it0 < it1; it0++, it1--, it2++, it3--, omega1_it++, omega3_it++)
                        {
                            auto omega1 = omega1_it[0], omega3 = omega3_it[0];
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
                        auto omega1_it = TableTy::get_it1() + 1, omega3_it = TableTy::get_it3() + 1;
                        for (; it0 < it1; it0++, it1--, it2++, it3--, omega1_it++, omega3_it++)
                        {
                            auto omega1 = omega1_it[0], omega3 = omega3_it[0];
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

            // 辅助选择类
            template <size_t LEN, typename FloatTy>
            struct FHTAlt
            {
                static void dit(FloatTy in_out[], size_t fft_len)
                {
                    if (fft_len > LEN)
                    {
                        FHTAlt<LEN * 2, FloatTy>::dit(in_out, fft_len);
                        return;
                    }
                    FHTDefault<LEN, Float64>::init();
                    FHTDefault<LEN, Float64>::dit(in_out);
                }
                static void dif(FloatTy in_out[], size_t fft_len)
                {
                    if (fft_len > LEN)
                    {
                        FHTAlt<LEN * 2, FloatTy>::dif(in_out, fft_len);
                        return;
                    }
                    FHTDefault<LEN, Float64>::init();
                    FHTDefault<LEN, Float64>::dif(in_out);
                }
            };
            template <typename FloatTy>
            struct FHTAlt<FHT_MAX_LEN * 2, FloatTy>
            {
                static void dit(FloatTy in_out[], size_t len)
                {
                    throw("Length of FHT can't be larger than FHT_MAX_LEN");
                }
                static void dif(FloatTy in_out[], size_t len)
                {
                    throw("Length of FHT can't be larger than FHT_MAX_LEN");
                }
            };

            // 时间抽取快速哈特莱变换
            template <typename FloatTy>
            inline void fht_dit(FloatTy in_out[], size_t fht_len)
            {
                FHTAlt<1, FloatTy>::dit(in_out, fht_len);
            }
            // 频率抽取快速哈特莱变换
            template <typename FloatTy>
            inline void fht_dif(FloatTy in_out[], size_t fht_len)
            {
                FHTAlt<1, FloatTy>::dif(in_out, fht_len);
            }

            // 离散哈特莱变换
            template <typename FloatTy>
            inline void dht(FloatTy in_out[], size_t len)
            {
                binary_reverse_swap(in_out, len);
                fht_dit(in_out, len);
            }
            // 离散哈特莱逆变换
            template <typename FloatTy>
            inline void idht(FloatTy in_out[], size_t len)
            {
                dht(in_out, len);
                Float64 inv = Float64(1) / len;
                for (size_t i = 0; i < len; i++)
                {
                    in_out[i] *= inv;
                }
            }

            // 实数快速傅里叶变换,利用FHT的结果进行变换
            template <typename FloatTy>
            inline void dft_real(const FloatTy in[], std::complex<FloatTy> out[], size_t len)
            {
                using Complex = std::complex<FloatTy>;
                len = int_floor2(len);
                std::vector<FloatTy> tmp_ary(in, in + len);
                dht(tmp_ary.data(), len);

                out[0] = tmp_ary[0];
                out[len / 2] = tmp_ary[len / 2];
                for (size_t i = 1; i < len / 2; i++)
                {
                    FloatTy temp0 = tmp_ary[i], temp1 = tmp_ary[len - i];
                    Complex x0 = Complex(temp0 + temp1, temp0 - temp1) * FloatTy(0.5);
                    out[i] = std::conj(x0), out[len - i] = x0;
                }
            }

            // 实数快速傅里叶逆变换
            template <typename FloatTy>
            inline void idft_real(const std::complex<FloatTy> in[], FloatTy out[], size_t len)
            {
                using Complex = std::complex<FloatTy>;
                len = int_floor2(len);
                out[0] = in[0].real();
                out[len / 2] = in[len / 2].real();
                for (size_t i = 1; i < len / 2; i++)
                {
                    Complex x0 = in[i];
                    FloatTy temp0 = x0.real(), temp1 = x0.imag();
                    out[i] = temp0 - temp1, out[len - i] = temp0 + temp1;
                }
                idht(out, len);
            }

            // FHT加速卷积
            template <typename FloatTy>
            inline void fht_convolution(FloatTy fht_ary1[], FloatTy fht_ary2[], FloatTy out[], size_t fht_len)
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
                const FloatTy inv = 0.5 / fht_len;
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
    // 进行2^64进制的乘法
    inline void FHTMul(uint64_t *out, const uint64_t *in1, size_t in_len1, const uint64_t *in2, size_t in_len2)
    {
        // Use 16bit binary as an element
        auto out_16 = reinterpret_cast<uint16_t *>(out);
        auto in1_16 = reinterpret_cast<const uint16_t *>(in1);
        auto in2_16 = reinterpret_cast<const uint16_t *>(in2);
        size_t in1_len16 = in_len1 * sizeof(*in1) / sizeof(*in1_16);
        size_t in2_len16 = in_len2 * sizeof(*in2) / sizeof(*in2_16);
        size_t out_len16 = in1_len16 + in2_len16, conv_len = out_len16 - 1, fht_len = int_ceil2(conv_len);

        std::vector<Float64> buffer1(fht_len), buffer2(fht_len); // FHT bufffer
        std::copy(in1_16, in1_16 + in1_len16, buffer1.data());
        std::copy(in2_16, in2_16 + in2_len16, buffer2.data());

        hint_transform::hint_fht::fht_convolution(buffer1.data(), buffer2.data(), buffer1.data(), fht_len); // 卷积

        uint64_t carry = 0;
        for (size_t i = 0; i < conv_len; i++)
        {
            carry += uint64_t(buffer1[i] + 0.5);
            out_16[i] = carry & UINT16_MAX;
            carry = carry >> 16;
        }
        out_16[conv_len] = carry & UINT16_MAX;
    }
    // 进行2^64进制的平方
    inline void FHTSquare(uint64_t *out, const uint64_t *in, size_t in_len)
    {
        // Use 16bit binary as an element
        auto out_16 = reinterpret_cast<uint16_t *>(out);
        auto in_16 = reinterpret_cast<const uint16_t *>(in);
        size_t in_len16 = in_len * sizeof(*in) / sizeof(*in_16);
        size_t out_len16 = in_len16 * 2, conv_len = out_len16 - 1, fht_len = int_ceil2(conv_len);

        std::vector<Float64> buffer(fht_len); // FHT bufffer
        std::copy(in_16, in_16 + in_len16, buffer.data());

        hint_transform::hint_fht::fht_convolution(buffer.data(), buffer.data(), buffer.data(), fht_len); // 卷积

        uint64_t carry = 0;
        for (size_t i = 0; i < conv_len; i++)
        {
            carry += uint64_t(buffer[i] + 0.5);
            out_16[i] = carry & UINT16_MAX;
            carry = carry >> 16;
        }
        out_16[conv_len] = carry & UINT16_MAX;
    }
}

using namespace std;

// // Example

// // 从二进制字符串到2^64进制大整数组
// vector<uint64_t> from_binary_str(const string &s)
// {
//     auto binstr_to_ui64 = [](const char *p, size_t len)
//     {
//         if (len == 0)
//         {
//             return uint64_t(0);
//         }
//         len = min(size_t(64), len);
//         string tmp(p, p + len);
//         return bitset<64>(tmp.data()).to_ullong();
//     };
//     size_t in_len = s.size();
//     vector<uint64_t> res;
//     size_t i = in_len;
//     while (i >= 64)
//     {
//         i -= 64;
//         res.push_back(binstr_to_ui64(s.data() + i, 64));
//     }
//     if (i > 0)
//     {
//         res.push_back(binstr_to_ui64(s.data(), i));
//     }
//     return res;
// }

// // 从2^64进制大整数组到二进制字符串
// string to_binary_str(const vector<uint64_t> &v)
// {
//     size_t in_len = v.size();
//     string res;
//     size_t i = in_len;
//     while (i > 0)
//     {
//         i--;
//         res += bitset<64>(v[i]).to_string();
//     }
//     i = 0;
//     while (res[i] == '0')
//     {
//         i++;
//     }
//     return res.substr(i, res.size() - i);
// }

// int main()
// {
//     // 正确性测试
//     string s1("10010001101011110010110100101101011010010100101101101001010010100101010100101110010100110110101001010000010101010101010010101001010101001010101001001010000110101110011100101010101110000010101101010010101010111100100010100101010010110100101101001011010110100101001010010111001010011011010100101000001010101001010101110000010101101010010010110100110101001001011010011010100101010101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010100101010010101010010101010010010100100111001010101011100000101011010100101010101111001000101001010111001010011011010100101000001010101011010100101101001010100101101001101010010010110100010010110101001011010010101001011010011010100100101101000100101");
//     string s2("10010001101011110010001100100011010111100101101001011010110100101001011010010001101011110010110100101101011010010100101101101001010010100101010100101110010100100011010111100101101001011010110100101001011011011010101010101001010100101010100101010100100101001101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010101010100010100101001010101001011100101101001010010100101010100101110010101011100111001010101011100000101011010100100101111010100100101101001101010010010110100110101000101001010100101010100101010100100101001101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010101010101010010101001010101001010101001001010010011100101010101110");
//     auto v1 = from_binary_str(s1);
//     auto v2 = from_binary_str(s2);
//     vector<uint64_t> v3(v1.size() + v2.size());
//     vector<uint64_t> v4(v2.size() * 2);

//     // 乘法
//     hint::FHTMul(v3.data(), v1.data(), v1.size(), v2.data(), v2.size()); // 计算s1,s2乘积

//     // 平方
//     hint::FHTSquare(v4.data(), v2.data(), v2.size()); // 计算s2平方

//     auto mul_res = to_binary_str(v3);
//     auto square_res = to_binary_str(v4);

//     cout << "mul_res: " << mul_res << std::endl;       // 101001011100111111000110101010110101110100000001101000110100110110001001010100101001110100011000001001000100011001010111011011011110101111010101000100010001101101100010011100110000111011110010010011011111001100111100010111011101000100010010101100011110110000010100100011001101110110111110100011110010111110010001011011000100011010110100011110111011001011001100111110110101001001110011100101001101101000010011011110000100100101110101001100111001111010100000010110111100010111001000010001101100000101000111010000111101000000001011110100010010100100000010100000010100110111001100111111110101000111011111110100100000011010001101010111101001111111101000100100100100000111010111110110011111101001001110111011100101100011111110011011011100001011010100101110110100000101011111111000110011100001001111001111110010011011001010000111111101000010001111011000010111111010010101110001011110011111010101110000101110010100000001101101101001110000111011100101100101110101100000011100111100111101001110010001100110100111001100001000001010001010011000101100111000101111001111111000111001111010101000100000000010011111101111110010011011011101101010001010000101001011100000010101111011110001101000001000101100010110000101111000111100100101000101101011000000001001110111010101011110011100001011001111110000101011001000010100100110011111000101100101110111100011100111100111000101000000111110001011100100111100001100011001110001100001111011110101001110111011100100110
//     cout << "square_res: " << square_res << std::endl; // 101001011100111110111011001111100100000000100000000101100011010010111101100101111001000111101100101000111001111001110000101010110111101011011001111111110110011110100000001000100101111011111101100001000001000100010110001011101011111100000011010101111001010111111110101111001111111101000001000110001010010111001100011001100101101111100011010001110011100101101000001110011011101101100000111001001100000000100110011111100111001001110000001111010101110000011111010100000101010011111110000101101000001100100110101011010111100101011001111011111001111001011000010111011111010101101000001101101110100100000001100110000000111000000100111110011010011110010000100001111000101101111011101011000111000101000101011101000011011110100101010011000000001100111100101111100111011000110010000101000000001001110010101000111000110100010000010011000100011110001111000100011000111101111001111000110110000011010011010000000110000001000100111110011101101110000010011001011001111111000100111110011101111110110111010110111000111110000110001111001000101111001011110011100000101001010001011011001011010010001011111011000101001010001010001000001100100111101110101001010011111011110000010100101100011011010111101010101101010100101101100000100010111010100111101101000101101110101100000110101101001110010100000000110001101100110101111111010111101001111111011000110010110011001111110011100010100111100010001110001111010000111101110001100010110100001110011101100110000111001000100

//     cout << "------------------------------------------------------------------------------------------\n";
//     cout << "Performance:\n";
//     // 性能测试
//     string s3(8520, '1'); // 计算8192位二进制数的乘法和平方
//     string s4(8500, '1'); // 计算8192位二进制数的乘法和平方

//     auto v5 = from_binary_str(s3);
//     auto v6 = from_binary_str(s4);

//     vector<uint64_t> v7(v5.size() + v6.size());
//     vector<uint64_t> v8(v6.size() * 2);

//     auto t1 = std::chrono::system_clock::now();
//     hint::FHTMul(v7.data(), v5.data(), v5.size(), v6.data(), v6.size());
//     auto t2 = std::chrono::system_clock::now();
//     hint::FHTSquare(v8.data(), v6.data(), v6.size());
//     auto t3 = std::chrono::system_clock::now();

//     cout << to_binary_str(v7) << "\n";
//     cout << to_binary_str(v8) << "\n";

//     cout << "Multiply time:" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us\n";
//     cout << "Square time:" << chrono::duration_cast<chrono::microseconds>(t3 - t2).count() << "us\n";
// }

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
// FHT convolution
// int main()
// {
//     hint::hint_transform::hint_fht::fht_init<double>();
//     int n = 18;
//     cin >> n;
//     size_t len = 1 << n; // 变换长度
//     uint64_t ele = 5;
//     vector<uint32_t> in1(len / 2, ele);
//     vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
//     auto t1 = chrono::steady_clock::now();
//     vector<uint32_t> res = poly_multiply(in1, in2);
//     auto t2 = chrono::steady_clock::now();
//     result_test<uint32_t>(res, ele); // 结果校验
//     cout << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us\n";
// }

// Example DFT for real data
int main()
{
    cin.tie(nullptr)->sync_with_stdio(false);
    cout.tie(nullptr)->sync_with_stdio(false);
    using namespace hint;
    using namespace hint_transform;
    using namespace hint_fht;

    constexpr size_t len = 1 << 18;
    static Float64 r[len]{};   // real data
    static Complex64 c[len]{}; // complex data
    for (size_t i = 0; i < len; i++)
    {
        c[i] = r[i] = i;
    }
    fht_init<Float64>();

    auto t1 = std::chrono::high_resolution_clock::now();
    dft_real(r, c, len); // Real time data -> complex frequency data
    auto t2 = std::chrono::high_resolution_clock::now();
    idft_real(c, r, len); // Complex frequency data -> Real time data
    auto t3 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < len; i++)
    {
        std::cout << r[i] << c[i] << "\n";
    }
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              << "us " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us\n";
}