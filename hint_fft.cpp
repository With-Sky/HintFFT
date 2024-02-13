// TSKY 2024/2/13
#include <vector>
#include <complex>
#include <iostream>
#include <array>
#include <chrono>
#include <cstdint>
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
                    table[i] = unit_root(FloatTy(i) * factor * unit);
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

            namespace split_radix
            {
                template <typename FloatTy, int log_len>
                class FFTTableSplitRadix
                {
                public:
                    using HalfTable = FFTTableSplitRadix<FloatTy, log_len - 1>;
                    using TableTy = ComplexTableDynamic<FloatTy, log_len, 4>;

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
                class FFTTableSplitRadix<FloatTy, 5>
                {
                public:
                    static constexpr int factor1 = -1;
                    static constexpr int factor3 = -3;
                    using TableTy = ComplexTableDynamic<FloatTy, 5, 4>;
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
                typename FFTTableSplitRadix<FloatTy, 5>::TableTy FFTTableSplitRadix<FloatTy, 5>::table1;
                template <typename FloatTy>
                typename FFTTableSplitRadix<FloatTy, 5>::TableTy FFTTableSplitRadix<FloatTy, 5>::table3;

                template <size_t LEN, typename FloatTy>
                struct FFT
                {
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

                        auto omega1_it = TableTy::get_it1(), omega3_it = TableTy::get_it3();
                        auto it = in_out;
                        for (; it < in_out + quarter_len; it++, omega1_it++, omega3_it++)
                        {
                            DataTy temp2 = it[quarter_len * 2] * omega1_it[0];
                            DataTy temp3 = it[quarter_len * 3] * omega3_it[0];

                            transform2(temp2, temp3);
                            temp3 = mul_neg_j(temp3);

                            DataTy temp0 = it[0];
                            DataTy temp1 = it[quarter_len];
                            it[0] = temp0 + temp2;
                            it[quarter_len] = temp1 + temp3;
                            it[quarter_len * 2] = temp0 - temp2;
                            it[quarter_len * 3] = temp1 - temp3;
                        }
                    }

                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        using value_type = typename std::iterator_traits<ComplexIt>::value_type;
                        static_assert(std::is_same<value_type, DataTy>::value, "Must be same as the FFT template float type");

                        auto omega1_it = TableTy::get_it1(), omega3_it = TableTy::get_it3();
                        auto it = in_out;
                        for (; it < in_out + quarter_len; it++, omega1_it++, omega3_it++)
                        {
                            DataTy temp0 = it[0];
                            DataTy temp1 = it[quarter_len];
                            DataTy temp2 = it[quarter_len * 2];
                            DataTy temp3 = it[quarter_len * 3];
                            it[0] = temp0 + temp2;
                            it[quarter_len] = temp1 + temp3;

                            temp2 = temp0 - temp2;
                            temp3 = temp1 - temp3;
                            temp3 = mul_neg_j(temp3);
                            transform2(temp2, temp3);

                            it[quarter_len * 2] = temp2 * omega1_it[0];
                            it[quarter_len * 3] = temp3 * omega3_it[0];
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
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp1);
                        transform2(temp2, temp3);
                        temp3 = mul_neg_j(temp3);

                        in_out[0] = temp0 + temp2;
                        in_out[1] = temp1 + temp3;
                        in_out[2] = temp0 - temp2;
                        in_out[3] = temp1 - temp3;
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        auto temp0 = in_out[0];
                        auto temp1 = in_out[1];
                        auto temp2 = in_out[2];
                        auto temp3 = in_out[3];

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        temp3 = mul_neg_j(temp3);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
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
                        temp3 = mul_neg_j(temp3);
                        temp7 = mul_neg_j(temp7);

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp7, temp5);
                        temp5 = COS_8 * Complex(temp5.real() - temp5.imag(), temp5.real() + temp5.imag());
                        temp6 = mul_neg_j(temp6);
                        temp7 = COS_8 * Complex(temp7.imag() + temp7.real(), temp7.imag() - temp7.real());

                        in_out[0] = temp0 + temp4;
                        in_out[1] = temp1 + temp7;
                        in_out[2] = temp2 + temp6;
                        in_out[3] = temp3 + temp5;
                        in_out[4] = temp0 - temp4;
                        in_out[5] = temp1 - temp7;
                        in_out[6] = temp2 - temp6;
                        in_out[7] = temp3 - temp5;
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
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
                        temp5 = COS_8 * Complex(temp5.imag() + temp5.real(), temp5.imag() - temp5.real());
                        temp6 = mul_neg_j(temp6);
                        temp7 = COS_8 * Complex(temp7.real() - temp7.imag(), temp7.real() + temp7.imag());

                        transform2(temp0, temp2);
                        transform2(temp1, temp3);
                        transform2(temp4, temp6);
                        transform2(temp5, temp7);
                        temp3 = mul_neg_j(temp3);
                        temp5 = mul_neg_j(temp5);

                        in_out[0] = temp0 + temp1;
                        in_out[1] = temp0 - temp1;
                        in_out[2] = temp2 + temp3;
                        in_out[3] = temp2 - temp3;
                        in_out[4] = temp4 + temp7;
                        in_out[5] = temp4 - temp7;
                        in_out[6] = temp6 + temp5;
                        in_out[7] = temp6 - temp5;
                    }
                };

                template <typename FloatTy>
                struct FFT<16, FloatTy>
                {
                    using Complex = std::complex<FloatTy>;
                    static constexpr Complex w1{COS_16, -SIN_16};
                    static constexpr Complex w3{SIN_16, -COS_16};
                    static constexpr Complex w9{-COS_16, SIN_16};
                    static void init() {}
                    template <typename ComplexIt>
                    static void dit(ComplexIt in_out)
                    {
                        FFT<8, FloatTy>::dit(in_out);
                        FFT<4, FloatTy>::dit(in_out + 8);
                        FFT<4, FloatTy>::dit(in_out + 12);

                        Complex temp2 = in_out[8];
                        Complex temp3 = in_out[12];
                        transform2(temp2, temp3);
                        temp3 = mul_neg_j(temp3);
                        Complex temp0 = in_out[0];
                        Complex temp1 = in_out[4];
                        in_out[0] = temp0 + temp2;
                        in_out[4] = temp1 + temp3;
                        in_out[8] = temp0 - temp2;
                        in_out[12] = temp1 - temp3;

                        temp2 = in_out[9] * w1;
                        temp3 = in_out[13] * w3;
                        transform2(temp2, temp3);
                        temp3 = mul_neg_j(temp3);
                        temp0 = in_out[1];
                        temp1 = in_out[5];
                        in_out[1] = temp0 + temp2;
                        in_out[5] = temp1 + temp3;
                        in_out[9] = temp0 - temp2;
                        in_out[13] = temp1 - temp3;

                        temp2 = in_out[10];
                        temp3 = in_out[14];
                        temp2 = COS_8 * Complex(temp2.imag() + temp2.real(), temp2.imag() - temp2.real());
                        temp3 = COS_8 * Complex(temp3.real() - temp3.imag(), temp3.real() + temp3.imag());
                        transform2(temp2, temp3);
                        temp2 = mul_neg_j(temp2);
                        temp0 = in_out[2];
                        temp1 = in_out[6];
                        in_out[2] = temp0 + temp3;
                        in_out[6] = temp1 + temp2;
                        in_out[10] = temp0 - temp3;
                        in_out[14] = temp1 - temp2;

                        temp2 = in_out[11] * w3;
                        temp3 = in_out[15] * w9;
                        transform2(temp2, temp3);
                        temp3 = mul_neg_j(temp3);
                        temp0 = in_out[3];
                        temp1 = in_out[7];
                        in_out[3] = temp0 + temp2;
                        in_out[7] = temp1 + temp3;
                        in_out[11] = temp0 - temp2;
                        in_out[15] = temp1 - temp3;
                    }
                    template <typename ComplexIt>
                    static void dif(ComplexIt in_out)
                    {
                        Complex temp0 = in_out[0];
                        Complex temp1 = in_out[4];
                        Complex temp2 = in_out[8];
                        Complex temp3 = in_out[12];
                        in_out[0] = temp0 + temp2;
                        in_out[4] = temp1 + temp3;
                        temp2 = temp0 - temp2;
                        temp3 = temp1 - temp3;
                        temp3 = mul_neg_j(temp3);
                        transform2(temp2, temp3);
                        in_out[8] = temp2;
                        in_out[12] = temp3;

                        temp0 = in_out[1];
                        temp1 = in_out[5];
                        temp2 = in_out[9];
                        temp3 = in_out[13];
                        in_out[1] = temp0 + temp2;
                        in_out[5] = temp1 + temp3;
                        temp2 = temp0 - temp2;
                        temp3 = temp1 - temp3;
                        temp3 = mul_neg_j(temp3);
                        transform2(temp2, temp3);
                        in_out[9] = temp2 * w1;
                        in_out[13] = temp3 * w3;

                        temp0 = in_out[2];
                        temp1 = in_out[6];
                        temp2 = in_out[10];
                        temp3 = in_out[14];
                        in_out[2] = temp0 + temp2;
                        in_out[6] = temp1 + temp3;
                        temp2 = temp0 - temp2;
                        temp3 = temp1 - temp3;
                        temp3 = mul_neg_j(temp3);
                        transform2(temp3, temp2);
                        temp2 = COS_8 * Complex(temp2.real() - temp2.imag(), temp2.real() + temp2.imag());
                        temp3 = COS_8 * Complex(temp3.imag() + temp3.real(), temp3.imag() - temp3.real());
                        in_out[10] = temp3;
                        in_out[14] = temp2;

                        temp0 = in_out[3];
                        temp1 = in_out[7];
                        temp2 = in_out[11];
                        temp3 = in_out[15];
                        in_out[3] = temp0 + temp2;
                        in_out[7] = temp1 + temp3;
                        temp2 = temp0 - temp2;
                        temp3 = temp1 - temp3;
                        temp3 = mul_neg_j(temp3);
                        transform2(temp2, temp3);
                        in_out[11] = temp2 * w3;
                        in_out[15] = temp3 * w9;

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
            }

            // 默认FFT为分裂基
            template <size_t len, typename FloatTy>
            using FFTDefault = split_radix::FFT<len, FloatTy>;

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

            // Bluestein algorithm,任意长度FFT
            namespace bluestein
            {
                template <typename FloatTy>
                class FFT
                {
                public:
                    using Complex = std::complex<FloatTy>;
                    FFT(size_t len) : fft_len(len)
                    {
                        if (is_2pow(fft_len))
                        {
                            return;
                        }
                        size_t conv_len = int_ceil2(fft_len * 3 - 2);
                        omega_array_conj.resize(fft_len);
                        omega_array_fft.resize(conv_len);
                        omega_array_conj[0] = Complex(1);
                        const FloatTy inv = FloatTy(1) / conv_len;
                        omega_array_fft[fft_len - 1] = inv;
                        for (size_t i = 1; i < fft_len; i++)
                        {
                            Complex temp = unit_root<FloatTy>(i * i * HINT_PI / FloatTy(fft_len));
                            omega_array_conj[i] = temp;
                            omega_array_fft[fft_len - 1 - i] = omega_array_fft[fft_len - 1 + i] = temp * inv;
                        }
                        fft_dif(omega_array_fft.data(), conv_len);
                    }
                    void forward(Complex in_out[]) const
                    {
                        if (is_2pow(fft_len))
                        {
                            dft(in_out, fft_len);
                            return;
                        }
                        size_t conv_len = int_ceil2(fft_len * 3 - 2);
                        std::vector<Complex> input_array(conv_len);
                        for (size_t i = 0; i < fft_len; i++)
                        {
                            input_array[i] = in_out[i] * std::conj(omega_array_conj[i]);
                        }
                        // 卷积
                        {
                            fft_dif(input_array.data(), conv_len);
                            for (size_t i = 0; i < conv_len; i++)
                            {
                                input_array[i] = std::conj(input_array[i] * omega_array_fft[i]);
                            }
                            fft_dit(input_array.data(), conv_len);
                        }
                        for (size_t i = 0; i < fft_len; i++)
                        {
                            in_out[i] = std::conj(omega_array_conj[i] * input_array[fft_len - 1 + i]);
                        }
                    }
                    void backward(Complex in_out[]) const
                    {
                        const FloatTy inv = FloatTy(1) / fft_len;
                        for (size_t i = 0; i < fft_len; i++)
                        {
                            in_out[i] = std::conj(in_out[i]);
                        }
                        forward(in_out);
                        for (size_t i = 0; i < fft_len; i++)
                        {
                            in_out[i] = std::conj(in_out[i]) * inv;
                        }
                    }

                private:
                    size_t fft_len;
                    std::vector<Complex> omega_array_conj;
                    std::vector<Complex> omega_array_fft;
                };
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
    for (size_t i = 0; i < fft_len; i++)
    {
        fft_ary1[i] = std::conj(fft_ary1[i] * fft_ary1[i]);
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
// int main()
// {
//     fft_init<Float64>();
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

// Example DFT for any length
// int main()
// {
//     constexpr size_t len = 6;
//     Complex64 a[len]{1, 2, 3, 4, 5, 6};
//     hint_fft::bluestein::FFT<Float64> bs(len);
//     bs.forward(a);
//     for (auto i : a)
//     {
//         cout << i << " ";
//     }
//     cout << "\n";
//     bs.backward(a);
//     for (auto i : a)
//     {
//         cout << i << " ";
//     }
//     cout << "\n";
// }

// Example DFT for length power of 2
// int main()
// {
//     constexpr size_t len = 16;
//     Complex64 a[len]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     dft(a, len);
//     for (auto i : a)
//     {
//         cout << i << " ";
//     }
//     cout << "\n";
//     idft(a, len);
//     for (auto i : a)
//     {
//         cout << i << " ";
//     }
//     cout << "\n";
// }

// Example for real convolution
// int main()
// {
//     Float64 a[4]{1, 2, 3, 4};
//     Float64 b[4]{5, 6, 7, 8};
//     Float64 c[7]{};
//     fft_convolution_real(a, 4, b, 4, c);
//     for (auto i : c)
//     {
//         cout << i << " ";
//     }
// }

// Example DFT for real data
int main()
{
    constexpr size_t len = 1 << 10;
    static Float64 b[len]{};   // real data
    static Float64 d[len]{};   // real data
    static Complex64 c[len]{}; // complex data
    for (size_t i = 0; i < len; i++)
    {
        c[i] = b[i] = i;
    }
    hint_fft::bluestein::FFT<Float64> bs(len);
    fft_init<Float64>();
    auto t1 = std::chrono::high_resolution_clock::now();
    dft_real(b, c, len); // Real time data -> complex frequency data
    auto t2 = std::chrono::high_resolution_clock::now();
    idft_real(c, d, len); // Complex frequency data -> Real time data
    auto t3 = std::chrono::high_resolution_clock::now();
    // check result
    for (size_t i = 0; i < len; i++)
    {
        std::cout << d[i] << c[i] << "\n";
    }
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
              << "us " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us\n";
}