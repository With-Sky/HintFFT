#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>
#include "stopwatch.hpp"

#define TABLE_TYPE 2
// #define FFT_R2_TEMPLATE

namespace hint
{
    using Complex = std::complex<double>; // 使用标准库的complex作为复数类
    using INT_32 = int32_t;
    using UINT_32 = uint32_t;

    constexpr double HINT_PI = 3.1415926535897932384626433832795;
    constexpr double HINT_2PI = HINT_PI * 2;

    template <typename T>
    constexpr T max_2pow(T n)
    {
        T res = 1;
        res <<= (sizeof(T) * 8 - 1);
        while (res > n)
        {
            res /= 2;
        }
        return res;
    }
    template <typename T>
    constexpr T min_2pow(T n)
    {
        T res = 1;
        while (res < n)
        {
            res *= 2;
        }
        return res;
    }
    template <typename T>
    inline T *ary_realloc(T *ptr, size_t len)
    {
        if (len * sizeof(T) < INT64_MAX)
        {
            ptr = static_cast<T *>(realloc(ptr, len * sizeof(T)));
        }
        if (ptr == nullptr)
        {
            throw("realloc error");
        }
        return ptr;
    }
    template <typename T>
    constexpr size_t hint_log2(T n)
    {
        T res = 0;
        while (n > 1)
        {
            n /= 2;
            res++;
        }
        return res;
    }
    template <typename T>
    void ary_copy(T *target, const T *source, size_t len)
    {
        if (len == 0 || target == source)
        {
            return;
        }
        if (len >= INT64_MAX)
        {
            throw("Ary too long\n");
        }
        std::memcpy(target, source, len * sizeof(T));
    }
    // 从其他类型数组拷贝到复数组
    template <typename T>
    inline void com_ary_combine_copy(Complex *target, const T &source1, size_t len1, const T &source2, size_t len2)
    {
        size_t min_len = std::min(len1, len2);
        size_t i = 0;
        while (i < min_len)
        {
            target[i] = Complex(source1[i], source2[i]);
            i++;
        }
        while (i < len1)
        {
            target[i].real(source1[i]);
            i++;
        }
        while (i < len2)
        {
            target[i].imag(source2[i]);
            i++;
        }
    }
    namespace hint_transform
    {

        class ComplexTable
        {
        private:
            std::vector<Complex> table;
            INT_32 max_log_size = 2;
            INT_32 cur_log_size = 2;

            static constexpr double PI = HINT_PI;

            ComplexTable(const ComplexTable &) = delete;
            ComplexTable &operator=(const ComplexTable &) = delete;

        public:
            ~ComplexTable() {}
            // 初始化可以生成平分圆1<<shift份产生的单位根的表
            ComplexTable(UINT_32 max_shift)
            {
                max_shift = std::max<size_t>(max_shift, 2);
                max_log_size = max_shift;
                size_t ary_size = 1ull << (max_shift - 1);
                table.resize(ary_size);
                table[0] = Complex(1);
                expend(max_shift);
            }
            void expend(INT_32 shift)
            {
                shift = std::max<INT_32>(shift, 3);
                if (shift > max_log_size)
                {
                    throw("FFT length too long for lut\n");
                }
                for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                {
                    size_t len = 1ull << i, vec_size = len / 4;
                    for (size_t pos = 0; pos < vec_size; pos++)
                    {
                        table[vec_size + pos] = unit_root(len, pos);
                    }
                }
                cur_log_size = std::max(cur_log_size, shift);
            }
            // 返回单位圆上辐角为theta的点
            static Complex unit_root(double theta)
            {
                return std::polar<double>(1.0, theta);
            }
            // 返回单位圆上平分m份的第n个
            static Complex unit_root(size_t m, size_t n)
            {
                return unit_root((2.0 * PI * n) / m);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根
            Complex get_complex(UINT_32 shift, size_t n) const
            {
                size_t rank = 1ull << shift;
                const size_t rank_ff = rank - 1, quad_n = n << 2;
                // n &= rank_ff;
                size_t zone = quad_n >> shift; // 第几象限
                if ((quad_n & rank_ff) == 0)
                {
                    static constexpr Complex ONES[4] = {Complex(1, 0), Complex(0, 1), Complex(-1, 0), Complex(0, -1)};
                    return ONES[zone];
                }
                Complex tmp;
                if ((zone & 2) == 0)
                {
                    if ((zone & 1) == 0)
                    {
                        tmp = table[rank / 4 + n];
                    }
                    else
                    {
                        tmp = table[rank - rank / 4 - n];
                        tmp.real(-tmp.real());
                    }
                }
                else
                {
                    if ((zone & 1) == 0)
                    {
                        tmp = -table[n - rank / 4];
                    }
                    else
                    {
                        tmp = table[rank + rank / 4 - n];
                        tmp.imag(-tmp.imag());
                    }
                }
                return tmp;
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根的共轭
            Complex get_complex_conj(UINT_32 shift, size_t n) const
            {
                size_t rank = 1ull << shift;
                const size_t rank_ff = rank - 1, quad_n = n << 2;
                // n &= rank_ff;
                size_t zone = quad_n >> shift; // 第几象限
                if ((quad_n & rank_ff) == 0)
                {
                    static constexpr Complex ONES_CONJ[4] = {Complex(1, 0), Complex(0, -1), Complex(-1, 0), Complex(0, 1)};
                    return ONES_CONJ[zone];
                }
                Complex tmp;
                if ((zone & 2) == 0)
                {
                    if ((zone & 1) == 0)
                    {
                        tmp = table[rank / 4 + n];
                        tmp.imag(-tmp.imag());
                    }
                    else
                    {
                        tmp = -table[rank - rank / 4 - n];
                    }
                }
                else
                {
                    if ((zone & 1) == 0)
                    {
                        tmp = table[n - (rank / 4)];
                        tmp.real(-tmp.real());
                    }
                    else
                    {
                        tmp = table[rank + rank / 4 - n];
                    }
                }
                return tmp;
            }
        };

        class ComplexTableX
        {
        private:
            std::vector<std::vector<Complex>> table;
            INT_32 max_log_size = 1;
            INT_32 cur_log_size = 1;

            static constexpr size_t FAC = 3;
            static constexpr double PI = HINT_PI;

            ComplexTableX(const ComplexTableX &) = delete;
            ComplexTableX &operator=(const ComplexTableX &) = delete;

        public:
            ~ComplexTableX() {}
            // 初始化可以生成平分圆1<<shift份产生的单位根的表
            ComplexTableX(UINT_32 max_shift)
            {
                max_shift = std::max<size_t>(max_shift, 1);
                max_log_size = max_shift;
                table.resize(max_shift + 1);
                table[0] = table[1] = std::vector<Complex>({1});
                expend(max_shift);
            }
            void expend(INT_32 shift)
            {
                shift = std::max<INT_32>(shift, 2);
                if (shift > max_log_size)
                {
                    throw("FFT length too long for lut\n");
                }
                for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                {
                    size_t len = 1ull << i, vec_size = len * FAC / 4;
                    table[i] = std::vector<Complex>(vec_size);
                    for (size_t pos = 0; pos < len / 4; pos++)
                    {
                        Complex tmp = std::conj(unit_root(len, pos));
                        table[i][pos] = tmp;
                        table[i][pos + len / 4] = Complex(tmp.imag(), -tmp.real());
                        table[i][pos + len / 2] = -tmp;
                    }
                }
                cur_log_size = std::max(cur_log_size, shift);
            }
            // 返回单位圆上辐角为theta的点
            static Complex unit_root(double theta)
            {
                return std::polar<double>(1.0, theta);
            }
            // 返回单位圆上平分m份的第n个
            static Complex unit_root(size_t m, size_t n)
            {
                return unit_root((2.0 * PI * n) / m);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根
            Complex get_complex(UINT_32 shift, size_t n) const
            {
                return std::conj(table[shift][n]);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根的共轭
            Complex get_complex_conj(UINT_32 shift, size_t n) const
            {
                return table[shift][n];
            }
        };
        class ComplexTableZ
        {
        private:
            std::vector<Complex> table;
            INT_32 max_log_size = 1;
            INT_32 cur_log_size = 1;

            static constexpr double PI = HINT_PI;
            static constexpr size_t FAC = 3;
            ComplexTableZ(const ComplexTableZ &) = delete;
            ComplexTableZ &operator=(const ComplexTableZ &) = delete;

        public:
            ~ComplexTableZ() {}
            // 初始化可以生成平分圆1<<shift份产生的单位根的表
            ComplexTableZ(UINT_32 max_shift)
            {
                max_shift = std::max<size_t>(max_shift, 1);
                max_log_size = max_shift;
                size_t vec_size = (1 << (max_shift - 1)) * FAC;
                table.resize(vec_size);
                table[0] = table[1] = Complex(1, 0), table[2] = Complex(0, 1);
                expend(max_shift);
            }
            void expend(INT_32 shift)
            {
                shift = std::max<INT_32>(shift, 2);
                if (shift > max_log_size)
                {
                    throw("FFT length too long for lut\n");
                }
                for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                {
                    size_t len = 1ull << i, vec_size = len * FAC / 4;
                    for (size_t pos = 0; pos < len / 4; pos++)
                    {
                        Complex tmp = std::conj(unit_root(len, pos));
                        table[vec_size + pos] = tmp;
                        table[vec_size + pos + len / 4] = Complex(tmp.imag(), -tmp.real());
                        table[vec_size + pos + len / 2] = -tmp;
                    }
                }
                cur_log_size = std::max(cur_log_size, shift);
            }
            // 返回单位圆上辐角为theta的点
            static Complex unit_root(double theta)
            {
                return std::polar<double>(1.0, theta);
            }
            // 返回单位圆上平分m份的第n个
            static Complex unit_root(size_t m, size_t n)
            {
                return unit_root((2.0 * PI * n) / m);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根
            Complex get_complex(UINT_32 shift, size_t n) const
            {
                return std::conj(table[(1 << (shift - 2)) * FAC + n]);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根的共轭
            Complex get_complex_conj(UINT_32 shift, size_t n) const
            {
                return table[(1 << (shift - 2)) * FAC + n];
            }
        };

        constexpr size_t lut_max_rank = 23;
#if TABLE_TYPE == 0
        static ComplexTable TABLE(lut_max_rank);
#elif TABLE_TYPE == 1
        static ComplexTableX TABLE(lut_max_rank);
#elif TABLE_TYPE == 2
        static ComplexTableZ TABLE(lut_max_rank);
#endif

        template <typename T>
        void binary_inverse_swap(T &ary, size_t len)
        {
            size_t i = 0;
            for (size_t j = 1; j < len - 1; j++)
            {
                size_t k = len >> 1;
                i ^= k;
                while (k > i)
                {
                    k >>= 1;
                    i ^= k;
                };
                if (j < i)
                {
                    std::swap(ary[i], ary[j]);
                }
            }
        }
        // 2点fft
        inline void fft_2point(Complex &sum, Complex &diff)
        {
            Complex tmp0 = sum;
            Complex tmp1 = diff;
            sum = tmp0 + tmp1;
            diff = tmp0 - tmp1;
        }
        // 4点fft
        inline void fft_4point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[rank] = tmp2 + tmp3;
            input[rank * 2] = tmp0 - tmp1;
            input[rank * 3] = tmp2 - tmp3;
        }
        inline void fft_dit_4point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp1);
            fft_2point(tmp2, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 2] = tmp0 - tmp2;
            input[rank * 3] = tmp1 - tmp3;
        }
        inline void fft_dit_8point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];
            Complex tmp4 = input[rank * 4];
            Complex tmp5 = input[rank * 5];
            Complex tmp6 = input[rank * 6];
            Complex tmp7 = input[rank * 7];
            fft_2point(tmp0, tmp1);
            fft_2point(tmp2, tmp3);
            fft_2point(tmp4, tmp5);
            fft_2point(tmp6, tmp7);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());
            tmp7 = Complex(tmp7.imag(), -tmp7.real());

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            fft_2point(tmp4, tmp6);
            fft_2point(tmp5, tmp7);
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            tmp5 = cos_1_8 * Complex(tmp5.imag() + tmp5.real(), tmp5.imag() - tmp5.real());
            tmp6 = Complex(tmp6.imag(), -tmp6.real());
            tmp7 = -cos_1_8 * Complex(tmp7.real() - tmp7.imag(), tmp7.real() + tmp7.imag());

            input[0] = tmp0 + tmp4;
            input[rank] = tmp1 + tmp5;
            input[rank * 2] = tmp2 + tmp6;
            input[rank * 3] = tmp3 + tmp7;
            input[rank * 4] = tmp0 - tmp4;
            input[rank * 5] = tmp1 - tmp5;
            input[rank * 6] = tmp2 - tmp6;
            input[rank * 7] = tmp3 - tmp7;
        }
        inline void fft_dit_16point(Complex *input, size_t rank = 1)
        {
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            static constexpr double cos_1_16 = 0.92387953251128675612818318939679;
            static constexpr double sin_1_16 = 0.3826834323650897717284599840304;
            static constexpr Complex w1(cos_1_16, -sin_1_16), w3(sin_1_16, -cos_1_16);
            static constexpr Complex w5(-sin_1_16, -cos_1_16), w7(-cos_1_16, -sin_1_16);

            fft_dit_8point(input, rank);
            fft_dit_8point(input + rank * 8, rank);

            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 8];
            Complex tmp3 = input[rank * 9] * w1;
            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 8] = tmp0 - tmp2;
            input[rank * 9] = tmp1 - tmp3;

            tmp0 = input[rank * 2];
            tmp1 = input[rank * 3];
            tmp2 = input[rank * 10];
            tmp3 = input[rank * 11] * w3;
            tmp2 = cos_1_8 * Complex(tmp2.imag() + tmp2.real(), tmp2.imag() - tmp2.real());
            input[rank * 2] = tmp0 + tmp2;
            input[rank * 3] = tmp1 + tmp3;
            input[rank * 10] = tmp0 - tmp2;
            input[rank * 11] = tmp1 - tmp3;

            tmp0 = input[rank * 4];
            tmp1 = input[rank * 5];
            tmp2 = input[rank * 12];
            tmp3 = input[rank * 13] * w5;
            tmp2 = Complex(tmp2.imag(), -tmp2.real());
            input[rank * 4] = tmp0 + tmp2;
            input[rank * 5] = tmp1 + tmp3;
            input[rank * 12] = tmp0 - tmp2;
            input[rank * 13] = tmp1 - tmp3;

            tmp0 = input[rank * 6];
            tmp1 = input[rank * 7];
            tmp2 = input[rank * 14];
            tmp3 = input[rank * 15] * w7;
            tmp2 = -cos_1_8 * Complex(tmp2.real() - tmp2.imag(), tmp2.real() + tmp2.imag());
            input[rank * 6] = tmp0 + tmp2;
            input[rank * 7] = tmp1 + tmp3;
            input[rank * 14] = tmp0 - tmp2;
            input[rank * 15] = tmp1 - tmp3;
        }
        inline void fft_dit_32point(Complex *input, size_t rank = 1)
        {
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            static constexpr double cos_1_16 = 0.92387953251128675612818318939679;
            static constexpr double sin_1_16 = 0.3826834323650897717284599840304;
            static constexpr double cos_1_32 = 0.98078528040323044912618223613424;
            static constexpr double sin_1_32 = 0.19509032201612826784828486847702;
            static constexpr double cos_3_32 = 0.83146961230254523707878837761791;
            static constexpr double sin_3_32 = 0.55557023301960222474283081394853;
            static constexpr Complex w1(cos_1_32, -sin_1_32), w2(cos_1_16, -sin_1_16), w3(cos_3_32, -sin_3_32);
            static constexpr Complex w5(sin_3_32, -cos_3_32), w6(sin_1_16, -cos_1_16), w7(sin_1_32, -cos_1_32);
            static constexpr Complex w9(-sin_1_32, -cos_1_32), w10(-sin_1_16, -cos_1_16), w11(-sin_3_32, -cos_3_32);
            static constexpr Complex w13(-cos_3_32, -sin_3_32), w14(-cos_1_16, -sin_1_16), w15(-cos_1_32, -sin_1_32);

            fft_dit_16point(input, rank);
            fft_dit_16point(input + rank * 16, rank);

            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 16];
            Complex tmp3 = input[rank * 17] * w1;
            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 16] = tmp0 - tmp2;
            input[rank * 17] = tmp1 - tmp3;

            tmp0 = input[rank * 2];
            tmp1 = input[rank * 3];
            tmp2 = input[rank * 18] * w2;
            tmp3 = input[rank * 19] * w3;
            input[rank * 2] = tmp0 + tmp2;
            input[rank * 3] = tmp1 + tmp3;
            input[rank * 18] = tmp0 - tmp2;
            input[rank * 19] = tmp1 - tmp3;

            tmp0 = input[rank * 4];
            tmp1 = input[rank * 5];
            tmp2 = input[rank * 20];
            tmp3 = input[rank * 21] * w5;
            tmp2 = cos_1_8 * Complex(tmp2.imag() + tmp2.real(), tmp2.imag() - tmp2.real());
            input[rank * 4] = tmp0 + tmp2;
            input[rank * 5] = tmp1 + tmp3;
            input[rank * 20] = tmp0 - tmp2;
            input[rank * 21] = tmp1 - tmp3;

            tmp0 = input[rank * 6];
            tmp1 = input[rank * 7];
            tmp2 = input[rank * 22] * w6;
            tmp3 = input[rank * 23] * w7;
            input[rank * 6] = tmp0 + tmp2;
            input[rank * 7] = tmp1 + tmp3;
            input[rank * 22] = tmp0 - tmp2;
            input[rank * 23] = tmp1 - tmp3;

            tmp0 = input[rank * 8];
            tmp1 = input[rank * 9];
            tmp2 = input[rank * 24];
            tmp3 = input[rank * 25] * w9;
            tmp2 = Complex(tmp2.imag(), -tmp2.real());
            input[rank * 8] = tmp0 + tmp2;
            input[rank * 9] = tmp1 + tmp3;
            input[rank * 24] = tmp0 - tmp2;
            input[rank * 25] = tmp1 - tmp3;

            tmp0 = input[rank * 10];
            tmp1 = input[rank * 11];
            tmp2 = input[rank * 26] * w10;
            tmp3 = input[rank * 27] * w11;
            input[rank * 10] = tmp0 + tmp2;
            input[rank * 11] = tmp1 + tmp3;
            input[rank * 26] = tmp0 - tmp2;
            input[rank * 27] = tmp1 - tmp3;

            tmp0 = input[rank * 12];
            tmp1 = input[rank * 13];
            tmp2 = input[rank * 28];
            tmp3 = input[rank * 29] * w13;
            tmp2 = -cos_1_8 * Complex(tmp2.real() - tmp2.imag(), tmp2.real() + tmp2.imag());
            input[rank * 12] = tmp0 + tmp2;
            input[rank * 13] = tmp1 + tmp3;
            input[rank * 28] = tmp0 - tmp2;
            input[rank * 29] = tmp1 - tmp3;

            tmp0 = input[rank * 14];
            tmp1 = input[rank * 15];
            tmp2 = input[rank * 30] * w14;
            tmp3 = input[rank * 31] * w15;

            input[rank * 14] = tmp0 + tmp2;
            input[rank * 15] = tmp1 + tmp3;
            input[rank * 30] = tmp0 - tmp2;
            input[rank * 31] = tmp1 - tmp3;
        }
        inline void fft_dif_4point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[rank] = tmp0 - tmp1;
            input[rank * 2] = tmp2 + tmp3;
            input[rank * 3] = tmp2 - tmp3;
        }
        inline void fft_dif_8point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];
            Complex tmp4 = input[rank * 4];
            Complex tmp5 = input[rank * 5];
            Complex tmp6 = input[rank * 6];
            Complex tmp7 = input[rank * 7];

            fft_2point(tmp0, tmp4);
            fft_2point(tmp1, tmp5);
            fft_2point(tmp2, tmp6);
            fft_2point(tmp3, tmp7);
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            tmp5 = cos_1_8 * Complex(tmp5.imag() + tmp5.real(), tmp5.imag() - tmp5.real());
            tmp6 = Complex(tmp6.imag(), -tmp6.real());
            tmp7 = -cos_1_8 * Complex(tmp7.real() - tmp7.imag(), tmp7.real() + tmp7.imag());

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            fft_2point(tmp4, tmp6);
            fft_2point(tmp5, tmp7);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());
            tmp7 = Complex(tmp7.imag(), -tmp7.real());

            input[0] = tmp0 + tmp1;
            input[rank] = tmp0 - tmp1;
            input[rank * 2] = tmp2 + tmp3;
            input[rank * 3] = tmp2 - tmp3;
            input[rank * 4] = tmp4 + tmp5;
            input[rank * 5] = tmp4 - tmp5;
            input[rank * 6] = tmp6 + tmp7;
            input[rank * 7] = tmp6 - tmp7;
        }
        inline void fft_dif_16point(Complex *input, size_t rank = 1)
        {
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            static constexpr double cos_1_16 = 0.92387953251128675612818318939679;
            static constexpr double sin_1_16 = 0.3826834323650897717284599840304;
            static constexpr Complex w1(cos_1_16, -sin_1_16), w3(sin_1_16, -cos_1_16);
            static constexpr Complex w5(-sin_1_16, -cos_1_16), w7(-cos_1_16, -sin_1_16);

            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 8];
            Complex tmp3 = input[rank * 9];
            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 8] = tmp0 - tmp2;
            input[rank * 9] = (tmp1 - tmp3) * w1;

            tmp0 = input[rank * 2];
            tmp1 = input[rank * 3];
            tmp2 = input[rank * 10];
            tmp3 = input[rank * 11];
            fft_2point(tmp0, tmp2);
            tmp2 = cos_1_8 * Complex(tmp2.imag() + tmp2.real(), tmp2.imag() - tmp2.real());
            input[rank * 2] = tmp0;
            input[rank * 3] = tmp1 + tmp3;
            input[rank * 10] = tmp2;
            input[rank * 11] = (tmp1 - tmp3) * w3;

            tmp0 = input[rank * 4];
            tmp1 = input[rank * 5];
            tmp2 = input[rank * 12];
            tmp3 = input[rank * 13];
            fft_2point(tmp0, tmp2);
            tmp2 = Complex(tmp2.imag(), -tmp2.real());
            input[rank * 4] = tmp0;
            input[rank * 5] = tmp1 + tmp3;
            input[rank * 12] = tmp2;
            input[rank * 13] = (tmp1 - tmp3) * w5;

            tmp0 = input[rank * 6];
            tmp1 = input[rank * 7];
            tmp2 = input[rank * 14];
            tmp3 = input[rank * 15];
            fft_2point(tmp0, tmp2);
            tmp2 = -cos_1_8 * Complex(tmp2.real() - tmp2.imag(), tmp2.real() + tmp2.imag());
            input[rank * 6] = tmp0;
            input[rank * 7] = tmp1 + tmp3;
            input[rank * 14] = tmp2;
            input[rank * 15] = (tmp1 - tmp3) * w7;

            fft_dif_8point(input, rank);
            fft_dif_8point(input + rank * 8, rank);
        }
        inline void fft_dif_32point(Complex *input, size_t rank = 1)
        {
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            static constexpr double cos_1_16 = 0.92387953251128675612818318939679;
            static constexpr double sin_1_16 = 0.3826834323650897717284599840304;
            static constexpr double cos_1_32 = 0.98078528040323044912618223613424;
            static constexpr double sin_1_32 = 0.19509032201612826784828486847702;
            static constexpr double cos_3_32 = 0.83146961230254523707878837761791;
            static constexpr double sin_3_32 = 0.55557023301960222474283081394853;
            static constexpr Complex w1(cos_1_32, -sin_1_32), w2(cos_1_16, -sin_1_16), w3(cos_3_32, -sin_3_32);
            static constexpr Complex w5(sin_3_32, -cos_3_32), w6(sin_1_16, -cos_1_16), w7(sin_1_32, -cos_1_32);
            static constexpr Complex w9(-sin_1_32, -cos_1_32), w10(-sin_1_16, -cos_1_16), w11(-sin_3_32, -cos_3_32);
            static constexpr Complex w13(-cos_3_32, -sin_3_32), w14(-cos_1_16, -sin_1_16), w15(-cos_1_32, -sin_1_32);

            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 16];
            Complex tmp3 = input[rank * 17];
            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 16] = tmp0 - tmp2;
            input[rank * 17] = (tmp1 - tmp3) * w1;

            tmp0 = input[rank * 2];
            tmp1 = input[rank * 3];
            tmp2 = input[rank * 18];
            tmp3 = input[rank * 19];
            input[rank * 2] = tmp0 + tmp2;
            input[rank * 3] = tmp1 + tmp3;
            input[rank * 18] = (tmp0 - tmp2) * w2;
            input[rank * 19] = (tmp1 - tmp3) * w3;

            tmp0 = input[rank * 4];
            tmp1 = input[rank * 5];
            tmp2 = input[rank * 20];
            tmp3 = input[rank * 21];
            fft_2point(tmp0, tmp2);
            tmp2 = cos_1_8 * Complex(tmp2.imag() + tmp2.real(), tmp2.imag() - tmp2.real());
            input[rank * 4] = tmp0;
            input[rank * 5] = tmp1 + tmp3;
            input[rank * 20] = tmp2;
            input[rank * 21] = (tmp1 - tmp3) * w5;

            tmp0 = input[rank * 6];
            tmp1 = input[rank * 7];
            tmp2 = input[rank * 22];
            tmp3 = input[rank * 23];
            input[rank * 6] = tmp0 + tmp2;
            input[rank * 7] = tmp1 + tmp3;
            input[rank * 22] = (tmp0 - tmp2) * w6;
            input[rank * 23] = (tmp1 - tmp3) * w7;

            tmp0 = input[rank * 8];
            tmp1 = input[rank * 9];
            tmp2 = input[rank * 24];
            tmp3 = input[rank * 25];
            fft_2point(tmp0, tmp2);
            tmp2 = Complex(tmp2.imag(), -tmp2.real());
            input[rank * 8] = tmp0;
            input[rank * 9] = tmp1 + tmp3;
            input[rank * 24] = tmp2;
            input[rank * 25] = (tmp1 - tmp3) * w9;

            tmp0 = input[rank * 10];
            tmp1 = input[rank * 11];
            tmp2 = input[rank * 26];
            tmp3 = input[rank * 27];
            input[rank * 10] = tmp0 + tmp2;
            input[rank * 11] = tmp1 + tmp3;
            input[rank * 26] = (tmp0 - tmp2) * w10;
            input[rank * 27] = (tmp1 - tmp3) * w11;

            tmp0 = input[rank * 12];
            tmp1 = input[rank * 13];
            tmp2 = input[rank * 28];
            tmp3 = input[rank * 29];
            fft_2point(tmp0, tmp2);
            tmp2 = -cos_1_8 * Complex(tmp2.real() - tmp2.imag(), tmp2.real() + tmp2.imag());
            input[rank * 12] = tmp0;
            input[rank * 13] = tmp1 + tmp3;
            input[rank * 28] = tmp2;
            input[rank * 29] = (tmp1 - tmp3) * w13;

            tmp0 = input[rank * 14];
            tmp1 = input[rank * 15];
            tmp2 = input[rank * 30];
            tmp3 = input[rank * 31];

            input[rank * 14] = tmp0 + tmp2;
            input[rank * 15] = tmp1 + tmp3;
            input[rank * 30] = (tmp0 - tmp2) * w14;
            input[rank * 31] = (tmp1 - tmp3) * w15;

            fft_dif_16point(input, rank);
            fft_dif_16point(input + rank * 16, rank);
        }

        // fft基2时间抽取蝶形变换
        inline void fft_radix2_dit_butterfly(Complex omega, Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank] * omega;
            input[0] = tmp0 + tmp1;
            input[rank] = tmp0 - tmp1;
        }
        // fft基2频率抽取蝶形变换
        inline void fft_radix2_dif_butterfly(Complex omega, Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            input[0] = tmp0 + tmp1;
            input[rank] = (tmp0 - tmp1) * omega;
        }

        // fft分裂基时间抽取蝶形变换
        inline void fft_split_radix_dit_butterfly(Complex omega, Complex omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2] * omega;
            Complex tmp3 = input[rank * 3] * omega_cube;

            fft_2point(tmp2, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 2] = tmp0 - tmp2;
            input[rank * 3] = tmp1 - tmp3;
        }
        // fft分裂基频率抽取蝶形变换
        inline void fft_split_radix_dif_butterfly(Complex omega, Complex omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(-tmp3.imag(), tmp3.real());

            input[0] = tmp0;
            input[rank] = tmp1;
            input[rank * 2] = (tmp2 - tmp3) * omega;
            input[rank * 3] = (tmp2 + tmp3) * omega_cube;
        }
        // fft基4时间抽取蝶形变换
        inline void fft_radix4_dit_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                             Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank] * omega;
            Complex tmp2 = input[rank * 2] * omega_sqr;
            Complex tmp3 = input[rank * 3] * omega_cube;

            Complex t0 = tmp0 + tmp2;
            Complex t1 = tmp1 + tmp3;
            Complex t2 = tmp0 - tmp2;
            Complex t3 = tmp1 - tmp3;
            t3 = Complex(t3.imag(), -t3.real());

            input[0] = t0 + t1;
            input[rank] = t2 + t3;
            input[rank * 2] = t0 - t1;
            input[rank * 3] = t2 - t3;
        }
        // fft基4频率抽取蝶形变换
        inline void fft_radix4_dif_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                             Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            Complex t0 = tmp0 + tmp2;
            Complex t1 = tmp1 + tmp3;
            Complex t2 = tmp0 - tmp2;
            Complex t3 = tmp1 - tmp3;
            t3 = Complex(t3.imag(), -t3.real());

            input[0] = t0 + t1;
            input[rank] = (t2 + t3) * omega;
            input[rank * 2] = (t0 - t1) * omega_sqr;
            input[rank * 3] = (t2 - t3) * omega_cube;
        }
        // 求共轭复数及归一化，逆变换用
        inline void fft_conj(Complex *input, size_t fft_len, double div = 1)
        {
            for (size_t i = 0; i < fft_len; i++)
            {
                input[i] = std::conj(input[i]) / div;
            }
        }
        // 归一化,逆变换用
        inline void fft_normalize(Complex *input, size_t fft_len)
        {
            double len = static_cast<double>(fft_len);
            for (size_t i = 0; i < fft_len; i++)
            {
                input[i] /= len;
            }
        }
        // 基2查时间抽取FFT
        void fft_radix2_dit_lut(Complex *input, size_t fft_len, bool bit_inv = true)
        {
            if (fft_len <= 1)
            {
                return;
            }
            if (fft_len == 2)
            {
                fft_2point(input[0], input[1]);
                return;
            }
            if (bit_inv)
            {
                binary_inverse_swap(input, fft_len);
            }
            TABLE.expend(hint_log2(fft_len));
            for (size_t i = 0; i < fft_len; i += 2)
            {
                fft_2point(input[i], input[i + 1]);
            }
            INT_32 log_len = 2;
            for (size_t rank = 2; rank < fft_len / 2; rank *= 2)
            {
                size_t gap = rank * 2;
                for (size_t begin = 0; begin < fft_len; begin += (gap * 2))
                {
                    fft_2point(input[begin], input[begin + rank]);
                    fft_2point(input[gap + begin], input[gap + begin + rank]);
                    for (size_t pos = begin + 1; pos < begin + rank; pos++)
                    {
                        Complex omega = TABLE.get_complex_conj(log_len, pos - begin);

                        fft_radix2_dit_butterfly(omega, input + pos, rank);
                        fft_radix2_dit_butterfly(omega, input + pos + gap, rank);
                    }
                }
                log_len++;
            }
            fft_len /= 2;
            for (size_t pos = 0; pos < fft_len; pos++)
            {
                Complex omega = TABLE.get_complex_conj(log_len, pos);
                fft_radix2_dit_butterfly(omega, input + pos, fft_len);
            }
        }
        // 基2查频率抽取FFT
        void fft_radix2_dif_lut(Complex *input, size_t fft_len, const bool bit_inv = true)
        {
            if (fft_len <= 1)
            {
                return;
            }
            if (fft_len == 2)
            {
                fft_2point(input[0], input[1]);
                return;
            }
            INT_32 log_len = hint_log2(fft_len);
            TABLE.expend(log_len);
            fft_len /= 2;
            for (size_t pos = 0; pos < fft_len; pos++)
            {
                Complex omega = TABLE.get_complex_conj(log_len, pos);
                fft_radix2_dif_butterfly(omega, input + pos, fft_len);
            }
            fft_len *= 2;
            log_len--;
            for (size_t rank = fft_len / 4; rank > 1; rank /= 2)
            {
                size_t gap = rank * 2;
                for (size_t begin = 0; begin < fft_len; begin += (gap * 2))
                {
                    fft_2point(input[begin], input[begin + rank]);
                    fft_2point(input[gap + begin], input[gap + begin + rank]);
                    for (size_t pos = begin + 1; pos < begin + rank; pos++)
                    {
                        Complex omega = TABLE.get_complex_conj(log_len, pos - begin);
                        fft_radix2_dif_butterfly(omega, input + pos, rank);
                        fft_radix2_dif_butterfly(omega, input + pos + gap, rank);
                    }
                }
                log_len--;
            }
            for (size_t i = 0; i < fft_len; i += 2)
            {
                fft_2point(input[i], input[i + 1]);
            }
            if (bit_inv)
            {
                binary_inverse_swap(input, fft_len);
            }
        }
#ifdef FFT_R2_TEMPLATE
        // FFT最外层循环模板展开
        template <size_t RANK, size_t LEN>
        struct FFT_LOOP1
        {
            static constexpr size_t gap = RANK * 2;
            static constexpr size_t log_len = hint_log2(gap);
            static void fft_dit_loop1(Complex *input)
            {
                FFT_LOOP1<RANK / 2, LEN>::fft_dit_loop1(input);
                for (size_t begin = 0; begin < LEN; begin += (gap * 2))
                {
                    fft_2point(input[begin], input[begin + RANK]);
                    fft_2point(input[gap + begin], input[gap + begin + RANK]);
                    for (size_t pos = begin + 1; pos < begin + RANK; pos++)
                    {
                        Complex omega = TABLE.get_complex_conj(log_len, pos - begin);

                        fft_radix2_dit_butterfly(omega, input + pos, RANK);
                        fft_radix2_dit_butterfly(omega, input + pos + gap, RANK);
                    }
                }
            }
            static void fft_dif_loop1(Complex *input)
            {
                for (size_t begin = 0; begin < LEN; begin += (gap * 2))
                {
                    fft_2point(input[begin], input[begin + RANK]);
                    fft_2point(input[gap + begin], input[gap + begin + RANK]);
                    for (size_t pos = begin + 1; pos < begin + RANK; pos++)
                    {
                        Complex omega = TABLE.get_complex_conj(log_len, pos - begin);
                        fft_radix2_dif_butterfly(omega, input + pos, RANK);
                        fft_radix2_dif_butterfly(omega, input + pos + gap, RANK);
                    }
                }
                FFT_LOOP1<RANK / 2, LEN>::fft_dif_loop1(input);
            }
        };
        template <size_t LEN>
        struct FFT_LOOP1<1, LEN>
        {
        public:
            static void fft_dit_loop1(Complex *input) {}
            static void fft_dif_loop1(Complex *input) {}
        };

        // 模板化时间抽取基2FFT
        template <size_t LEN>
        void fft_radix2_dit_template(Complex *input)
        {
            for (size_t i = 0; i < LEN; i += 2)
            {
                fft_2point(input[i], input[i + 1]);
            }
            FFT_LOOP1<LEN / 4, LEN>::fft_dit_loop1(input);
            constexpr INT_32 log_len = hint_log2(LEN);
            for (size_t pos = 0; pos < LEN / 2; pos++)
            {
                Complex omega = TABLE.get_complex_conj(log_len, pos);
                fft_radix2_dit_butterfly(omega, input + pos, LEN / 2);
            }
        }
        template <>
        void fft_radix2_dit_template<0>(Complex *input) {}
        template <>
        void fft_radix2_dit_template<1>(Complex *input) {}
        template <>
        void fft_radix2_dit_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
        template <size_t LEN>
        // 模板化频率抽取基2FFT
        void fft_radix2_dif_template(Complex *input)
        {
            constexpr INT_32 log_len = hint_log2(LEN);
            for (size_t pos = 0; pos < LEN / 2; pos++)
            {
                Complex omega = TABLE.get_complex_conj(log_len, pos);
                fft_radix2_dif_butterfly(omega, input + pos, LEN / 2);
            }
            FFT_LOOP1<LEN / 4, LEN>::fft_dif_loop1(input);
            for (size_t i = 0; i < LEN; i += 2)
            {
                fft_2point(input[i], input[i + 1]);
            }
        }
        template <>
        void fft_radix2_dif_template<0>(Complex *input) {}
        template <>
        void fft_radix2_dif_template<1>(Complex *input) {}
        template <>
        void fft_radix2_dif_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
#endif
        // 模板化时间抽取分裂基fft
        template <size_t LEN>
        void fft_split_radix_dit_template(Complex *input)
        {
            constexpr size_t log_len = hint_log2(LEN);
            constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
            fft_split_radix_dit_template<half_len>(input);
            fft_split_radix_dit_template<quarter_len>(input + half_len);
            fft_split_radix_dit_template<quarter_len>(input + half_len + quarter_len);
            for (size_t i = 0; i < quarter_len; i++)
            {
                Complex omega0 = TABLE.get_complex_conj(log_len, i);
                Complex omega_cube0 = TABLE.get_complex_conj(log_len, i * 3);
                fft_split_radix_dit_butterfly(omega0, omega_cube0, input + i, quarter_len);
            }
        }
        template <>
        void fft_split_radix_dit_template<0>(Complex *input) {}
        template <>
        void fft_split_radix_dit_template<1>(Complex *input) {}
        template <>
        void fft_split_radix_dit_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
        template <>
        void fft_split_radix_dit_template<4>(Complex *input)
        {
            fft_dit_4point(input, 1);
        }
        template <>
        void fft_split_radix_dit_template<8>(Complex *input)
        {
            fft_dit_8point(input, 1);
        }
        template <>
        void fft_split_radix_dit_template<16>(Complex *input)
        {
#ifdef FFT_R2_TEMPLATE
            fft_radix2_dit_template<16>(input);
#else
            fft_dit_16point(input, 1);
#endif
        }
        template <>
        void fft_split_radix_dit_template<32>(Complex *input)
        {
#ifdef FFT_R2_TEMPLATE
            fft_radix2_dit_template<32>(input);
#else
            fft_dit_32point(input, 1);
#endif
        }

        // 模板化频率抽取分裂基fft
        template <size_t LEN>
        void fft_split_radix_dif_template(Complex *input)
        {
            constexpr size_t log_len = hint_log2(LEN);
            constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
            for (size_t i = 0; i < quarter_len; i++)
            {
                Complex omega0 = TABLE.get_complex_conj(log_len, i);
                Complex omega_cube0 = TABLE.get_complex_conj(log_len, i * 3);
                fft_split_radix_dif_butterfly(omega0, omega_cube0, input + i, quarter_len);
            }
            fft_split_radix_dif_template<half_len>(input);
            fft_split_radix_dif_template<quarter_len>(input + half_len);
            fft_split_radix_dif_template<quarter_len>(input + half_len + quarter_len);
        }
        template <>
        void fft_split_radix_dif_template<0>(Complex *input) {}
        template <>
        void fft_split_radix_dif_template<1>(Complex *input) {}
        template <>
        void fft_split_radix_dif_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
        template <>
        void fft_split_radix_dif_template<4>(Complex *input)
        {
            fft_dif_4point(input, 1);
        }
        template <>
        void fft_split_radix_dif_template<8>(Complex *input)
        {
            fft_dif_8point(input, 1);
        }
        template <>
        void fft_split_radix_dif_template<16>(Complex *input)
        {
#ifdef FFT_R2_TEMPLATE
            fft_radix2_dif_template<16>(input);
#else
            fft_dif_16point(input, 1);
#endif
        }
        template <>
        void fft_split_radix_dif_template<32>(Complex *input)
        {
#ifdef FFT_R2_TEMPLATE
            fft_radix2_dif_template<32>(input);
#else
            fft_dif_32point(input, 1);
#endif
        }

        template <size_t LEN = 1>
        void fft_dit_template(Complex *input, size_t fft_len, bool bit_inv = true)
        {
            if (fft_len > LEN)
            {
                fft_dit_template<LEN * 2>(input, fft_len, bit_inv);
                return;
            }
            TABLE.expend(hint_log2(LEN));
            if (bit_inv)
            {
                binary_inverse_swap(input, LEN);
            }
            fft_split_radix_dit_template<LEN>(input);
        }
        template <>
        void fft_dit_template<1 << 24>(Complex *input, size_t fft_len, bool bit_inv) {}

        template <size_t LEN = 1>
        void fft_dif_template(Complex *input, size_t fft_len, bool bit_inv = true)
        {
            if (fft_len > LEN)
            {
                fft_dif_template<LEN * 2>(input, fft_len, bit_inv);
                return;
            }
            TABLE.expend(hint_log2(LEN));
            fft_split_radix_dif_template<LEN>(input);
            if (bit_inv)
            {
                binary_inverse_swap(input, LEN);
            }
        }
        template <>
        void fft_dif_template<1 << 24>(Complex *input, size_t fft_len, bool is_ifft) {}

        /// @brief 时间抽取基2fft
        /// @param input 复数组
        /// @param fft_len 数组长度
        /// @param bit_inv 是否逆序
        inline void fft_dit(Complex *input, size_t fft_len, bool bit_inv = true)
        {
            fft_len = max_2pow(fft_len);
            fft_dit_template<1>(input, fft_len, bit_inv);
        }

        /// @brief 频率抽取基2fft
        /// @param input 复数组
        /// @param fft_len 数组长度
        /// @param bit_inv 是否逆序
        inline void fft_dif(Complex *input, size_t fft_len, bool bit_inv = true)
        {
            fft_len = max_2pow(fft_len);
            fft_dif_template<1>(input, fft_len, bit_inv);
        }
        /// @brief 快速傅里叶变换
        /// @param input 复数组
        /// @param fft_len 变换长度
        /// @param r4_bit_inv 基4是否进行比特逆序,与逆变换同时设为false可以提高性能
        inline void fft(Complex *input, size_t fft_len, const bool bit_inv = true)
        {
            size_t log_len = hint_log2(fft_len);
            fft_len = 1ull << log_len;
            if (fft_len <= 1)
            {
                return;
            }
            fft_dif(input, fft_len, bit_inv);
        }
        /// @brief 快速傅里叶逆变换
        /// @param input 复数组
        /// @param fft_len 变换长度
        /// @param r4_bit_inv 基4是否进行比特逆序,与逆变换同时设为false可以提高性能
        inline void ifft(Complex *input, size_t fft_len, const bool bit_inv = true)
        {
            size_t log_len = hint_log2(fft_len);
            fft_len = 1ull << log_len;
            if (fft_len <= 1)
            {
                return;
            }
            fft_len = max_2pow(fft_len);
            fft_conj(input, fft_len);
            fft_dit(input, fft_len, bit_inv);
            fft_conj(input, fft_len, fft_len);
        }
    }
}

using namespace std;
using namespace hint;
using namespace hint_transform;

template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2 - 1;
    vector<T> result(out_len);
    size_t fft_len = min_2pow(out_len);
    Complex *fft_ary = new Complex[fft_len];
    com_ary_combine_copy(fft_ary, in1, len1, in2, len2);
    // fft_radix2_dif_lut(fft_ary, fft_len, false); // 经典FFT
    fft_dif(fft_ary, fft_len, false); // 优化FFT
    for (size_t i = 0; i < fft_len; i++)
    {
        Complex tmp = fft_ary[i];
        tmp *= tmp;
        fft_ary[i] = std::conj(tmp);
    }
    // fft_radix2_dit_lut(fft_ary, fft_len, false); // 经典FFT
    fft_dit(fft_ary, fft_len, false); // 优化FFT
    double inv = -0.5 / fft_len;
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(fft_ary[i].imag() * inv + 0.5);
    }
    delete[] fft_ary;
    return result;
}

int main()
{
    StopWatch w(1000);
    int n = 0;
    cin >> n;
    size_t len = 1 << n;
    uint64_t ele = 9;
    vector<uint32_t> in1(len / 2, ele);
    vector<uint32_t> in2(len / 2, ele);
    w.start();
    vector<uint32_t> res = poly_multiply(in1, in2);
    w.stop();
    // 验证卷积结果
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << (i + 1) * ele * ele << "\t" << y << "\n";
            break;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            break;
        }
    }
    cout << w.duration() << "ms\n";
    // cin.get();
    return 0;
}