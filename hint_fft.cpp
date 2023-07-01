#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>
#include "stopwatch.hpp"

#define MULTITHREAD 0   // 多线程 0 means no, 1 means yes
#define TABLE_TYPE 1    // 复数表的类型 0 means ComplexTable, 1 means ComplexTableX
#define TABLE_PRELOAD 1 // 是否提前初始化表 0 means no, 1 means yes

namespace hint
{
    using UINT_8 = uint8_t;
    using UINT_16 = uint16_t;
    using UINT_32 = uint32_t;
    using UINT_64 = uint64_t;
    using INT_32 = int32_t;
    using INT_64 = int64_t;
    using ULONG = unsigned long;
    using LONG = long;
    using HintFloat = double;
    using Complex = std::complex<HintFloat>;

    constexpr UINT_64 HINT_CHAR_BIT = 8;
    constexpr UINT_64 HINT_SHORT_BIT = 16;
    constexpr UINT_64 HINT_INT_BIT = 32;
    constexpr UINT_64 HINT_INT8_0XFF = UINT8_MAX;
    constexpr UINT_64 HINT_INT8_0X10 = (UINT8_MAX + 1ull);
    constexpr UINT_64 HINT_INT16_0XFF = UINT16_MAX;
    constexpr UINT_64 HINT_INT16_0X10 = (UINT16_MAX + 1ull);
    constexpr UINT_64 HINT_INT32_0XFF = UINT32_MAX;
    constexpr UINT_64 HINT_INT32_0X01 = 1;
    constexpr UINT_64 HINT_INT32_0X80 = 0X80000000ull;
    constexpr UINT_64 HINT_INT32_0X7F = INT32_MAX;
    constexpr UINT_64 HINT_INT32_0X10 = (UINT32_MAX + 1ull);
    constexpr UINT_64 HINT_INT64_0X80 = INT64_MIN;
    constexpr UINT_64 HINT_INT64_0X7F = INT64_MAX;
    constexpr UINT_64 HINT_INT64_0XFF = UINT64_MAX;

    constexpr HintFloat HINT_PI = 3.1415926535897932384626433832795;
    constexpr HintFloat HINT_2PI = HINT_PI * 2;
    constexpr HintFloat HINT_HSQ_ROOT2 = 0.70710678118654752440084436210485;

    constexpr UINT_64 NTT_MOD1 = 3221225473;
    constexpr UINT_64 NTT_ROOT1 = 5;
    constexpr UINT_64 NTT_MOD2 = 3489660929;
    constexpr UINT_64 NTT_ROOT2 = 3;
    constexpr size_t NTT_MAX_LEN = 1ull << 28;

    /// @brief 生成不大于n的最大的2的幂次的数
    /// @param n
    /// @return 不大于n的最大的2的幂次的数
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
    /// @brief 生成不小于n的最小的2的幂次的数
    /// @param n
    /// @return 不小于n的最小的2的幂次的数
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
    constexpr T hint_log2(T n)
    {
        T res = 0;
        while (n > 1)
        {
            n /= 2;
            res++;
        }
        return res;
    }
#if MULTITHREAD == 1
    const UINT_32 hint_threads = std::thread::hardware_concurrency();
    const UINT_32 log2_threads = std::ceil(hint_log2(hint_threads));
    std::atomic<UINT_32> cur_ths;
#endif

    // 模板数组拷贝
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
    // 数组交错重排
    template <UINT_64 N, typename T>
    void ary_interlace(T ary[], size_t len)
    {
        size_t sub_len = len / N;
        T *tmp_ary = new T[len - sub_len];
        for (size_t i = 0; i < len; i += N)
        {
            ary[i / N] = ary[i];
            for (size_t j = 0; j < N - 1; j++)
            {
                tmp_ary[j * sub_len + i / N] = ary[i + j + 1];
            }
        }
        ary_copy(ary + sub_len, tmp_ary, len - sub_len);
        delete[] tmp_ary;
    }
    // FFT与类FFT变换的命名空间
    namespace hint_transform
    {
        // 二进制逆序
        template <typename T>
        void binary_reverse_swap(T &ary, size_t len)
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
        // 四进制逆序
        template <typename SizeType = UINT_32, typename T>
        void quaternary_reverse_swap(T &ary, size_t len)
        {
            SizeType log_n = hint_log2(len);
            SizeType *rev = new SizeType[len / 4];
            rev[0] = 0;
            for (SizeType i = 1; i < len; i++)
            {
                SizeType index = (rev[i >> 2] >> 2) | ((i & 3) << (log_n - 2)); // 求rev交换数组
                if (i < len / 4)
                {
                    rev[i] = index;
                }
                if (i < index)
                {
                    std::swap(ary[i], ary[index]);
                }
            }
            delete[] rev;
        }
        namespace hint_fft
        {
            // 返回单位圆上辐角为theta的点
            static Complex unit_root(HintFloat theta)
            {
                return std::polar<HintFloat>(1.0, theta);
            }
            // 返回单位圆上平分m份的第n个
            static Complex unit_root(size_t m, size_t n)
            {
                return unit_root((HINT_2PI * n) / m);
            }
            class ComplexTable
            {
            private:
                std::vector<Complex> table;
                INT_32 max_log_size = 2;
                INT_32 cur_log_size = 2;

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
#if TABLE_PRELOAD == 1
                    expand(max_shift);
#endif
                }
                void expand(INT_32 shift)
                {
                    if (shift > max_log_size)
                    {
                        throw("FFT length too long for lut\n");
                    }
                    for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                    {
                        size_t len = 1ull << i, vec_size = len / 4;
                        table[vec_size] = Complex(1, 0);
                        for (size_t pos = 0; pos < vec_size / 2; pos++)
                        {
                            table[vec_size + pos * 2] = table[vec_size / 2 + pos];
                            if (pos % 2 == 1)
                            {
                                Complex tmp = unit_root(len, pos);
                                table[vec_size + pos] = tmp;
                                table[vec_size * 2 - pos] = Complex(tmp.imag(), tmp.real());
                            }
                        }
                        table[vec_size + vec_size / 2] = unit_root(8, 1);
                    }
                    cur_log_size = std::max(cur_log_size, shift);
                }
                // shift表示圆平分为1<<shift份,n表示第几个单位根
                Complex get_omega(UINT_32 shift, size_t n) const
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
                INT_32 max_log_size = 2;
                INT_32 cur_log_size = 2;

                ComplexTableX(const ComplexTableX &) = delete;
                ComplexTableX &operator=(const ComplexTableX &) = delete;

            public:
                ~ComplexTableX() {}
                // 初始化可以生成平分圆1<<shift份产生的单位根的表
                ComplexTableX(UINT_32 max_shift)
                {
                    max_shift = std::max<size_t>(max_shift, cur_log_size);
                    max_log_size = max_shift;
                    table.resize(max_shift + 1);
                    table[0] = table[1] = std::vector<Complex>{1};
                    table[2] = std::vector<Complex>{Complex(1, 0), Complex(0, -1), Complex(-1, 0)};
#if TABLE_PRELOAD == 1
                    expand(max_shift);
#endif
                }
                void expand(INT_32 shift)
                {
                    if (shift > max_log_size)
                    {
                        throw("FFT length too long for lut\n");
                    }
                    for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                    {
                        size_t len = 1ull << i, vec_size = len * 3 / 4;
                        table[i].resize(vec_size);
                        for (size_t pos = 0; pos < len / 8; pos++)
                        {
                            table[i][pos * 2] = table[i - 1][pos];
                            table[i][pos * 2 + len / 4] = table[i - 1][pos + len / 8];
                            table[i][pos * 2 + len / 2] = table[i - 1][pos + len / 4];
                            if (pos % 2 == 1)
                            {
                                HintFloat cos_theta = std::cos(HINT_2PI * pos / len);
                                HintFloat sin_theta = std::sin(HINT_2PI * pos / len);
                                Complex tmp1(cos_theta, -sin_theta);
                                Complex tmp2(sin_theta, -cos_theta);

                                table[i][pos] = tmp1;
                                table[i][pos + len / 4] = Complex(tmp1.imag(), -tmp1.real());
                                table[i][pos + len / 2] = -tmp1;

                                table[i][len / 4 - pos] = tmp2;
                                table[i][len / 2 - pos] = Complex(tmp2.imag(), -tmp2.real());
                                table[i][vec_size - pos] = -tmp2;
                            }
                        }
                        table[i][len / 8] = std::conj(unit_root(8, 1));
                        table[i][len * 3 / 8] = std::conj(unit_root(8, 3));
                        table[i][len * 5 / 8] = std::conj(unit_root(8, 5));
                    }
                    cur_log_size = std::max(cur_log_size, shift);
                }
                // shift表示圆平分为1<<shift份,n表示第几个单位根
                Complex get_omega(UINT_32 shift, size_t n) const
                {
                    return table[shift][n];
                }
            };

            constexpr size_t lut_max_rank = 23;
#if TABLE_TYPE == 0
            static ComplexTable TABLE(lut_max_rank);
#elif TABLE_TYPE == 1
            static ComplexTableX TABLE(lut_max_rank);
#else
#error TABLE_TYPE must be 0,1
#endif
            // 2点fft
            template <typename T>
            inline void fft_2point(T &sum, T &diff)
            {
                T tmp0 = sum;
                T tmp1 = diff;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                static constexpr HintFloat cos_1_16 = 0.92387953251128675612818318939679;
                static constexpr HintFloat sin_1_16 = 0.3826834323650897717284599840304;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                static constexpr HintFloat cos_1_16 = 0.92387953251128675612818318939679;
                static constexpr HintFloat sin_1_16 = 0.3826834323650897717284599840304;
                static constexpr HintFloat cos_1_32 = 0.98078528040323044912618223613424;
                static constexpr HintFloat sin_1_32 = 0.19509032201612826784828486847702;
                static constexpr HintFloat cos_3_32 = 0.83146961230254523707878837761791;
                static constexpr HintFloat sin_3_32 = 0.55557023301960222474283081394853;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                static constexpr HintFloat cos_1_16 = 0.92387953251128675612818318939679;
                static constexpr HintFloat sin_1_16 = 0.3826834323650897717284599840304;
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
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                static constexpr HintFloat cos_1_16 = 0.92387953251128675612818318939679;
                static constexpr HintFloat sin_1_16 = 0.3826834323650897717284599840304;
                static constexpr HintFloat cos_1_32 = 0.98078528040323044912618223613424;
                static constexpr HintFloat sin_1_32 = 0.19509032201612826784828486847702;
                static constexpr HintFloat cos_3_32 = 0.83146961230254523707878837761791;
                static constexpr HintFloat sin_3_32 = 0.55557023301960222474283081394853;
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
                tmp3 = Complex(tmp3.imag(), -tmp3.real());

                input[0] = tmp0;
                input[rank] = tmp1;
                input[rank * 2] = (tmp2 + tmp3) * omega;
                input[rank * 3] = (tmp2 - tmp3) * omega_cube;
            }
            // fft基4时间抽取蝶形变换
            inline void fft_radix4_dit_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                                 Complex *input, size_t rank)
            {
                Complex tmp0 = input[0];
                Complex tmp1 = input[rank] * omega;
                Complex tmp2 = input[rank * 2] * omega_sqr;
                Complex tmp3 = input[rank * 3] * omega_cube;

                fft_2point(tmp0, tmp2);
                fft_2point(tmp1, tmp3);
                tmp3 = Complex(tmp3.imag(), -tmp3.real());

                input[0] = tmp0 + tmp1;
                input[rank] = tmp2 + tmp3;
                input[rank * 2] = tmp0 - tmp1;
                input[rank * 3] = tmp2 - tmp3;
            }
            // fft基4频率抽取蝶形变换
            inline void fft_radix4_dif_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                                 Complex *input, size_t rank)
            {
                Complex tmp0 = input[0];
                Complex tmp1 = input[rank];
                Complex tmp2 = input[rank * 2];
                Complex tmp3 = input[rank * 3];

                fft_2point(tmp0, tmp2);
                fft_2point(tmp1, tmp3);
                tmp3 = Complex(tmp3.imag(), -tmp3.real());

                input[0] = tmp0 + tmp1;
                input[rank] = (tmp2 + tmp3) * omega;
                input[rank * 2] = (tmp0 - tmp1) * omega_sqr;
                input[rank * 3] = (tmp2 - tmp3) * omega_cube;
            }
            // 求共轭复数及归一化，逆变换用
            inline void fft_conj(Complex *input, size_t fft_len, HintFloat div = 1)
            {
                div = 1.0 / div;
                for (size_t i = 0; i < fft_len; i++)
                {
                    input[i] = std::conj(input[i]) * div;
                }
            }
            // 归一化,逆变换用
            inline void fft_normalize(Complex *input, size_t fft_len)
            {
                HintFloat len = static_cast<HintFloat>(fft_len);
                for (size_t i = 0; i < fft_len; i++)
                {
                    input[i] /= len;
                }
            }
            // 经典模板,学习用
            void fft_radix2_dit(Complex *input, size_t fft_len)
            {
                fft_len = max_2pow(fft_len);
                binary_reverse_swap(input, fft_len);
                for (size_t rank = 1; rank < fft_len; rank *= 2)
                {
                    // rank表示上一级fft的长度,gap表示由两个上一级可以迭代计算出这一级的长度
                    size_t gap = rank * 2;
                    Complex unit_omega = std::polar<HintFloat>(1, -HINT_2PI / gap);
                    for (size_t begin = 0; begin < fft_len; begin += gap)
                    {
                        Complex omega = Complex(1, 0);
                        for (size_t pos = begin; pos < begin + rank; pos++)
                        {
                            fft_radix2_dit_butterfly(omega, input + pos, rank);
                            omega *= unit_omega;
                        }
                    }
                }
            }
            // 基4快速傅里叶变换,模板,学习用
            void fft_radix4_dit(Complex *input, size_t fft_len)
            {
                size_t log4_len = hint_log2(fft_len) / 2;
                fft_len = 1ull << (log4_len * 2);
                quaternary_reverse_swap(input, fft_len);
                for (size_t pos = 0; pos < fft_len; pos += 4)
                {
                    fft_4point(input + pos, 1);
                }
                for (size_t rank = 4; rank < fft_len; rank *= 4)
                {
                    // rank表示上一级fft的长度,gap表示由四个上一级可以迭代计算出这一级的长度
                    size_t gap = rank * 4;
                    Complex unit_omega = std::polar<HintFloat>(1, -HINT_2PI / gap);
                    Complex unit_sqr = std::polar<HintFloat>(1, -HINT_2PI * 2 / gap);
                    Complex unit_cube = std::polar<HintFloat>(1, -HINT_2PI * 3 / gap);
                    for (size_t begin = 0; begin < fft_len; begin += gap)
                    {
                        fft_4point(input + begin, rank);
                        Complex omega = unit_omega;
                        Complex omega_sqr = unit_sqr;
                        Complex omega_cube = unit_cube;
                        for (size_t pos = begin + 1; pos < begin + rank; pos++)
                        {
                            fft_radix4_dit_butterfly(omega, omega_sqr, omega_cube, input + pos, rank);
                            omega *= unit_omega;
                            omega_sqr *= unit_sqr;
                            omega_cube *= unit_cube;
                        }
                    }
                }
            }
            // 基2查表时间抽取FFT
            void fft_radix2_dit_lut(Complex *input, size_t fft_len, bool bit_rev = true)
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
                if (bit_rev)
                {
                    binary_reverse_swap(input, fft_len);
                }
                TABLE.expand(hint_log2(fft_len));
                for (size_t i = 0; i < fft_len; i += 2)
                {
                    fft_2point(input[i], input[i + 1]);
                }
                INT_32 log_len = 2;
                for (size_t rank = 2; rank < fft_len / 2; rank *= 2)
                {
                    size_t gap = rank * 2;
                    for (size_t begin = 0; begin < fft_len; begin += gap)
                    {
                        fft_2point(input[begin], input[begin + rank]);
                        for (size_t pos = begin + 1; pos < begin + rank; pos++)
                        {
                            Complex omega = TABLE.get_omega(log_len, pos - begin);
                            fft_radix2_dit_butterfly(omega, input + pos, rank);
                        }
                    }
                    log_len++;
                }
                fft_len /= 2;
                for (size_t pos = 0; pos < fft_len; pos++)
                {
                    Complex omega = TABLE.get_omega(log_len, pos);
                    fft_radix2_dit_butterfly(omega, input + pos, fft_len);
                }
            }
            // 基2查表频率抽取FFT
            void fft_radix2_dif_lut(Complex *input, size_t fft_len, const bool bit_rev = true)
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
                TABLE.expand(log_len);
                fft_len /= 2;
                for (size_t pos = 0; pos < fft_len; pos++)
                {
                    Complex omega = TABLE.get_omega(log_len, pos);
                    fft_radix2_dif_butterfly(omega, input + pos, fft_len);
                }
                fft_len *= 2;
                log_len--;
                for (size_t rank = fft_len / 4; rank > 1; rank /= 2)
                {
                    size_t gap = rank * 2;
                    for (size_t begin = 0; begin < fft_len; begin += gap)
                    {
                        fft_2point(input[begin], input[begin + rank]);
                        for (size_t pos = begin + 1; pos < begin + rank; pos++)
                        {
                            Complex omega = TABLE.get_omega(log_len, pos - begin);
                            fft_radix2_dif_butterfly(omega, input + pos, rank);
                        }
                    }
                    log_len--;
                }
                for (size_t i = 0; i < fft_len; i += 2)
                {
                    fft_2point(input[i], input[i + 1]);
                }
                if (bit_rev)
                {
                    binary_reverse_swap(input, fft_len);
                }
            }
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
                    Complex omega = TABLE.get_omega(log_len, i);
                    Complex omega_cube = TABLE.get_omega(log_len, i * 3);
                    fft_split_radix_dit_butterfly(omega, omega_cube, input + i, quarter_len);
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
                fft_dit_16point(input, 1);
            }
            template <>
            void fft_split_radix_dit_template<32>(Complex *input)
            {
                fft_dit_32point(input, 1);
            }

            // 模板化频率抽取分裂基fft
            template <size_t LEN>
            void fft_split_radix_dif_template(Complex *input)
            {
                constexpr size_t log_len = hint_log2(LEN);
                constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
                for (size_t i = 0; i < quarter_len; i++)
                {
                    Complex omega = TABLE.get_omega(log_len, i);
                    Complex omega_cube = TABLE.get_omega(log_len, i * 3);
                    fft_split_radix_dif_butterfly(omega, omega_cube, input + i, quarter_len);
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
                fft_dif_16point(input, 1);
            }
            template <>
            void fft_split_radix_dif_template<32>(Complex *input)
            {
                fft_dif_32point(input, 1);
            }

            // 辅助选择函数
            template <size_t LEN = 1>
            void fft_split_radix_dit_template_alt(Complex *input, size_t fft_len)
            {
                if (fft_len < LEN)
                {
                    fft_split_radix_dit_template_alt<LEN / 2>(input, fft_len);
                    return;
                }
                TABLE.expand(hint_log2(LEN));
                fft_split_radix_dit_template<LEN>(input);
            }
            template <>
            void fft_split_radix_dit_template_alt<0>(Complex *input, size_t fft_len) {}

            // 辅助选择函数
            template <size_t LEN = 1>
            void fft_split_radix_dif_template_alt(Complex *input, size_t fft_len)
            {
                if (fft_len < LEN)
                {
                    fft_split_radix_dif_template_alt<LEN / 2>(input, fft_len);
                    return;
                }
                TABLE.expand(hint_log2(LEN));
                fft_split_radix_dif_template<LEN>(input);
            }
            template <>
            void fft_split_radix_dif_template_alt<0>(Complex *input, size_t fft_len) {}

            auto fft_split_radix_dit = fft_split_radix_dit_template_alt<size_t(1) << lut_max_rank>;
            auto fft_split_radix_dif = fft_split_radix_dif_template_alt<size_t(1) << lut_max_rank>;

            /// @brief 时间抽取基2fft
            /// @param input 复数组
            /// @param fft_len 数组长度
            /// @param bit_rev 是否逆序
            inline void fft_dit(Complex *input, size_t fft_len, bool bit_rev = true)
            {
                fft_len = max_2pow(fft_len);
                if (bit_rev)
                {
                    binary_reverse_swap(input, fft_len);
                }
                fft_split_radix_dit(input, fft_len);
            }

            /// @brief 频率抽取基2fft
            /// @param input 复数组
            /// @param fft_len 数组长度
            /// @param bit_rev 是否逆序
            inline void fft_dif(Complex *input, size_t fft_len, bool bit_rev = true)
            {
                fft_len = max_2pow(fft_len);
                fft_split_radix_dif(input, fft_len);
                if (bit_rev)
                {
                    binary_reverse_swap(input, fft_len);
                }
            }

            /// @brief fft正变换
            /// @param input 复数组
            /// @param fft_len 变换长度
            inline void fft(Complex *input, size_t fft_len)
            {
                size_t log_len = hint_log2(fft_len);
                fft_len = 1ull << log_len;
                if (fft_len <= 1)
                {
                    return;
                }
                fft_dif(input, fft_len, true);
            }

            /// @brief fft逆变换
            /// @param input 复数组
            /// @param fft_len 变换长度
            inline void ifft(Complex *input, size_t fft_len)
            {
                size_t log_len = hint_log2(fft_len);
                fft_len = 1ull << log_len;
                if (fft_len <= 1)
                {
                    return;
                }
                fft_len = max_2pow(fft_len);
                fft_conj(input, fft_len);
                fft_dit(input, fft_len, true);
                fft_conj(input, fft_len, fft_len);
            }
#if MULTITHREAD == 1
            void fft_dit_2ths(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    fft_dit(input, fft_len);
                    return;
                }
                const size_t half_len = fft_len / 2;
                const INT_32 log_len = hint_log2(fft_len);
                TABLE.expand(log_len);
                auto th = std::async(fft_dit, input, half_len, false);
                fft_dit(input + half_len, half_len, false);
                th.wait();
                auto proc = [&](size_t start, size_t end)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        Complex omega = TABLE.get_omega(log_len, i);
                        fft_radix2_dit_butterfly(omega, input + i, half_len);
                    }
                };
                th = std::async(proc, 0, half_len / 2);
                proc(half_len / 2, half_len);
                th.wait();
            }
            void fft_dif_2ths(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    fft_dif(input, fft_len);
                    return;
                }
                const size_t half_len = fft_len / 2;
                const INT_32 log_len = hint_log2(fft_len);
                TABLE.expand(log_len);
                auto proc = [&](size_t start, size_t end)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        Complex omega = TABLE.get_omega(log_len, i);
                        fft_radix2_dif_butterfly(omega, input + i, half_len);
                    }
                };
                auto th = std::async(proc, 0, half_len / 2);
                proc(half_len / 2, half_len);
                th.wait();
                th = std::async(fft_dif, input, half_len, false);
                fft_dif(input + half_len, half_len, false);
                th.wait();
            }
            void fft_dit_4ths(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    fft_dit(input, fft_len);
                    return;
                }
                const size_t half_len = fft_len / 2;
                const INT_32 log_len = hint_log2(fft_len);
                TABLE.expand(log_len);
                auto th1 = std::async(fft_dit_2ths, input, half_len);
                fft_dit_2ths(input + half_len, half_len);
                th1.wait();

                auto proc = [&](size_t start, size_t end)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        Complex omega = TABLE.get_omega(log_len, i);
                        fft_radix2_dit_butterfly(omega, input + i, half_len);
                    }
                };
                const size_t sub_len = fft_len / 8;
                th1 = std::async(proc, 0, sub_len);
                auto th2 = std::async(proc, sub_len, sub_len * 2);
                auto th3 = std::async(proc, sub_len * 2, sub_len * 3);
                proc(sub_len * 3, sub_len * 4);
                th1.wait();
                th2.wait();
                th3.wait();
            }
            void fft_dif_4ths(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    fft_dif(input, fft_len);
                    return;
                }
                const size_t half_len = fft_len / 2;
                const INT_32 log_len = hint_log2(fft_len);
                TABLE.expand(log_len);
                auto proc = [&](size_t start, size_t end)
                {
                    for (size_t i = start; i < end; i++)
                    {
                        Complex omega = TABLE.get_omega(log_len, i);
                        fft_radix2_dif_butterfly(omega, input + i, half_len);
                    }
                };
                const size_t sub_len = fft_len / 8;
                auto th1 = std::async(proc, 0, sub_len);
                auto th2 = std::async(proc, sub_len, sub_len * 2);
                auto th3 = std::async(proc, sub_len * 2, sub_len * 3);
                proc(sub_len * 3, sub_len * 4);
                th1.wait();
                th2.wait();
                th3.wait();

                th1 = std::async(fft_dif_2ths, input, half_len);
                fft_dif_2ths(input + half_len, half_len);
                th1.wait();
            }
#endif
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
    size_t fft_len = min_2pow(out_len);
    Complex *fft_ary = new Complex[fft_len];
    com_ary_combine_copy(fft_ary, in1, len1, in2, len2);
#if MULTITHREAD == 1
    fft_dif_4ths(fft_ary, fft_len);
#else
    fft_dif(fft_ary, fft_len, false); // 优化FFT
#endif
    for (size_t i = 0; i < fft_len; i++)
    {
        Complex tmp = fft_ary[i];
        tmp *= tmp;
        fft_ary[i] = std::conj(tmp);
    }
#if MULTITHREAD == 1
    fft_dit_4ths(fft_ary, fft_len);
#else
    fft_dit(fft_ary, fft_len, false); // 优化FFT
#endif
    HintFloat rev = -0.5 / fft_len;
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(fft_ary[i].imag() * rev + 0.5);
    }
    delete[] fft_ary;
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
    StopWatch w(1000);
    int n = 18;
    cin >> n;
    size_t len = 1 << n; // 变换长度
    uint64_t ele = 9;
    vector<uint32_t> in1(len / 2, ele);
    vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    w.start();
    vector<uint32_t> res = poly_multiply(in1, in2);
    w.stop();
    result_test<uint32_t>(res, ele); // 结果校验
    cout << w.duration() << "ms\n";
}