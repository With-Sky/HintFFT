#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>

#define TABLE_TYPE 1    // 复数表的类型 0 means ComplexTable, 1 means ComplexTableE
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
    using HintFloat = float;
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
            class ComplexTableE
            {
            private:
                std::vector<std::vector<Complex>> table1;
                std::vector<std::vector<Complex>> table3;
                INT_32 max_log_size = 2;
                INT_32 cur_log_size = 2;

                static constexpr size_t FAC = 1;

                ComplexTableE(const ComplexTableE &) = delete;
                ComplexTableE &operator=(const ComplexTableE &) = delete;

            public:
                ~ComplexTableE() {}
                // 初始化可以生成平分圆1<<shift份产生的单位根的表
                ComplexTableE(UINT_32 max_shift)
                {
                    max_shift = std::max<size_t>(max_shift, 1);
                    max_log_size = max_shift;
                    table1.resize(max_shift + 1);
                    table3.resize(max_shift + 1);
                    table1[0] = table1[1] = table3[0] = table3[1] = std::vector<Complex>{1};
                    table1[2] = table3[2] = std::vector<Complex>{1};
#if TABLE_PRELOAD == 1
                    expand(max_shift);
#endif
                }
                void expand(INT_32 shift)
                {
                    shift = std::max<INT_32>(shift, 2);
                    if (shift > max_log_size)
                    {
                        throw("FFT length too long for lut\n");
                    }
                    for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                    {
                        size_t len = 1ull << i, vec_size = len * FAC / 4;
                        table1[i].resize(vec_size);
                        table3[i].resize(vec_size);
                        table1[i][0] = table3[i][0] = Complex(1, 0);
                        for (size_t pos = 0; pos < vec_size / 2; pos++)
                        {
                            table1[i][pos * 2] = table1[i - 1][pos];
                            table3[i][pos * 2] = table3[i - 1][pos];
                        }
                        for (size_t pos = 1; pos < vec_size / 2; pos += 2)
                        {
                            HintFloat cos_theta = std::cos(HINT_2PI * pos / len);
                            HintFloat sin_theta = std::sin(HINT_2PI * pos / len);
                            table1[i][pos] = Complex(cos_theta, -sin_theta);
                            table1[i][vec_size - pos] = Complex(sin_theta, -cos_theta);
                        }
                        table1[i][vec_size / 2] = std::conj(unit_root(8, 1));
                        for (size_t pos = 1; pos < vec_size / 2; pos += 2)
                        {
                            Complex tmp = get_full_omega(i, pos * 3);
                            table3[i][pos] = tmp;
                            table3[i][vec_size - pos] = Complex(tmp.imag(), tmp.real());
                        }
                        table3[i][vec_size / 2] = std::conj(unit_root(8, 3));
                    }
                    cur_log_size = std::max(cur_log_size, shift);
                }
                // shift表示圆平分为1<<shift份,n表示第几个单位根
                Complex get_full_omega(UINT_32 shift, size_t n) const
                {
                    size_t rank = 1ull << shift;
                    const size_t rank_ff = rank - 1, quad_n = n << 2;
                    // n &= rank_ff;
                    size_t zone = quad_n >> shift; // 第几象限
                    if ((quad_n & rank_ff) == 0)
                    {
                        static constexpr Complex ONES[4] = {Complex(1, 0), Complex(0, -1), Complex(-1, 0), Complex(0, 1)};
                        return ONES[zone];
                    }
                    Complex tmp;
                    if ((zone & 2) == 0)
                    {
                        if ((zone & 1) == 0)
                        {
                            tmp = table1[shift][n];
                        }
                        else
                        {
                            tmp = table1[shift][rank / 2 - n];
                            tmp.real(-tmp.real());
                        }
                    }
                    else
                    {
                        if ((zone & 1) == 0)
                        {
                            tmp = -table1[shift][n - rank / 2];
                        }
                        else
                        {
                            tmp = table1[shift][rank - n];
                            tmp.imag(-tmp.imag());
                        }
                    }
                    return tmp;
                }
                // shift表示圆平分为1<<shift份,n表示第几个单位根
                const Complex &get_omega(UINT_32 shift, size_t n) const
                {
                    return table1[shift][n];
                }
                // shift表示圆平分为1<<shift份,3n表示第几个单位根
                const Complex &get_omega3(UINT_32 shift, size_t n) const
                {
                    return table3[shift][n];
                }
            };

            constexpr size_t lut_max_rank = 23;
            static ComplexTableE TABLE(lut_max_rank);

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
            inline void fft_4point(Complex *input)
            {
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[4];
                HintFloat i1 = finput[5];
                HintFloat r2 = finput[2];
                HintFloat i2 = finput[3];
                HintFloat r3 = finput[6];
                HintFloat i3 = finput[7];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r1;
                i0 += i1;
                r1 = tmpr - r1;
                i1 = tmpi - i1;

                tmpr = r2;
                tmpi = i2;
                r2 += r3;
                i2 += i3;
                tmpr = tmpr - r3;
                r3 = tmpi - i3;
                i3 = -tmpr;

                finput[0] = r0 + r2;
                finput[1] = i0 + i2;
                finput[2] = r1 + r3;
                finput[3] = i1 + i3;
                finput[4] = r0 - r2;
                finput[5] = i0 - i2;
                finput[6] = r1 - r3;
                finput[7] = i1 - i3;
            }
            inline void fft_8point(Complex *cinput)
            {
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                HintFloat *finput = reinterpret_cast<HintFloat *>(cinput);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[8];
                HintFloat i1 = finput[9];
                HintFloat r2 = finput[4];
                HintFloat i2 = finput[5];
                HintFloat r3 = finput[12];
                HintFloat i3 = finput[13];
                HintFloat r4 = finput[2];
                HintFloat i4 = finput[3];
                HintFloat r5 = finput[10];
                HintFloat i5 = finput[11];
                HintFloat r6 = finput[6];
                HintFloat i6 = finput[7];
                HintFloat r7 = finput[14];
                HintFloat i7 = finput[15];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r1;
                i0 += i1;
                r1 = tmpr - r1;
                i1 = tmpi - i1;

                tmpr = r2;
                tmpi = i2;
                r2 += r3;
                i2 += i3;
                tmpr = tmpr - r3;
                r3 = tmpi - i3;
                i3 = -tmpr;

                tmpr = r4;
                tmpi = i4;
                r4 += r5;
                i4 += i5;
                r5 = tmpr - r5;
                i5 = tmpi - i5;

                tmpr = r6;
                tmpi = i6;
                r6 += r7;
                i6 += i7;
                tmpr = tmpr - r7;
                r7 = tmpi - i7;
                i7 = -tmpr;

                tmpr = r0;
                tmpi = i0;
                r0 += r2;
                i0 += i2;
                r2 = tmpr - r2;
                i2 = tmpi - i2;

                tmpr = r1;
                tmpi = i1;
                r1 += r3;
                i1 += i3;
                r3 = tmpr - r3;
                i3 = tmpi - i3;

                tmpr = r4;
                tmpi = i4;
                r4 += r6;
                i4 += i6;
                r6 = tmpr - r6;
                i6 = tmpi - i6;

                tmpr = r5;
                tmpi = i5;
                r5 += r7;
                i5 += i7;
                r7 = tmpr - r7;
                i7 = tmpi - i7;

                tmpr = r5 + i5;
                i5 -= r5;
                r5 = tmpr * cos_1_8;
                i5 *= cos_1_8;

                tmpr = r6;
                r6 = i6;
                i6 = -tmpr;

                tmpi = r7 + i7;
                r7 -= i7;
                r7 *= -cos_1_8;
                i7 = -tmpi * cos_1_8;

                finput[0] = r0 + r4;
                finput[1] = i0 + i4;
                finput[2] = r1 + r5;
                finput[3] = i1 + i5;
                finput[4] = r2 + r6;
                finput[5] = i2 + i6;
                finput[6] = r3 + r7;
                finput[7] = i3 + i7;
                finput[8] = r0 - r4;
                finput[9] = i0 - i4;
                finput[10] = r1 - r5;
                finput[11] = i1 - i5;
                finput[12] = r2 - r6;
                finput[13] = i2 - i6;
                finput[14] = r3 - r7;
                finput[15] = i3 - i7;
            }
            inline void fft_dit_4point(Complex *input)
            {
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[2];
                HintFloat i1 = finput[3];
                HintFloat r2 = finput[4];
                HintFloat i2 = finput[5];
                HintFloat r3 = finput[6];
                HintFloat i3 = finput[7];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r1;
                i0 += i1;
                r1 = tmpr - r1;
                i1 = tmpi - i1;

                tmpr = r2;
                tmpi = i2;
                r2 += r3;
                i2 += i3;
                r3 = tmpr - r3;
                i3 = tmpi - i3;

                tmpr = r3;
                r3 = i3;
                i3 = -tmpr;

                finput[0] = r0 + r2;
                finput[1] = i0 + i2;
                finput[2] = r1 + r3;
                finput[3] = i1 + i3;
                finput[4] = r0 - r2;
                finput[5] = i0 - i2;
                finput[6] = r1 - r3;
                finput[7] = i1 - i3;
            }
            inline void fft_dit_8point(Complex *input)
            {
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[2];
                HintFloat i1 = finput[3];
                HintFloat r2 = finput[4];
                HintFloat i2 = finput[5];
                HintFloat r3 = finput[6];
                HintFloat i3 = finput[7];
                HintFloat r4 = finput[8];
                HintFloat i4 = finput[9];
                HintFloat r5 = finput[10];
                HintFloat i5 = finput[11];
                HintFloat r6 = finput[12];
                HintFloat i6 = finput[13];
                HintFloat r7 = finput[14];
                HintFloat i7 = finput[15];

                HintFloat tmpr0 = r0;
                HintFloat tmpi0 = i0;
                r0 += r1;
                i0 += i1;
                r1 = tmpr0 - r1;
                i1 = tmpi0 - i1;

                HintFloat tmpr1 = r2;
                HintFloat tmpi1 = i2;
                r2 += r3;
                i2 += i3;
                r3 = tmpr1 - r3;
                i3 = tmpi1 - i3;

                HintFloat tmpr2 = r4;
                HintFloat tmpi2 = i4;
                r4 += r5;
                i4 += i5;
                r5 = tmpr2 - r5;
                i5 = tmpi2 - i5;

                HintFloat tmpr3 = r6;
                HintFloat tmpi3 = i6;
                r6 += r7;
                i6 += i7;
                r7 = tmpr3 - r7;
                i7 = tmpi3 - i7;

                // ary[3]*(-i),ary[7]*(-i)
                tmpr0 = r3;
                r3 = i3;
                i3 = -tmpr0;

                tmpr1 = r7;
                r7 = i7;
                i7 = -tmpr1;

                tmpr0 = r0;
                tmpi0 = i0;
                r0 += r2;
                i0 += i2;
                r2 = tmpr0 - r2;
                i2 = tmpi0 - i2;

                tmpr1 = r1;
                tmpi1 = i1;
                r1 += r3;
                i1 += i3;
                r3 = tmpr1 - r3;
                i3 = tmpi1 - i3;

                tmpr2 = r4;
                tmpi2 = i4;
                r4 += r6;
                i4 += i6;
                r6 = tmpr2 - r6;
                i6 = tmpi2 - i6;

                tmpr3 = r5;
                tmpi3 = i5;
                r5 += r7;
                i5 += i7;
                r7 = tmpr3 - r7;
                i7 = tmpi3 - i7;

                tmpr0 = r5 + i5;
                i5 -= r5;
                r5 = tmpr0 * cos_1_8;
                i5 *= cos_1_8;

                tmpr1 = r6;
                r6 = i6;
                i6 = -tmpr1;

                tmpi2 = r7 + i7;
                r7 -= i7;
                r7 *= -cos_1_8;
                i7 = -tmpi2 * cos_1_8;

                finput[0] = r0 + r4;
                finput[1] = i0 + i4;
                finput[2] = r1 + r5;
                finput[3] = i1 + i5;
                finput[4] = r2 + r6;
                finput[5] = i2 + i6;
                finput[6] = r3 + r7;
                finput[7] = i3 + i7;
                finput[8] = r0 - r4;
                finput[9] = i0 - i4;
                finput[10] = r1 - r5;
                finput[11] = i1 - i5;
                finput[12] = r2 - r6;
                finput[13] = i2 - i6;
                finput[14] = r3 - r7;
                finput[15] = i3 - i7;
            }
            inline void fft_dif_4point(Complex *input)
            {
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[2];
                HintFloat i1 = finput[3];
                HintFloat r2 = finput[4];
                HintFloat i2 = finput[5];
                HintFloat r3 = finput[6];
                HintFloat i3 = finput[7];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r2;
                i0 += i2;
                r2 = tmpr - r2;
                i2 = tmpi - i2;

                tmpr = r1;
                tmpi = i1;
                r1 += r3;
                i1 += i3;
                r3 = tmpr - r3;
                i3 = tmpi - i3;

                tmpr = r3;
                r3 = i3;
                i3 = -tmpr;

                finput[0] = r0 + r1;
                finput[1] = i0 + i1;
                finput[2] = r0 - r1;
                finput[3] = i0 - i1;
                finput[4] = r2 + r3;
                finput[5] = i2 + i3;
                finput[6] = r2 - r3;
                finput[7] = i2 - i3;
            }
            inline void fft_dif_8point(Complex *input)
            {
                static constexpr HintFloat cos_1_8 = 0.70710678118654752440084436210485;
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[2];
                HintFloat i1 = finput[3];
                HintFloat r2 = finput[4];
                HintFloat i2 = finput[5];
                HintFloat r3 = finput[6];
                HintFloat i3 = finput[7];
                HintFloat r4 = finput[8];
                HintFloat i4 = finput[9];
                HintFloat r5 = finput[10];
                HintFloat i5 = finput[11];
                HintFloat r6 = finput[12];
                HintFloat i6 = finput[13];
                HintFloat r7 = finput[14];
                HintFloat i7 = finput[15];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r4;
                i0 += i4;
                r4 = tmpr - r4;
                i4 = tmpi - i4;

                tmpr = r1;
                tmpi = i1;
                r1 += r5;
                i1 += i5;
                r5 = tmpr - r5;
                i5 = tmpi - i5;

                tmpr = r2;
                tmpi = i2;
                r2 += r6;
                i2 += i6;
                r6 = tmpr - r6;
                i6 = tmpi - i6;

                tmpr = r3;
                tmpi = i3;
                r3 += r7;
                i3 += i7;
                r7 = tmpr - r7;
                i7 = tmpi - i7;

                tmpr = r5 + i5;
                i5 -= r5;
                r5 = tmpr * cos_1_8;
                i5 *= cos_1_8;

                tmpr = r6;
                r6 = i6;
                i6 = -tmpr;

                tmpi = r7 + i7;
                r7 -= i7;
                r7 *= -cos_1_8;
                i7 = -tmpi * cos_1_8;

                tmpr = r0;
                tmpi = i0;
                r0 += r2;
                i0 += i2;
                r2 = tmpr - r2;
                i2 = tmpi - i2;

                tmpr = r1;
                tmpi = i1;
                r1 += r3;
                i1 += i3;
                r3 = tmpr - r3;
                i3 = tmpi - i3;

                tmpr = r4;
                tmpi = i4;
                r4 += r6;
                i4 += i6;
                r6 = tmpr - r6;
                i6 = tmpi - i6;

                tmpr = r5;
                tmpi = i5;
                r5 += r7;
                i5 += i7;
                r7 = tmpr - r7;
                i7 = tmpi - i7;

                // ary[3]*(-i),ary[7]*(-i)
                tmpr = r3;
                r3 = i3;
                i3 = -tmpr;
                tmpr = r7;
                r7 = i7;
                i7 = -tmpr;

                finput[0] = r0 + r1;
                finput[1] = i0 + i1;
                finput[2] = r0 - r1;
                finput[3] = i0 - i1;
                finput[4] = r2 + r3;
                finput[5] = i2 + i3;
                finput[6] = r2 - r3;
                finput[7] = i2 - i3;
                finput[8] = r4 + r5;
                finput[9] = i4 + i5;
                finput[10] = r4 - r5;
                finput[11] = i4 - i5;
                finput[12] = r6 + r7;
                finput[13] = i6 + i7;
                finput[14] = r6 - r7;
                finput[15] = i6 - i7;
            }

            // fft分裂基时间抽取蝶形变换
            inline void fft_split_radix_dit_butterfly(const Complex &omega, const Complex &omega_cube,
                                                      Complex *input, size_t rank)
            {
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[rank * 2];
                HintFloat i1 = finput[rank * 2 + 1];
                HintFloat r2 = finput[rank * 4];
                HintFloat i2 = finput[rank * 4 + 1];
                HintFloat r3 = finput[rank * 6];
                HintFloat i3 = finput[rank * 6 + 1];

                HintFloat tmpr0 = r2 * omega.real() - i2 * omega.imag();
                HintFloat tmpi0 = r2 * omega.imag() + i2 * omega.real();
                HintFloat tmpr1 = r3 * omega_cube.real() - i3 * omega_cube.imag();
                HintFloat tmpi1 = r3 * omega_cube.imag() + i3 * omega_cube.real();
                r2 = tmpr0 + tmpr1;
                i2 = tmpi0 + tmpi1;
                r3 = tmpi0 - tmpi1;
                i3 = tmpr1 - tmpr0;

                finput[0] = r0 + r2;
                finput[1] = i0 + i2;
                finput[rank * 2] = r1 + r3;
                finput[rank * 2 + 1] = i1 + i3;
                finput[rank * 4] = r0 - r2;
                finput[rank * 4 + 1] = i0 - i2;
                finput[rank * 6] = r1 - r3;
                finput[rank * 6 + 1] = i1 - i3;
            }

            // fft分裂基频率抽取蝶形变换
            inline void fft_split_radix_dif_butterfly(const Complex &omega, const Complex &omega_cube,
                                                      Complex *input, size_t rank)
            {
                HintFloat *finput = reinterpret_cast<HintFloat *>(input);
                HintFloat r0 = finput[0];
                HintFloat i0 = finput[1];
                HintFloat r1 = finput[rank * 2];
                HintFloat i1 = finput[rank * 2 + 1];
                HintFloat r2 = finput[rank * 4];
                HintFloat i2 = finput[rank * 4 + 1];
                HintFloat r3 = finput[rank * 6];
                HintFloat i3 = finput[rank * 6 + 1];

                HintFloat tmpr = r0;
                HintFloat tmpi = i0;
                r0 += r2;
                i0 += i2;
                r2 = tmpr - r2;
                i2 = tmpi - i2;

                tmpr = r1;
                tmpi = i1;
                r1 += r3;
                i1 += i3;
                tmpr = tmpr - r3;
                r3 = tmpi - i3;
                i3 = -tmpr;

                finput[0] = r0;
                finput[1] = i0;
                finput[rank * 2] = r1;
                finput[rank * 2 + 1] = i1;

                tmpr = r2;
                tmpi = i2;
                r2 += r3;
                i2 += i3;
                r3 = tmpr - r3;
                i3 = tmpi - i3;

                tmpr = r2 * omega.real() - i2 * omega.imag();
                tmpi = r2 * omega.imag() + i2 * omega.real();
                finput[rank * 4] = tmpr;
                finput[rank * 4 + 1] = tmpi;
                tmpr = r3 * omega_cube.real() - i3 * omega_cube.imag();
                tmpi = r3 * omega_cube.imag() + i3 * omega_cube.real();
                finput[rank * 6] = tmpr;
                finput[rank * 6 + 1] = tmpi;
            }
            // 模板化时间抽取分裂基fft
            void fft_split_radix_dit(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    if (fft_len == 8)
                    {
                        fft_dit_8point(input);
                    }
                    else if (fft_len == 4)
                    {
                        fft_dit_4point(input);
                    }
                    else if (fft_len == 2)
                    {
                        fft_2point(input[0], input[1]);
                    }
                    return;
                }
                const size_t log_len = hint_log2(fft_len);
                const size_t half_len = fft_len / 2, quarter_len = fft_len / 4;
                fft_split_radix_dit(input, half_len);
                fft_split_radix_dit(input + half_len, quarter_len);
                fft_split_radix_dit(input + half_len + quarter_len, quarter_len);
                for (size_t i = 0; i < quarter_len; i++)
                {
                    const Complex &omega = TABLE.get_omega(log_len, i);
                    const Complex &omega_cube = TABLE.get_omega3(log_len, i);
                    fft_split_radix_dit_butterfly(omega, omega_cube, input + i, quarter_len);
                }
            }
            // 模板化频率抽取分裂基fft
            void fft_split_radix_dif(Complex *input, size_t fft_len)
            {
                if (fft_len <= 8)
                {
                    if (fft_len == 8)
                    {
                        fft_dif_8point(input);
                    }
                    else if (fft_len == 4)
                    {
                        fft_dif_4point(input);
                    }
                    else if (fft_len == 2)
                    {
                        fft_2point(input[0], input[1]);
                    }
                    return;
                }
                const size_t log_len = hint_log2(fft_len);
                const size_t half_len = fft_len / 2, quarter_len = fft_len / 4;
                for (size_t i = 0; i < quarter_len; i++)
                {
                    const Complex &omega = TABLE.get_omega(log_len, i);
                    const Complex &omega_cube = TABLE.get_omega3(log_len, i);
                    fft_split_radix_dif_butterfly(omega, omega_cube, input + i, quarter_len);
                }
                fft_split_radix_dif(input, half_len);
                fft_split_radix_dif(input + half_len, quarter_len);
                fft_split_radix_dif(input + half_len + quarter_len, quarter_len);
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
                    const Complex &omega = TABLE.get_omega(log_len, i);
                    const Complex &omega_cube = TABLE.get_omega3(log_len, i);
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
                fft_dit_4point(input);
            }
            template <>
            void fft_split_radix_dit_template<8>(Complex *input)
            {
                fft_dit_8point(input);
            }

            // 模板化频率抽取分裂基fft
            template <size_t LEN>
            void fft_split_radix_dif_template(Complex *input)
            {
                constexpr size_t log_len = hint_log2(LEN);
                constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
                for (size_t i = 0; i < quarter_len; i++)
                {
                    const Complex &omega = TABLE.get_omega(log_len, i);
                    const Complex &omega_cube = TABLE.get_omega3(log_len, i);
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
                fft_dif_4point(input);
            }
            template <>
            void fft_split_radix_dif_template<8>(Complex *input)
            {
                fft_dif_8point(input);
            }

            // 辅助选择函数
            template <size_t LEN = 1>
            void fft_split_radix_dit_template_alt(Complex *input, size_t fft_len)
            {
                if (fft_len > LEN)
                {
                    fft_split_radix_dit_template_alt<LEN * 2>(input, fft_len);
                    return;
                }
                TABLE.expand(hint_log2(LEN));
                fft_split_radix_dit_template<LEN>(input);
            }
            template <>
            void fft_split_radix_dit_template_alt<1 << 24>(Complex *input, size_t fft_len) {}

            // 辅助选择函数
            template <size_t LEN = 1>
            void fft_split_radix_dif_template_alt(Complex *input, size_t fft_len)
            {
                if (fft_len > LEN)
                {
                    fft_split_radix_dif_template_alt<LEN * 2>(input, fft_len);
                    return;
                }
                TABLE.expand(hint_log2(LEN));
                fft_split_radix_dif_template<LEN>(input);
            }
            template <>
            void fft_split_radix_dif_template_alt<1 << 24>(Complex *input, size_t fft_len) {}

            auto fft_split_radix_dit_t = fft_split_radix_dit_template_alt<1>;
            auto fft_split_radix_dif_t = fft_split_radix_dif_template_alt<1>;
            // 求共轭复数及归一化，逆变换用
            inline void fft_conj(Complex *input, size_t fft_len, HintFloat div = 1)
            {
                for (size_t i = 0; i < fft_len; i++)
                {
                    input[i] = std::conj(input[i]) / div;
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

            /// @brief 时间抽取基2fft
            /// @param input 复数组
            /// @param fft_len 数组长度
            /// @param bit_rev 是否逆序
            inline void fft_dit(Complex *input, size_t fft_len, bool bit_rev = true)
            {
                fft_len = max_2pow(fft_len);
                TABLE.expand(hint_log2(fft_len));
                if (bit_rev)
                {
                    binary_reverse_swap(input, fft_len);
                }
                fft_split_radix_dit_t(input, fft_len);
            }
            /// @brief 频率抽取基2fft
            /// @param input 复数组
            /// @param fft_len 数组长度
            /// @param bit_rev 是否逆序
            inline void fft_dif(Complex *input, size_t fft_len, bool bit_rev = true)
            {
                fft_len = max_2pow(fft_len);
                TABLE.expand(hint_log2(fft_len));
                fft_split_radix_dif_t(input, fft_len);
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
                fft_dif(input, fft_len, true);
                fft_conj(input, fft_len, fft_len);
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
    size_t fft_len = min_2pow(out_len);
    Complex *fft_ary = new Complex[fft_len];
    com_ary_combine_copy(fft_ary, in1, len1, in2, len2);
    fft_dif(fft_ary, fft_len, false); // 优化FFT
    for (size_t i = 0; i < fft_len; i++)
    {
        Complex tmp = fft_ary[i];
        tmp *= tmp;
        fft_ary[i] = std::conj(tmp);
    }
    fft_dit(fft_ary, fft_len, false); // 优化FFT
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
#include "stopwatch.hpp"

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