#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>
#include "stopwatch.hpp"

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

    constexpr double PI = 3.1415926535897932384626433832795;
    constexpr double PI_2 = PI * 2;
    template <typename T>
    constexpr bool is_odd(T x)
    {
        return static_cast<bool>(x & 1);
    }
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
    // 模板快速幂
    template <typename T>
    constexpr T qpow(T m, UINT_64 n)
    {
        T result = 1;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                result = result * m;
            }
            m = m * m;
            n >>= 1;
        }
        return result;
    }
    // 取模快速幂
    constexpr UINT_64 qpow(UINT_64 m, UINT_64 n, UINT_64 mod)
    {
        if (m <= 1)
        {
            return m;
        }
        UINT_64 result = 1;
        while (n > 0)
        {
            if ((n & 1) != 0)
            {
                result = result * m % mod;
            }
            m = m * m % mod;
            n >>= 1;
        }
        return result;
    }
    // 模意义下的逆元
    constexpr UINT_64 mod_inv(UINT_64 n, UINT_64 mod)
    {
        return qpow(n, mod - 2, mod);
    }
    namespace hint_transform
    {
        // 二进制逆序
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
        // 四进制逆序
        template <typename SizeType = UINT_32, typename T>
        void quaternary_inverse_swap(T &ary, size_t len)
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
        template <typename T>
        inline void ary_mul(const T in1[], const T in2[], T out[], size_t len)
        {
            for (size_t i = 0; i < len; i++)
            {
                out[i] = in1[i] * in2[i];
            }
        }

        template <UINT_64 MOD, typename DataTy = UINT_32>
        struct ModInt
        {
            // 实际存放的整数
            DataTy data = 0;
            // 等价复数的i
            constexpr ModInt() noexcept {}
            constexpr ModInt(DataTy num) noexcept
            {
                data = num;
            }
            constexpr ModInt operator+(ModInt n) const
            {
                UINT_64 sum = static_cast<UINT_64>(data) + n.data;
                return sum < MOD ? sum : sum - MOD;
            }
            constexpr ModInt operator-(ModInt n) const
            {
                return data < n.data ? MOD + data - n.data : data - n.data;
            }
            constexpr ModInt operator*(ModInt n) const
            {
                return static_cast<UINT_64>(data) * n.data % MOD;
            }
            constexpr ModInt operator/(ModInt n) const
            {
                return *this * inv();
            }
            constexpr ModInt inv() const
            {
                return power(MOD - 2);
            }
            constexpr ModInt power(DataTy n) const
            {
                UINT_64 m = data;
                if (m <= 1)
                {
                    return m;
                }
                UINT_64 result = 1;
                while (n > 0)
                {
                    if ((n & 1) != 0)
                    {
                        result = result * m % MOD;
                    }
                    m = m * m % MOD;
                    n >>= 1;
                }
                return result;
            }
            constexpr ModInt normalize() const
            {
                return data % MOD;
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct NTT
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            template <typename T>
            using ModIntTy = ModInt<MOD, T>;
            template <typename T>
            static constexpr void ntt_normalize(ModIntTy<T> *input, size_t ntt_len)
            {
                const ModIntTy<T> inv = ModIntTy<T>(ntt_len).inv();
                size_t mod4 = ntt_len % 4;
                ntt_len -= mod4;
                for (size_t i = 0; i < ntt_len; i += 4)
                {
                    input[i] = inv * input[i];
                    input[i + 1] = inv * input[i + 1];
                    input[i + 2] = inv * input[i + 2];
                    input[i + 3] = inv * input[i + 3];
                }
                for (size_t i = ntt_len; i < ntt_len + mod4; i++)
                {
                    input[i] = inv * input[i];
                }
            }
            // 2点NTT
            template <typename T>
            static constexpr void ntt_2point(ModIntTy<T> &sum, ModIntTy<T> &diff)
            {
                ModIntTy<T> tmp1 = sum;
                ModIntTy<T> tmp2 = diff;
                sum = tmp1 + tmp2;
                diff = tmp1 - tmp2;
            }
            template <typename T>
            static constexpr void ntt_dit_4point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_4_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 4); // 等价于复数i
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2];
                ModIntTy<T> tmp3 = input[rank * 3];

                ntt_2point(tmp0, tmp1);
                ntt_2point(tmp2, tmp3);
                tmp3 = tmp3 * W_4_1;

                input[0] = tmp0 + tmp2;
                input[rank] = tmp1 + tmp3;
                input[rank * 2] = tmp0 - tmp2;
                input[rank * 3] = tmp1 - tmp3;
            }
            template <typename T>
            static constexpr void ntt_dit_8point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_8_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 8);
                constexpr ModIntTy<T> W_8_2 = W_8_1.power(2);
                constexpr ModIntTy<T> W_8_3 = W_8_1.power(3);
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2];
                ModIntTy<T> tmp3 = input[rank * 3];
                ModIntTy<T> tmp4 = input[rank * 4];
                ModIntTy<T> tmp5 = input[rank * 5];
                ModIntTy<T> tmp6 = input[rank * 6];
                ModIntTy<T> tmp7 = input[rank * 7];
                ntt_2point(tmp0, tmp1);
                ntt_2point(tmp2, tmp3);
                ntt_2point(tmp4, tmp5);
                ntt_2point(tmp6, tmp7);
                tmp3 = tmp3 * W_8_2;
                tmp7 = tmp7 * W_8_2;

                ntt_2point(tmp0, tmp2);
                ntt_2point(tmp1, tmp3);
                ntt_2point(tmp4, tmp6);
                ntt_2point(tmp5, tmp7);
                tmp5 = tmp5 * W_8_1;
                tmp6 = tmp6 * W_8_2;
                tmp7 = tmp7 * W_8_3;

                input[0] = tmp0 + tmp4;
                input[rank] = tmp1 + tmp5;
                input[rank * 2] = tmp2 + tmp6;
                input[rank * 3] = tmp3 + tmp7;
                input[rank * 4] = tmp0 - tmp4;
                input[rank * 5] = tmp1 - tmp5;
                input[rank * 6] = tmp2 - tmp6;
                input[rank * 7] = tmp3 - tmp7;
            }
            template <typename T>
            static constexpr void ntt_dit_16point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_16_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 16);
                constexpr ModIntTy<T> W_16_2 = W_16_1.power(2);
                constexpr ModIntTy<T> W_16_3 = W_16_1.power(3);
                constexpr ModIntTy<T> W_16_4 = W_16_1.power(4);
                constexpr ModIntTy<T> W_16_5 = W_16_1.power(5);
                constexpr ModIntTy<T> W_16_6 = W_16_1.power(6);
                constexpr ModIntTy<T> W_16_7 = W_16_1.power(7);

                ntt_dit_8point(input, rank);
                ntt_dit_8point(input + rank * 8, rank);

                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 8];
                ModIntTy<T> tmp3 = input[rank * 9] * W_16_1;
                input[0] = tmp0 + tmp2;
                input[rank] = tmp1 + tmp3;
                input[rank * 8] = tmp0 - tmp2;
                input[rank * 9] = tmp1 - tmp3;

                tmp0 = input[rank * 2];
                tmp1 = input[rank * 3];
                tmp2 = input[rank * 10] * W_16_2;
                tmp3 = input[rank * 11] * W_16_3;
                input[rank * 2] = tmp0 + tmp2;
                input[rank * 3] = tmp1 + tmp3;
                input[rank * 10] = tmp0 - tmp2;
                input[rank * 11] = tmp1 - tmp3;

                tmp0 = input[rank * 4];
                tmp1 = input[rank * 5];
                tmp2 = input[rank * 12] * W_16_4;
                tmp3 = input[rank * 13] * W_16_5;
                input[rank * 4] = tmp0 + tmp2;
                input[rank * 5] = tmp1 + tmp3;
                input[rank * 12] = tmp0 - tmp2;
                input[rank * 13] = tmp1 - tmp3;

                tmp0 = input[rank * 6];
                tmp1 = input[rank * 7];
                tmp2 = input[rank * 14] * W_16_6;
                tmp3 = input[rank * 15] * W_16_7;
                input[rank * 6] = tmp0 + tmp2;
                input[rank * 7] = tmp1 + tmp3;
                input[rank * 14] = tmp0 - tmp2;
                input[rank * 15] = tmp1 - tmp3;
            }
            template <typename T>
            static constexpr void ntt_dif_4point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_4_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 4); // 等价于复数i
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2];
                ModIntTy<T> tmp3 = input[rank * 3];

                ntt_2point(tmp0, tmp2);
                ntt_2point(tmp1, tmp3);
                tmp3 = tmp3 * W_4_1;

                input[0] = tmp0 + tmp1;
                input[rank] = tmp0 - tmp1;
                input[rank * 2] = tmp2 + tmp3;
                input[rank * 3] = tmp2 - tmp3;
            }
            template <typename T>
            static constexpr void ntt_dif_8point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_8_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 8);
                constexpr ModIntTy<T> W_8_2 = W_8_1.power(2);
                constexpr ModIntTy<T> W_8_3 = W_8_1.power(3);
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2];
                ModIntTy<T> tmp3 = input[rank * 3];
                ModIntTy<T> tmp4 = input[rank * 4];
                ModIntTy<T> tmp5 = input[rank * 5];
                ModIntTy<T> tmp6 = input[rank * 6];
                ModIntTy<T> tmp7 = input[rank * 7];

                ntt_2point(tmp0, tmp4);
                ntt_2point(tmp1, tmp5);
                ntt_2point(tmp2, tmp6);
                ntt_2point(tmp3, tmp7);
                tmp5 = tmp5 * W_8_1;
                tmp6 = tmp6 * W_8_2;
                tmp7 = tmp7 * W_8_3;

                ntt_2point(tmp0, tmp2);
                ntt_2point(tmp1, tmp3);
                ntt_2point(tmp4, tmp6);
                ntt_2point(tmp5, tmp7);
                tmp3 = tmp3 * W_8_2;
                tmp7 = tmp7 * W_8_2;

                input[0] = tmp0 + tmp1;
                input[rank] = tmp0 - tmp1;
                input[rank * 2] = tmp2 + tmp3;
                input[rank * 3] = tmp2 - tmp3;
                input[rank * 4] = tmp4 + tmp5;
                input[rank * 5] = tmp4 - tmp5;
                input[rank * 6] = tmp6 + tmp7;
                input[rank * 7] = tmp6 - tmp7;
            }
            template <typename T>
            static constexpr void ntt_dif_16point(ModIntTy<T> *input, size_t rank = 1)
            {
                constexpr ModIntTy<T> W_16_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 16);
                constexpr ModIntTy<T> W_16_2 = W_16_1.power(2);
                constexpr ModIntTy<T> W_16_3 = W_16_1.power(3);
                constexpr ModIntTy<T> W_16_4 = W_16_1.power(4);
                constexpr ModIntTy<T> W_16_5 = W_16_1.power(5);
                constexpr ModIntTy<T> W_16_6 = W_16_1.power(6);
                constexpr ModIntTy<T> W_16_7 = W_16_1.power(7);

                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 8];
                ModIntTy<T> tmp3 = input[rank * 9];
                input[0] = tmp0 + tmp2;
                input[rank] = tmp1 + tmp3;
                input[rank * 8] = tmp0 - tmp2;
                input[rank * 9] = (tmp1 - tmp3) * W_16_1;

                tmp0 = input[rank * 2];
                tmp1 = input[rank * 3];
                tmp2 = input[rank * 10];
                tmp3 = input[rank * 11];
                input[rank * 2] = tmp0 + tmp2;
                input[rank * 3] = tmp1 + tmp3;
                input[rank * 10] = (tmp0 - tmp2) * W_16_2;
                input[rank * 11] = (tmp1 - tmp3) * W_16_3;

                tmp0 = input[rank * 4];
                tmp1 = input[rank * 5];
                tmp2 = input[rank * 12];
                tmp3 = input[rank * 13];
                input[rank * 4] = tmp0 + tmp2;
                input[rank * 5] = tmp1 + tmp3;
                input[rank * 12] = (tmp0 - tmp2) * W_16_4;
                input[rank * 13] = (tmp1 - tmp3) * W_16_5;

                tmp0 = input[rank * 6];
                tmp1 = input[rank * 7];
                tmp2 = input[rank * 14];
                tmp3 = input[rank * 15];
                input[rank * 6] = tmp0 + tmp2;
                input[rank * 7] = tmp1 + tmp3;
                input[rank * 14] = (tmp0 - tmp2) * W_16_6;
                input[rank * 15] = (tmp1 - tmp3) * W_16_7;

                ntt_dif_8point(input, rank);
                ntt_dif_8point(input + rank * 8, rank);
            }
            // 基2时间抽取ntt蝶形
            template <typename T>
            static constexpr void ntt_radix2_dit_butterfly(ModIntTy<T> omega, ModIntTy<T> *input, size_t rank)
            {
                ModIntTy<T> tmp1 = input[0];
                ModIntTy<T> tmp2 = input[rank] * omega;
                input[0] = tmp1 + tmp2;
                input[rank] = tmp1 - tmp2;
            }
            // 基2频率抽取ntt蝶形
            template <typename T>
            static constexpr void ntt_radix2_dif_butterfly(ModIntTy<T> omega, ModIntTy<T> *input, size_t rank)
            {
                ModIntTy<T> tmp1 = input[0];
                ModIntTy<T> tmp2 = input[rank];
                input[0] = tmp1 + tmp2;
                input[rank] = (tmp1 - tmp2) * omega;
            }
            // ntt分裂基时间抽取蝶形变换
            template <typename T>
            static constexpr void ntt_split_radix_dit_butterfly(ModIntTy<T> omega, ModIntTy<T> omega_cube,
                                                                ModIntTy<T> *input, size_t rank)
            {
                constexpr ModIntTy<T> W_4_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 4); // 等价于复数i
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2] * omega;
                ModIntTy<T> tmp3 = input[rank * 3] * omega_cube;

                ntt_2point(tmp2, tmp3);
                tmp3 = tmp3 * W_4_1;

                input[0] = tmp0 + tmp2;
                input[rank] = tmp1 + tmp3;
                input[rank * 2] = tmp0 - tmp2;
                input[rank * 3] = tmp1 - tmp3;
            }
            // ntt分裂基频率抽取蝶形变换
            template <typename T>
            static constexpr void ntt_split_radix_dif_butterfly(ModIntTy<T> omega, ModIntTy<T> omega_cube,
                                                                ModIntTy<T> *input, size_t rank)
            {
                constexpr ModIntTy<T> W_4_1 = ModIntTy<T>(G_ROOT).power((MOD - 1) / 4); // 等价于复数i
                ModIntTy<T> tmp0 = input[0];
                ModIntTy<T> tmp1 = input[rank];
                ModIntTy<T> tmp2 = input[rank * 2];
                ModIntTy<T> tmp3 = input[rank * 3];

                ntt_2point(tmp0, tmp2);
                ntt_2point(tmp1, tmp3);
                tmp3 = tmp3 * W_4_1;

                input[0] = tmp0;
                input[rank] = tmp1;
                input[rank * 2] = (tmp2 + tmp3) * omega;
                input[rank * 3] = (tmp2 - tmp3) * omega_cube;
            }
            // 基2时间抽取ntt
            template <typename T>
            static void ntt_radix2_dit(ModIntTy<T> *input, size_t ntt_len, bool bit_inv = true)
            {
                ntt_len = max_2pow(ntt_len);
                if (ntt_len <= 1)
                {
                    return;
                }
                if (ntt_len == 2)
                {
                    ntt_2point(input[0], input[1]);
                    return;
                }
                if (bit_inv)
                {
                    binary_inverse_swap(input, ntt_len);
                }
                for (size_t pos = 0; pos < ntt_len; pos += 2)
                {
                    ntt_2point(input[pos], input[pos + 1]);
                }
                for (size_t rank = 2; rank < ntt_len / 2; rank *= 2)
                {
                    size_t gap = rank * 2;
                    ModIntTy<T> unit_omega = ModIntTy<T>(G_ROOT).power((MOD - 1) / gap);
                    for (size_t begin = 0; begin < ntt_len; begin += (gap * 2))
                    {
                        ntt_2point(input[begin], input[begin + rank]);
                        ntt_2point(input[begin + gap], input[begin + rank + gap]);
                        ModIntTy<T> omega = unit_omega;
                        for (size_t pos = begin + 1; pos < begin + rank; pos++)
                        {
                            ntt_radix2_dit_butterfly(omega, input + pos, rank);
                            ntt_radix2_dit_butterfly(omega, input + pos + gap, rank);
                            omega = omega * unit_omega;
                        }
                    }
                }
                ModIntTy<T> omega = 1, unit_omega = ModIntTy<T>(G_ROOT).power((MOD - 1) / ntt_len);
                ntt_len /= 2;
                for (size_t pos = 0; pos < ntt_len; pos++)
                {
                    ntt_radix2_dit_butterfly(omega, input + pos, ntt_len);
                    omega = omega * unit_omega;
                }
            }
            // 基2频率抽取ntt
            template <typename T>
            static void ntt_radix2_dif(ModIntTy<T> *input, size_t ntt_len, bool bit_inv = true)
            {
                ntt_len = max_2pow(ntt_len);
                if (ntt_len <= 1)
                {
                    return;
                }
                if (ntt_len == 2)
                {
                    ntt_2point(input[0], input[1]);
                    return;
                }
                ModIntTy<T> unit_omega = ModIntTy<T>(G_ROOT).power((MOD - 1) / ntt_len);
                ModIntTy<T> omega = 1;
                for (size_t pos = 0; pos < ntt_len / 2; pos++)
                {
                    ntt_radix2_dif_butterfly(omega, input + pos, ntt_len / 2);
                    omega = omega * unit_omega;
                }
                unit_omega = unit_omega * unit_omega;
                for (size_t rank = ntt_len / 4; rank > 1; rank /= 2)
                {
                    size_t gap = rank * 2;
                    for (size_t begin = 0; begin < ntt_len; begin += (gap * 2))
                    {
                        ntt_2point(input[begin], input[begin + rank]);
                        ntt_2point(input[begin + gap], input[begin + rank + gap]);
                        ModIntTy<T> omega = unit_omega;
                        for (size_t pos = begin + 1; pos < begin + rank; pos++)
                        {
                            ntt_radix2_dif_butterfly(omega, input + pos, rank);
                            ntt_radix2_dif_butterfly(omega, input + pos + gap, rank);
                            omega = omega * unit_omega;
                        }
                    }
                    unit_omega = unit_omega * unit_omega;
                }
                for (size_t pos = 0; pos < ntt_len; pos += 2)
                {
                    ntt_2point(input[pos], input[pos + 1]);
                }
                if (bit_inv)
                {
                    binary_inverse_swap(input, ntt_len);
                }
            }
        };
        // template <UINT_64 MOD, UINT_64 G_ROOT>
        template <size_t LEN, UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
            // 模板化时间抽取分裂基ntt
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input)
            {
                SPLIT_RADIX_NTT<half_len, MOD, G_ROOT>::ntt_split_radix_dit_template(input);
                SPLIT_RADIX_NTT<quarter_len, MOD, G_ROOT>::ntt_split_radix_dit_template(input + half_len);
                SPLIT_RADIX_NTT<quarter_len, MOD, G_ROOT>::ntt_split_radix_dit_template(input + half_len + quarter_len);
                constexpr ModInt32 unit = ModInt32(G_ROOT).power((MOD - 1) / LEN);
                constexpr ModInt32 unit_cube = unit.power(3);
                ModInt32 omega(1), omega_cube(1);
                for (size_t i = 0; i < quarter_len; i++)
                {
                    NTT<MOD, G_ROOT>::ntt_split_radix_dit_butterfly(omega, omega_cube, input + i, quarter_len);
                    omega = omega * unit;
                    omega_cube = omega_cube * unit_cube;
                }
            }
            // 模板化频率抽取分裂基ntt
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input)
            {
                constexpr ModInt32 unit = ModInt32(G_ROOT).power((MOD - 1) / LEN);
                constexpr ModInt32 unit_cube = unit.power(3);
                ModInt32 omega(1), omega_cube(1);
                for (size_t i = 0; i < quarter_len; i++)
                {
                    NTT<MOD, G_ROOT>::ntt_split_radix_dif_butterfly(omega, omega_cube, input + i, quarter_len);
                    omega = omega * unit;
                    omega_cube = omega_cube * unit_cube;
                }
                SPLIT_RADIX_NTT<half_len, MOD, G_ROOT>::ntt_split_radix_dif_template(input);
                SPLIT_RADIX_NTT<quarter_len, MOD, G_ROOT>::ntt_split_radix_dif_template(input + half_len);
                SPLIT_RADIX_NTT<quarter_len, MOD, G_ROOT>::ntt_split_radix_dif_template(input + half_len + quarter_len);
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<0, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input) {}
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input) {}
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<1, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input) {}
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input) {}
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<2, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_2point(input[0], input[1]);
            }
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_2point(input[0], input[1]);
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<4, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dit_4point(input, 1);
            }
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dif_4point(input, 1);
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<8, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dit_8point(input, 1);
            }
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dif_8point(input, 1);
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct SPLIT_RADIX_NTT<16, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_split_radix_dit_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dit_16point(input, 1);
            }
            static constexpr void ntt_split_radix_dif_template(ModInt32 *input)
            {
                NTT<MOD, G_ROOT>::ntt_dif_16point(input, 1);
            }
        };
        template <size_t LEN, UINT_64 MOD, UINT_64 G_ROOT>
        struct NTT_ALERT
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_dit_template(ModInt32 *input, size_t ntt_len)
            {
                if (ntt_len > LEN)
                {
                    NTT_ALERT<LEN * 2, MOD, G_ROOT>::ntt_dit_template(input, ntt_len);
                    return;
                }
                SPLIT_RADIX_NTT<LEN, MOD, G_ROOT>::ntt_split_radix_dit_template(input);
            }
            static constexpr void ntt_dif_template(ModInt32 *input, size_t ntt_len)
            {
                if (ntt_len > LEN)
                {
                    NTT_ALERT<LEN * 2, MOD, G_ROOT>::ntt_dif_template(input, ntt_len);
                    return;
                }
                SPLIT_RADIX_NTT<LEN, MOD, G_ROOT>::ntt_split_radix_dif_template(input);
            }
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        struct NTT_ALERT<1 << 30, MOD, G_ROOT>
        {
            using ModInt32 = ModInt<MOD, UINT_32>;
            static constexpr void ntt_dit_template(ModInt32 *input, size_t ntt_len) {}
            static constexpr void ntt_dif_template(ModInt32 *input, size_t ntt_len) {}
        };
        template <UINT_64 MOD, UINT_64 G_ROOT>
        inline void ntt_dit(UINT_32 *input, size_t ntt_len, bool bit_inv = true)
        {
            ntt_len = max_2pow(ntt_len);
            if (bit_inv)
            {
                binary_inverse_swap(input, ntt_len);
            }
            NTT_ALERT<1, MOD, G_ROOT>::ntt_dit_template(reinterpret_cast<ModInt<MOD, UINT_32> *>(input), ntt_len);
        }
        template <UINT_64 MOD, UINT_64 G_ROOT>
        inline void ntt_dif(UINT_32 *input, size_t ntt_len, bool bit_inv = true)
        {
            ntt_len = max_2pow(ntt_len);
            NTT_ALERT<1, MOD, G_ROOT>::ntt_dif_template(reinterpret_cast<ModInt<MOD, UINT_32> *>(input), ntt_len);
            if (bit_inv)
            {
                binary_inverse_swap(input, ntt_len);
            }
        }
    }
}

using namespace std;
using namespace hint;
using namespace hint_transform;
template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    constexpr UINT_64 NTT_MOD1 = 3221225473;
    constexpr UINT_64 NTT_ROOT1 = 5, I_GROOT1 = mod_inv(NTT_ROOT1, NTT_MOD1);
    constexpr UINT_64 NTT_MOD2 = 3489660929;
    constexpr UINT_64 NTT_ROOT2 = 3, I_GROOT2 = mod_inv(NTT_ROOT2, NTT_MOD2);
    vector<T> result(out_len);
    size_t ntt_len = min_2pow(out_len);
    uint32_t *ary1 = new uint32_t[ntt_len]{};
    uint32_t *ary2 = new uint32_t[ntt_len]{};
    using ntt1 = NTT<NTT_MOD1, NTT_ROOT1>;
    using intt1 = NTT<NTT_MOD1, I_GROOT1>;
    using ntt2 = NTT<NTT_MOD2, NTT_ROOT2>;
    using intt2 = NTT<NTT_MOD2, I_GROOT2>;

    auto mod_ary1 = reinterpret_cast<ntt2::ModInt32 *>(ary1);
    auto mod_ary2 = reinterpret_cast<ntt2::ModInt32 *>(ary2);

    for (size_t i = 0; i < len1; i++)
    {
        ary1[i] = in1[i];
    }
    for (size_t i = 0; i < len2; i++)
    {
        ary2[i] = in2[i];
    }
    ntt_dif<NTT_MOD2, NTT_ROOT2>(ary1, ntt_len, false);
    ntt_dif<NTT_MOD2, NTT_ROOT2>(ary2, ntt_len, false);
    ary_mul(mod_ary1, mod_ary2, mod_ary1, ntt_len);
    ntt_dit<NTT_MOD2, I_GROOT2>(ary1, ntt_len, false);
    // intt2::ntt_radix2_dit(mod_ary1, 64, false);
    intt2::ntt_normalize(mod_ary1, ntt_len);
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = ary1[i];
    }
    return result;
}
int main()
{
    StopWatch w(1000);
    int n = 18;
    cin >> n;
    size_t len = 1 << n;
    uint64_t ele = 5;
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