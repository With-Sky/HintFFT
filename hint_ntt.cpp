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

    constexpr UINT_64 NTT_MOD1 = 3221225473;
    constexpr UINT_64 NTT_ROOT1 = 5;
    constexpr UINT_64 NTT_MOD2 = 3489660929;
    constexpr UINT_64 NTT_ROOT2 = 3;
    constexpr size_t NTT_MAX_LEN1 = 1ull << 28;

    constexpr UINT_64 NTT_MOD3 = 79164837199873;
    constexpr UINT_64 NTT_ROOT3 = 5;
    constexpr UINT_64 NTT_MOD4 = 96757023244289;
    constexpr UINT_64 NTT_ROOT4 = 3;
    constexpr size_t NTT_MAX_LEN2 = 1ull << 43;

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
    template <typename T>
    inline void ary_mul(const T in1[], const T in2[], T out[], size_t len)
    {
        for (size_t i = 0; i < len; i++)
        {
            out[i] = in1[i] * in2[i];
        }
    }
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
        namespace hint_ntt
        {
            template <typename DataTy, UINT_64 MOD>
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
                    UINT_64 sum = UINT_64(data) + n.data;
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
                    return qpow(*this, MOD - 2u);
                }
                constexpr static ModInt mod()
                {
                    return MOD;
                }
            };
            template <UINT_64 MOD>
            struct ModInt<UINT_64, MOD>
            {
                // 实际存放的整数
                UINT_64 data = 0;
                // 等价复数的i
                constexpr ModInt() noexcept {}
                constexpr ModInt(UINT_64 num) noexcept
                {
                    data = num;
                }
                constexpr ModInt operator+(ModInt n) const
                {
                    return (n.data += data) < MOD ? n.data : n.data - MOD;
                }
                constexpr ModInt operator-(ModInt n) const
                {
                    return data < n.data ? MOD + data - n.data : data - n.data;
                }
                constexpr ModInt operator*(ModInt n) const
                {
                    UINT_64 b1 = n.data & 0xffff;
                    UINT_64 b2 = (n.data >> 16) & 0xffff;
                    UINT_64 b3 = n.data >> 32;
                    b1 = data * b1;
                    b2 = data * b2;
                    b3 = (((data * b3) % MOD) << 16) + b2;
                    b3 = ((b3 % MOD) << 16) + b1;
                    return b3 % MOD;
                    // UINT_64 b2 = n.data >> 20;
                    // n.data &= 0xfffff;
                    // n.data *= data;
                    // b2 = data * b2 % MOD;
                    // return (n.data + (b2 << 20)) % MOD;
                }
                constexpr ModInt operator/(ModInt n) const
                {
                    return *this * inv();
                }
                constexpr ModInt inv() const
                {
                    return qpow(*this, MOD - 2ull);
                }
                constexpr static ModInt mod()
                {
                    return MOD;
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct NTT_BASIC
            {
                static constexpr UINT_64 mod()
                {
                    return MOD;
                }
                static constexpr UINT_64 root()
                {
                    return ROOT;
                }

                using NTTModInt32 = ModInt<UINT_32, MOD>;
                using NTTModInt64 = ModInt<UINT_64, MOD>;

                template <typename T>
                static constexpr void ntt_normalize(T *input, size_t ntt_len)
                {
                    const T inv = T(ntt_len).inv();
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
                static constexpr void ntt_2point(T &sum, T &diff)
                {
                    T tmp1 = sum;
                    T tmp2 = diff;
                    sum = tmp1 + tmp2;
                    diff = tmp1 - tmp2;
                }
                template <typename T>
                static constexpr void ntt_dit_4point(T *input, size_t rank = 1)
                {
                    constexpr T W_4_1 = qpow(T(ROOT), (MOD - 1) / 4); // 等价于复数i
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2];
                    T tmp3 = input[rank * 3];

                    ntt_2point(tmp0, tmp1);
                    ntt_2point(tmp2, tmp3);
                    tmp3 = tmp3 * W_4_1;

                    input[0] = tmp0 + tmp2;
                    input[rank] = tmp1 + tmp3;
                    input[rank * 2] = tmp0 - tmp2;
                    input[rank * 3] = tmp1 - tmp3;
                }
                template <typename T>
                static constexpr void ntt_dit_8point(T *input, size_t rank = 1)
                {
                    constexpr T W_8_1 = qpow(T(ROOT), (MOD - 1) / 8);
                    constexpr T W_8_2 = qpow(W_8_1, 2);
                    constexpr T W_8_3 = qpow(W_8_1, 3);
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2];
                    T tmp3 = input[rank * 3];
                    T tmp4 = input[rank * 4];
                    T tmp5 = input[rank * 5];
                    T tmp6 = input[rank * 6];
                    T tmp7 = input[rank * 7];
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
                static constexpr void ntt_dit_16point(T *input, size_t rank = 1)
                {
                    constexpr T W_16_1 = qpow(T(ROOT), (MOD - 1) / 16);
                    constexpr T W_16_2 = qpow(W_16_1, 2);
                    constexpr T W_16_3 = qpow(W_16_1, 3);
                    constexpr T W_16_4 = qpow(W_16_1, 4);
                    constexpr T W_16_5 = qpow(W_16_1, 5);
                    constexpr T W_16_6 = qpow(W_16_1, 6);
                    constexpr T W_16_7 = qpow(W_16_1, 7);

                    ntt_dit_8point(input, rank);
                    ntt_dit_8point(input + rank * 8, rank);

                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 8];
                    T tmp3 = input[rank * 9] * W_16_1;
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
                static constexpr void ntt_dif_4point(T *input, size_t rank = 1)
                {
                    constexpr T W_4_1 = qpow(T(ROOT), (MOD - 1) / 4);
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2];
                    T tmp3 = input[rank * 3];

                    ntt_2point(tmp0, tmp2);
                    ntt_2point(tmp1, tmp3);
                    tmp3 = tmp3 * W_4_1;

                    input[0] = tmp0 + tmp1;
                    input[rank] = tmp0 - tmp1;
                    input[rank * 2] = tmp2 + tmp3;
                    input[rank * 3] = tmp2 - tmp3;
                }
                template <typename T>
                static constexpr void ntt_dif_8point(T *input, size_t rank = 1)
                {
                    constexpr T W_8_1 = qpow(T(ROOT), (MOD - 1) / 8);
                    constexpr T W_8_2 = qpow(W_8_1, 2);
                    constexpr T W_8_3 = qpow(W_8_1, 3);
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2];
                    T tmp3 = input[rank * 3];
                    T tmp4 = input[rank * 4];
                    T tmp5 = input[rank * 5];
                    T tmp6 = input[rank * 6];
                    T tmp7 = input[rank * 7];

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
                static constexpr void ntt_dif_16point(T *input, size_t rank = 1)
                {
                    constexpr T W_16_1 = qpow(T(ROOT), (MOD - 1) / 16);
                    constexpr T W_16_2 = qpow(W_16_1, 2);
                    constexpr T W_16_3 = qpow(W_16_1, 3);
                    constexpr T W_16_4 = qpow(W_16_1, 4);
                    constexpr T W_16_5 = qpow(W_16_1, 5);
                    constexpr T W_16_6 = qpow(W_16_1, 6);
                    constexpr T W_16_7 = qpow(W_16_1, 7);

                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 8];
                    T tmp3 = input[rank * 9];
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
                static constexpr void ntt_radix2_dit_butterfly(T omega, T *input, size_t rank)
                {
                    T tmp1 = input[0];
                    T tmp2 = input[rank] * omega;
                    input[0] = tmp1 + tmp2;
                    input[rank] = tmp1 - tmp2;
                }
                // 基2频率抽取ntt蝶形
                template <typename T>
                static constexpr void ntt_radix2_dif_butterfly(T omega, T *input, size_t rank)
                {
                    T tmp1 = input[0];
                    T tmp2 = input[rank];
                    input[0] = tmp1 + tmp2;
                    input[rank] = (tmp1 - tmp2) * omega;
                }
                // ntt分裂基时间抽取蝶形变换
                template <typename T>
                static constexpr void ntt_split_radix_dit_butterfly(T omega, T omega_cube,
                                                                    T *input, size_t rank)
                {
                    constexpr T W_4_1 = qpow(T(ROOT), (MOD - 1) / 4); // 等价于复数i
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2] * omega;
                    T tmp3 = input[rank * 3] * omega_cube;

                    ntt_2point(tmp2, tmp3);
                    tmp3 = tmp3 * W_4_1;

                    input[0] = tmp0 + tmp2;
                    input[rank] = tmp1 + tmp3;
                    input[rank * 2] = tmp0 - tmp2;
                    input[rank * 3] = tmp1 - tmp3;
                }
                // ntt分裂基频率抽取蝶形变换
                template <typename T>
                static constexpr void ntt_split_radix_dif_butterfly(T omega, T omega_cube,
                                                                    T *input, size_t rank)
                {
                    constexpr T W_4_1 = qpow(T(ROOT), (MOD - 1) / 4); // 等价于复数i
                    T tmp0 = input[0];
                    T tmp1 = input[rank];
                    T tmp2 = input[rank * 2];
                    T tmp3 = input[rank * 3];

                    ntt_2point(tmp0, tmp2);
                    ntt_2point(tmp1, tmp3);
                    tmp3 = tmp3 * W_4_1;

                    input[0] = tmp0;
                    input[rank] = tmp1;
                    input[rank * 2] = (tmp2 + tmp3) * omega;
                    input[rank * 3] = (tmp2 - tmp3) * omega_cube;
                }
            };
            // 模板递归分裂基NTT
            template <size_t LEN, UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                static constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
                // 模板化时间抽取分裂基ntt
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[])
                {
                    SPLIT_RADIX_NTT<half_len, MOD, ROOT>::ntt_split_radix_dit_template(input);
                    SPLIT_RADIX_NTT<quarter_len, MOD, ROOT>::ntt_split_radix_dit_template(input + half_len);
                    SPLIT_RADIX_NTT<quarter_len, MOD, ROOT>::ntt_split_radix_dit_template(input + half_len + quarter_len);
                    constexpr T unit = qpow(T(ROOT), (MOD - 1) / LEN);
                    constexpr T unit_cube = qpow(unit, 3);
                    T omega(1), omega_cube(1);
                    for (size_t i = 0; i < quarter_len; i++)
                    {
                        ntt_basic::ntt_split_radix_dit_butterfly(omega, omega_cube, input + i, quarter_len);
                        omega = omega * unit;
                        omega_cube = omega_cube * unit_cube;
                    }
                }
                // 模板化频率抽取分裂基ntt
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[])
                {
                    constexpr T unit = qpow(T(ROOT), (MOD - 1) / LEN);
                    constexpr T unit_cube = qpow(unit, 3);
                    T omega(1), omega_cube(1);
                    for (size_t i = 0; i < quarter_len; i++)
                    {
                        ntt_basic::ntt_split_radix_dif_butterfly(omega, omega_cube, input + i, quarter_len);
                        omega = omega * unit;
                        omega_cube = omega_cube * unit_cube;
                    }
                    SPLIT_RADIX_NTT<half_len, MOD, ROOT>::ntt_split_radix_dif_template(input);
                    SPLIT_RADIX_NTT<quarter_len, MOD, ROOT>::ntt_split_radix_dif_template(input + half_len);
                    SPLIT_RADIX_NTT<quarter_len, MOD, ROOT>::ntt_split_radix_dif_template(input + half_len + quarter_len);
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<0, MOD, ROOT>
            {
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[]) {}
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[]) {}
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<1, MOD, ROOT>
            {
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[]) {}
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[]) {}
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<2, MOD, ROOT>
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[])
                {
                    ntt_basic::ntt_2point(input[0], input[1]);
                }
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[])
                {
                    ntt_basic::ntt_2point(input[0], input[1]);
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<4, MOD, ROOT>
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[])
                {
                    ntt_basic::ntt_dit_4point(input, 1);
                }
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[])
                {
                    ntt_basic::ntt_dif_4point(input, 1);
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<8, MOD, ROOT>
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[])
                {
                    ntt_basic::ntt_dit_8point(input, 1);
                }
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[])
                {
                    ntt_basic::ntt_dif_8point(input, 1);
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct SPLIT_RADIX_NTT<16, MOD, ROOT>
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                template <typename T>
                static constexpr void ntt_split_radix_dit_template(T input[])
                {
                    ntt_basic::ntt_dit_16point(input, 1);
                }
                template <typename T>
                static constexpr void ntt_split_radix_dif_template(T input[])
                {
                    ntt_basic::ntt_dif_16point(input, 1);
                }
            };
            // 分裂基辅助选择类
            template <size_t LEN, UINT_64 MOD, UINT_64 ROOT>
            struct NTT_ALT
            {
                template <typename T>
                static constexpr void ntt_dit_template(T input[], size_t ntt_len)
                {
                    if (ntt_len > LEN)
                    {
                        NTT_ALT<LEN * 2, MOD, ROOT>::ntt_dit_template(input, ntt_len);
                        return;
                    }
                    SPLIT_RADIX_NTT<LEN, MOD, ROOT>::ntt_split_radix_dit_template(input);
                }
                template <typename T>
                static constexpr void ntt_dif_template(T input[], size_t ntt_len)
                {
                    if (ntt_len > LEN)
                    {
                        NTT_ALT<LEN * 2, MOD, ROOT>::ntt_dif_template(input, ntt_len);
                        return;
                    }
                    SPLIT_RADIX_NTT<LEN, MOD, ROOT>::ntt_split_radix_dif_template(input);
                }
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct NTT_ALT<size_t(1) << 43, MOD, ROOT>
            {
                template <typename T>
                static constexpr void ntt_dit_template(T input[], size_t ntt_len) {}
                template <typename T>
                static constexpr void ntt_dif_template(T input[], size_t ntt_len) {}
            };
            template <UINT_64 MOD, UINT_64 ROOT>
            struct NTT
            {
                using ntt_basic = NTT_BASIC<MOD, ROOT>;
                using NTTModInt32 = ModInt<UINT_32, MOD>;
                using NTTModInt64 = ModInt<UINT_64, MOD>;
                static constexpr UINT_64 mod()
                {
                    return MOD;
                }
                static constexpr UINT_64 root()
                {
                    return ROOT;
                }
                static constexpr UINT_64 iroot()
                {
                    return NTTModInt64(ROOT).inv().data;
                }
                // 基2时间抽取ntt,学习用
                template <typename T>
                static void ntt_radix2_dit(T *input, size_t ntt_len, bool bit_rev = true)
                {
                    ntt_len = max_2pow(ntt_len);
                    if (ntt_len <= 1)
                    {
                        return;
                    }
                    if (ntt_len == 2)
                    {
                        ntt_basic::ntt_2point(input[0], input[1]);
                        return;
                    }
                    if (bit_rev)
                    {
                        binary_reverse_swap(input, ntt_len);
                    }
                    for (size_t pos = 0; pos < ntt_len; pos += 2)
                    {
                        ntt_basic::ntt_2point(input[pos], input[pos + 1]);
                    }
                    for (size_t rank = 2; rank < ntt_len / 2; rank *= 2)
                    {
                        size_t gap = rank * 2;
                        T unit_omega = qpow(T(ROOT), (MOD - 1) / gap);
                        for (size_t begin = 0; begin < ntt_len; begin += (gap * 2))
                        {
                            ntt_basic::ntt_2point(input[begin], input[begin + rank]);
                            ntt_basic::ntt_2point(input[begin + gap], input[begin + rank + gap]);
                            T omega = unit_omega;
                            for (size_t pos = begin + 1; pos < begin + rank; pos++)
                            {
                                ntt_basic::ntt_radix2_dit_butterfly(omega, input + pos, rank);
                                ntt_basic::ntt_radix2_dit_butterfly(omega, input + pos + gap, rank);
                                omega = omega * unit_omega;
                            }
                        }
                    }
                    T omega = 1, unit_omega = qpow(T(ROOT), (MOD - 1) / ntt_len);
                    ntt_len /= 2;
                    for (size_t pos = 0; pos < ntt_len; pos++)
                    {
                        ntt_basic::ntt_radix2_dit_butterfly(omega, input + pos, ntt_len);
                        omega = omega * unit_omega;
                    }
                }
                // 基2频率抽取ntt,学习用
                template <typename T>
                static void ntt_radix2_dif(T *input, size_t ntt_len, bool bit_rev = true)
                {
                    ntt_len = max_2pow(ntt_len);
                    if (ntt_len <= 1)
                    {
                        return;
                    }
                    if (ntt_len == 2)
                    {
                        ntt_basic::ntt_2point(input[0], input[1]);
                        return;
                    }
                    T unit_omega = qpow(T(ROOT), (MOD - 1) / ntt_len);
                    T omega = 1;
                    for (size_t pos = 0; pos < ntt_len / 2; pos++)
                    {
                        ntt_basic::ntt_radix2_dif_butterfly(omega, input + pos, ntt_len / 2);
                        omega = omega * unit_omega;
                    }
                    unit_omega = unit_omega * unit_omega;
                    for (size_t rank = ntt_len / 4; rank > 1; rank /= 2)
                    {
                        size_t gap = rank * 2;
                        for (size_t begin = 0; begin < ntt_len; begin += (gap * 2))
                        {
                            ntt_basic::ntt_2point(input[begin], input[begin + rank]);
                            ntt_basic::ntt_2point(input[begin + gap], input[begin + rank + gap]);
                            T omega = unit_omega;
                            for (size_t pos = begin + 1; pos < begin + rank; pos++)
                            {
                                ntt_basic::ntt_radix2_dif_butterfly(omega, input + pos, rank);
                                ntt_basic::ntt_radix2_dif_butterfly(omega, input + pos + gap, rank);
                                omega = omega * unit_omega;
                            }
                        }
                        unit_omega = unit_omega * unit_omega;
                    }
                    for (size_t pos = 0; pos < ntt_len; pos += 2)
                    {
                        ntt_basic::ntt_2point(input[pos], input[pos + 1]);
                    }
                    if (bit_rev)
                    {
                        binary_reverse_swap(input, ntt_len);
                    }
                }
                template <typename T>
                static constexpr void ntt_split_radix_dit(T input[], size_t ntt_len)
                {
                    NTT_ALT<1, MOD, ROOT>::ntt_dit_template(input, ntt_len);
                }
                template <typename T>
                static constexpr void ntt_split_radix_dif(T input[], size_t ntt_len)
                {
                    NTT_ALT<1, MOD, ROOT>::ntt_dif_template(input, ntt_len);
                }
                template <typename T>
                static constexpr void ntt_dit(T input[], size_t ntt_len)
                {
                    ntt_split_radix_dit(input, ntt_len);
                }
                template <typename T>
                static constexpr void ntt_dif(T input[], size_t ntt_len)
                {
                    ntt_split_radix_dif(input, ntt_len);
                }
            };
            using ntt1 = NTT<NTT_MOD1, NTT_ROOT1>;
            using intt1 = NTT<ntt1::mod(), ntt1::iroot()>;

            using ntt2 = NTT<NTT_MOD2, NTT_ROOT2>;
            using intt2 = NTT<ntt2::mod(), ntt2::iroot()>;

            using ntt3 = NTT<NTT_MOD3, NTT_ROOT3>;
            using intt3 = NTT<ntt3::mod(), ntt3::iroot()>;

            using ntt4 = NTT<NTT_MOD4, NTT_ROOT4>;
            using intt4 = NTT<ntt4::mod(), ntt4::iroot()>;
        }
    }
}

using namespace std;
using namespace hint;
using namespace hint_transform;
using namespace hint_ntt;
template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    vector<T> result(out_len);
    size_t ntt_len = min_2pow(out_len);

    using ntt = ntt1;
    using intt = intt1;

    auto mod_ary1 = new ntt::NTTModInt32[ntt_len]();
    auto mod_ary2 = new ntt::NTTModInt32[ntt_len]();

    for (size_t i = 0; i < len1; i++)
    {
        mod_ary1[i] = in1[i];
    }
    for (size_t i = 0; i < len2; i++)
    {
        mod_ary2[i] = in2[i];
    }
    ntt::ntt_dif(mod_ary1, ntt_len);
    ntt::ntt_dif(mod_ary2, ntt_len);
    ary_mul(mod_ary1, mod_ary2, mod_ary1, ntt_len);
    intt::ntt_dit(mod_ary1, ntt_len);
    intt::ntt_basic::ntt_normalize(mod_ary1, ntt_len);
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = mod_ary1[i].data;
    }
    return result;
}
template <typename T>
void result_test(const vector<T> &res, UINT_64 ele)
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
#include <numeric>

int main()
{
    StopWatch w(1000);
    int n = 18;
    cin >> n;
    size_t len = 1ull << n; // 变换长度
    cout << "fft len:" << len << "\n";
    uint64_t ele = 9;
    vector<uint32_t> in1(len / 2, ele);
    vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2,每个元素为ele的卷积
    w.start();
    vector<uint32_t> res = poly_multiply(in1, in2);
    w.stop();
    result_test(res, ele); // 结果校验
    cout << w.duration() << "ms\n";
}