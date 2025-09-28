#include "fft_avx.hpp"
#include <vector>
#include <iostream>
#include <chrono>

namespace string_util
{
    using namespace hint;
    using namespace transform;
    using namespace fft;
    class ItoStrBase10000
    {
    private:
        uint32_t table[10000]{};

    public:
        static constexpr uint32_t itosbase10000(uint32_t num)
        {
            uint32_t res = (num / 1000 % 10) | ((num / 100 % 10) << 8) |
                           ((num / 10 % 10) << 16) | ((num % 10) << 24);
            return res + '0' * 0x1010101;
        }
        constexpr ItoStrBase10000()
        {
            for (size_t i = 0; i < 10000; i++)
            {
                table[i] = itosbase10000(i);
            }
        }
        void tostr(char *str, uint32_t num) const
        {
            // *reinterpret_cast<uint32_t *>(str) = table[num];
            std::memcpy(str, &table[num], sizeof(num));
        }
        uint32_t tostr(uint32_t num) const
        {
            return table[num];
        }
    };

    class StrtoIBase100
    {
    private:
        static constexpr size_t TABLE_SIZE = size_t(1) << 15;
        uint16_t table[TABLE_SIZE]{};

    public:
        static constexpr uint16_t itosbase100(uint16_t num)
        {
            uint16_t res = (num / 10 % 10) | ((num % 10) << 8);
            return res + '0' * 0x0101;
        }
        constexpr StrtoIBase100()
        {
            for (size_t i = 0; i < TABLE_SIZE; i++)
            {
                table[i] = UINT16_MAX;
            }
            for (size_t i = 0; i < 100; i++)
            {
                table[itosbase100(i)] = i;
            }
        }
        uint16_t toInt(const char *str) const
        {
            uint16_t num;
            std::memcpy(&num, str, sizeof(num));
            return table[num];
        }
    };

    constexpr ItoStrBase10000 itosbase10000{};
    constexpr StrtoIBase100 strtoibase100{};

    inline uint32_t stobase10000(const char *s)
    {
        // return s[0] * 1000 + s[1] * 100 + s[2] * 10 + s[3] - '0' * 1111;
        return strtoibase100.toInt(s) * 100 + strtoibase100.toInt(s + 2);
    }

    template <typename T>
    inline size_t str_num_to_array_base10000(const char *str, size_t len, T *ary)
    {
        constexpr size_t BLOCK = 4;
        auto end = str + len, p = str;
        size_t i = 0;
        for (auto ed = end - len % BLOCK; p < ed; p += BLOCK, i++)
        {
            ary[i] = stobase10000(p);
        }
        size_t shift = 0;
        if (p < end)
        {
            size_t rem = end - p;
            int n = 0;
            for (; p < end; p++)
            {
                n = n * 10 + *p - '0';
            }
            shift = BLOCK - rem;
            for (; rem < BLOCK; rem++)
            {
                n *= 10;
            }
            ary[i] = n;
            i++;
        }
        return shift;
    }

    template <typename T>
    inline size_t conv_to_str_base10000(const T *ary, size_t conv_len, size_t shift, char *res, size_t &res_len)
    {
        constexpr size_t BLOCK = 4, BASE = 10000;
        res_len = (conv_len + 1) * BLOCK;
        auto end = res + res_len;
        size_t i = conv_len;
        uint64_t carry = 0;
        while (i > 0)
        {
            i--;
            end -= BLOCK;
            carry += ary[i];
            itosbase10000.tostr(end, carry % BASE);
            carry /= BASE;
        }
        assert(carry < BASE);
        end -= 4;
        itosbase10000.tostr(end, carry);
        while (*end == '0')
        {
            end++;
        }
        size_t offset = end - res;
        res_len -= (offset + shift);
        return offset;
    }

    // return result begin
    inline char *big_mul(const char *str1, size_t len1, const char *str2, size_t len2, char *res, size_t &res_len)
    {
        constexpr size_t BLOCK = 4, BASE = 10000;
        size_t block_len1 = (len1 + BLOCK - 1) / BLOCK, block_len2 = (len2 + BLOCK - 1) / BLOCK;
        size_t conv_len = block_len1 + block_len2 - 1, fft_len = hint::int_ceil2(conv_len);
        fft_len = std::max(fft_len, FFTAVX::SHORT_LEN * 4);
        AlignMem<Float64> ary1(fft_len), ary2(fft_len);
        size_t shift = str_num_to_array_base10000(str1, len1, &ary1[0]);
        shift += str_num_to_array_base10000(str2, len2, &ary2[0]);
        fill_zero(ary1.begin() + block_len1, ary1.end());
        fill_zero(ary2.begin() + block_len2, ary2.end());
        real_conv_avx<true>(ary1.begin(), ary2.begin(), fft_len);
        return res + conv_to_str_base10000((uint64_t *)ary1.begin(), conv_len, shift, res, res_len);
    }

    inline size_t preserve_strlen(size_t len1, size_t len2)
    {
        constexpr size_t BLOCK = 4;
        size_t block_len1 = (len1 + BLOCK - 1) / BLOCK, block_len2 = (len2 + BLOCK - 1) / BLOCK;
        return (block_len1 + block_len2) * BLOCK;
    }

    inline size_t digit_strlen(const char *str)
    {
        auto begin = str;
        while (*str >= '0')
        {
            str++;
        }
        return str - begin;
    }

    inline void mul()
    {
        constexpr size_t STR_LEN = 2000008;
        static char str[STR_LEN];
        fread(str, 1, STR_LEN, stdin);
        char *s1 = str, *s2;
        size_t len1 = digit_strlen(str);
        s2 = s1 + len1;
        while (*s2 < '0')
        {
            s2++;
        }
        size_t len2 = digit_strlen(s2);
        size_t res_len = preserve_strlen(len1, len2);
        auto begin = big_mul(s1, len1, s2, len2, str, res_len);
        auto end = begin + res_len;
        fwrite(begin, 1, res_len, stdout);
    }
}

template <typename T>
inline std::vector<T> poly_multiply(const std::vector<T> &in1, const std::vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size();
    size_t conv_len = len1 + len2;
    size_t float_len = hint::int_ceil2(conv_len);
    size_t fft_len = float_len / 2;
    auto p1 = (double *)_mm_malloc(float_len * sizeof(double), 32);
    auto p2 = (double *)_mm_malloc(float_len * sizeof(double), 32);
    std::copy(in1.begin(), in1.end(), p1);
    std::copy(in2.begin(), in2.end(), p2);
    std::fill(p1 + len1, p1 + float_len, 0);
    std::fill(p2 + len2, p2 + float_len, 0);
    hint::transform::fft::real_conv_avx<true>(p1, p2, float_len);
    auto i64_p = reinterpret_cast<uint64_t *>(p1);
    std::vector<T> res(conv_len);
    for (size_t i = 0; i < conv_len; i++)
    {
        res[i] = i64_p[i];
    }
    _mm_free(p1);
    _mm_free(p2);
    return res;
}

template <typename T>
inline void result_test(const std::vector<T> &res, uint64_t ele1, uint64_t ele2)
{
    size_t len = res.size();
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele1 * ele2;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele1 * ele2;
        uint64_t y = res[i];
        if (x != y)
        {
            std::cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    std::cout << "success\n";
}

void perf_conv()
{
    int n = 13;
    std::cin >> n;
    size_t len = size_t(1) << n; // 变换长度
    std::cout << "conv len:" << len << "\n";
    int MAX = 1 << 8;
    uint64_t ele1 = MAX - 1, ele2 = MAX - 1;
    std::vector<uint64_t> in1(len / 2, 0);
    std::vector<uint64_t> in2(len / 2, 0); // 计算两个长度为len/2，每个元素为ele的卷积
    for (size_t i = 0; i < len / 2; i++)
    {
        in1[i] = ele1;
        in2[i] = ele2;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::vector<uint64_t> res = poly_multiply(in1, in2);
    auto t2 = std::chrono::steady_clock::now();
    for (auto i : res)
    {
        // std::cout << i << " ";
    }
    std::cout << "\n";
    result_test<uint64_t>(res, ele1, ele2); // 结果校验
    std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us\n";
}
int main()
{
    // mul(); // 计算大数乘法
    perf_conv(); // 卷积性能测试
}