#include <vector>
#include <complex>
#include <iostream>
#include <future>
#include <chrono>
#include <string>
#include <bitset>
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
    constexpr size_t FHT_MAX_LEN = size_t(1) << 18;

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
        template <typename T>
        inline void transform_2point(T &sum, T &diff)
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
        namespace hint_fht
        {
            template <typename FloatTy>
            class DynamicTable
            {
            public:
                using Complex = std::complex<FloatTy>;
                using CompVec = std::vector<Complex>;
                DynamicTable() {}
                DynamicTable(int log_len, int factor, bool conj = false)
                {
                    size_t vec_len = (1 << log_len) / 4;
                    table = CompVec(vec_len);
                    init(factor, conj);
                }
                void init(int factor, bool conj)
                {
                    size_t len = table.size() * 4;
                    FloatTy unity = -HINT_2PI * factor / len;
                    if (conj)
                    {
                        unity = -unity;
                    }
                    for (size_t i = 0; i < table.size(); i++)
                    {
                        table[i] = unit_root(unity * i);
                    }
                }
                const Complex &operator[](size_t n) const
                {
                    return table[n];
                }
                Complex &operator[](size_t n)
                {
                    return table[n];
                }
                auto get_it(size_t n = 0)
                {
                    return &table[n];
                }

            private:
                CompVec table;
            };
            template <size_t LEN, typename FloatTy>
            struct FHT
            {
                enum
                {
                    fht_len = LEN,
                    half_len = LEN / 2,
                    quater_len = LEN / 4,
                    log_len = hint_log2(fht_len)
                };
                using HalfFHT = FHT<half_len, FloatTy>;
                static DynamicTable<FloatTy> TABLE;
                template <typename FloatIt>
                static void dit(FloatIt in_out)
                {
                    HalfFHT::dit(in_out);
                    HalfFHT::dit(in_out + half_len);

                    transform_2point(in_out[0], in_out[half_len]);
                    transform_2point(in_out[quater_len], in_out[half_len + quater_len]);

                    auto it0 = in_out + 1, it1 = in_out + half_len - 1;
                    auto it2 = in_out + half_len + 1, it3 = in_out + fht_len - 1;
                    auto omega_it = TABLE.get_it(1);
                    for (; it0 < it1; ++it0, --it1, ++it2, --it3, omega_it++)
                    {
                        auto temp0 = it2[0], temp1 = it3[0];
                        auto omega = omega_it[0];
                        auto temp2 = temp0 * omega.real() + temp1 * omega.imag();
                        auto temp3 = temp0 * omega.imag() - temp1 * omega.real();
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
                    transform_2point(in_out[0], in_out[half_len]);
                    transform_2point(in_out[quater_len], in_out[half_len + quater_len]);

                    auto it0 = in_out + 1, it1 = in_out + half_len - 1;
                    auto it2 = in_out + half_len + 1, it3 = in_out + fht_len - 1;
                    auto omega_it = TABLE.get_it(1);
                    for (; it0 < it1; ++it0, --it1, ++it2, --it3, omega_it++)
                    {
                        auto temp0 = it0[0], temp1 = it1[0];
                        auto temp2 = it2[0], temp3 = it3[0];
                        it0[0] = temp0 + temp2;
                        it1[0] = temp1 + temp3;
                        temp0 = temp0 - temp2;
                        temp1 = temp1 - temp3;
                        auto omega = omega_it[0];
                        it2[0] = temp0 * omega.real() + temp1 * omega.imag();
                        it3[0] = temp0 * omega.imag() - temp1 * omega.real();
                    }

                    HalfFHT::dif(in_out);
                    HalfFHT::dif(in_out + half_len);
                }
            };
            template <size_t LEN, typename FloatTy>
            DynamicTable<FloatTy> FHT<LEN, FloatTy>::TABLE(FHT<LEN, FloatTy>::log_len, 1, true);

            template <typename FloatTy>
            struct FHT<0, FloatTy>
            {
                template <typename FloatIt>
                static void dit(FloatIt in_out) {}
                template <typename FloatIt>
                static void dif(FloatIt in_out) {}
                template <typename FloatIt>
                static void dif2(FloatIt in_out0, FloatIt in_out1) {}
            };

            template <typename FloatTy>
            struct FHT<1, FloatTy>
            {
                template <typename FloatIt>
                static void dit(FloatIt in_out) {}
                template <typename FloatIt>
                static void dif(FloatIt in_out) {}
                template <typename FloatIt>
                static void dif2(FloatIt in_out0, FloatIt in_out1) {}
            };

            template <typename FloatTy>
            struct FHT<2, FloatTy>
            {
                template <typename FloatIt>
                static void dit(FloatIt in_out)
                {
                    transform_2point(in_out[0], in_out[1]);
                }
                template <typename FloatIt>
                static void dif(FloatIt in_out)
                {
                    transform_2point(in_out[0], in_out[1]);
                }
                template <typename FloatIt>
                static void dif2(FloatIt in_out0, FloatIt in_out1)
                {
                    dif(in_out0);
                    dif(in_out1);
                }
            };

            template <typename FloatTy>
            struct FHT<4, FloatTy>
            {
                template <typename FloatIt>
                static void dit(FloatIt in_out)
                {
                    auto temp0 = in_out[0], temp1 = in_out[1];
                    auto temp2 = in_out[2], temp3 = in_out[3];
                    transform_2point(temp0, temp1);
                    transform_2point(temp2, temp3);
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
                    transform_2point(temp0, temp2);
                    transform_2point(temp1, temp3);
                    in_out[0] = temp0 + temp1;
                    in_out[1] = temp0 - temp1;
                    in_out[2] = temp2 + temp3;
                    in_out[3] = temp2 - temp3;
                }
                template <typename FloatIt>
                static void dif2(FloatIt in_out0, FloatIt in_out1)
                {
                    dif(in_out0);
                    dif(in_out1);
                }
            };

            template <typename FloatTy>
            struct FHT<8, FloatTy>
            {
                template <typename FloatIt>
                static void dit(FloatIt in_out)
                {
                    auto temp0 = in_out[0], temp1 = in_out[1];
                    auto temp2 = in_out[2], temp3 = in_out[3];
                    auto temp4 = in_out[4], temp5 = in_out[5];
                    auto temp6 = in_out[6], temp7 = in_out[7];
                    transform_2point(temp0, temp1);
                    transform_2point(temp2, temp3);
                    transform_2point(temp4, temp5);
                    transform_2point(temp6, temp7);
                    transform_2point(temp0, temp2);
                    transform_2point(temp1, temp3);
                    transform_2point(temp4, temp6);
                    transform_2point(temp5, temp7);

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
                    transform_2point(temp0, temp4);
                    transform_2point(temp1, temp5);
                    transform_2point(temp2, temp6);
                    transform_2point(temp3, temp7);
                    transform_2point(temp0, temp1);
                    transform_2point(temp2, temp3);
                    in_out[0] = temp0 + temp2;
                    in_out[1] = temp1 + temp3;
                    in_out[2] = temp0 - temp2;
                    in_out[3] = temp1 - temp3;
                    static constexpr decltype(temp0) SQRT_2_2 = 0.70710678118654757;
                    temp0 = (temp5 + temp7) * SQRT_2_2;
                    temp2 = (temp5 - temp7) * SQRT_2_2;
                    transform_2point(temp4, temp6);
                    transform_2point(temp0, temp2);
                    in_out[4] = temp4 + temp0;
                    in_out[5] = temp4 - temp0;
                    in_out[6] = temp6 + temp2;
                    in_out[7] = temp6 - temp2;
                }
                template <typename FloatIt>
                static void dif2(FloatIt in_out0, FloatIt in_out1)
                {
                    dif(in_out0);
                    dif(in_out1);
                }
            };

            // 辅助选择函数
            template <size_t LEN = 1>
            void fht_dit_template_alt(Float64 *input, size_t fft_len)
            {
                if (fft_len < LEN)
                {
                    fht_dit_template_alt<LEN / 2>(input, fft_len);
                    return;
                }
                FHT<LEN, Float64>::dit(input);
            }
            template <>
            void fht_dit_template_alt<0>(Float64 *input, size_t fft_len) {}

            // 辅助选择函数
            template <size_t LEN = 1>
            void fht_dif_template_alt(Float64 *input, size_t fft_len)
            {
                if (fft_len < LEN)
                {
                    fht_dif_template_alt<LEN / 2>(input, fft_len);
                    return;
                }
                FHT<LEN, Float64>::dif(input);
            }
            template <>
            void fht_dif_template_alt<0>(Float64 *input, size_t fft_len) {}

            auto fht_dit = fht_dit_template_alt<FHT_MAX_LEN>;
            auto fht_dif = fht_dif_template_alt<FHT_MAX_LEN>;

            // FHT加速卷积
            void fht_convolution(Float64 fht_ary1[], Float64 fht_ary2[], Float64 out[], size_t fht_len)
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
                transform_2point(temp0, temp1);
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
                        transform_2point(temp0, temp1);
                        *it4 = (temp2 * temp0 + temp3 * temp1) * inv;
                        *it5 = (temp3 * temp0 - temp2 * temp1) * inv;
                        temp0 = *(it1 - 1), temp1 = *(it0 + 1), temp2 = *(it3 - 1), temp3 = *(it2 + 1);
                        transform_2point(temp0, temp1);
                        *(it5 - 1) = (temp2 * temp0 + temp3 * temp1) * inv;
                        *(it4 + 1) = (temp3 * temp0 - temp2 * temp1) * inv;
                    }
                }
                fht_dit(out, fht_len);
            }
        }
    }
    // 进行2^64进制的乘法
    void FHTMul(uint64_t *out, const uint64_t *in1, size_t in_len1, const uint64_t *in2, size_t in_len2)
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
    void FHTSquare(uint64_t *out, const uint64_t *in, size_t in_len)
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

// Example

// 从二进制字符串到2^64进制大整数组
vector<uint64_t> from_binary_str(const string &s)
{
    auto binstr_to_ui64 = [](const char *p, size_t len)
    {
        if (len == 0)
        {
            return uint64_t(0);
        }
        len = min(size_t(64), len);
        string tmp(p, p + len);
        return bitset<64>(tmp.data()).to_ullong();
    };
    size_t in_len = s.size();
    vector<uint64_t> res;
    size_t i = in_len;
    while (i >= 64)
    {
        i -= 64;
        res.push_back(binstr_to_ui64(s.data() + i, 64));
    }
    if (i > 0)
    {
        res.push_back(binstr_to_ui64(s.data(), i));
    }
    return res;
}

// 从2^64进制大整数组到二进制字符串
string to_binary_str(const vector<uint64_t> &v)
{
    size_t in_len = v.size();
    string res;
    size_t i = in_len;
    while (i > 0)
    {
        i--;
        res += bitset<64>(v[i]).to_string();
    }
    i = 0;
    while (res[i] == '0')
    {
        i++;
    }
    return res.substr(i, res.size() - i);
}

int main()
{
    // 正确性测试
    string s1("10010001101011110010110100101101011010010100101101101001010010100101010100101110010100110110101001010000010101010101010010101001010101001010101001001010000110101110011100101010101110000010101101010010101010111100100010100101010010110100101101001011010110100101001010010111001010011011010100101000001010101001010101110000010101101010010010110100110101001001011010011010100101010101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010100101010010101010010101010010010100100111001010101011100000101011010100101010101111001000101001010111001010011011010100101000001010101011010100101101001010100101101001101010010010110100010010110101001011010010101001011010011010100100101101000100101");
    string s2("10010001101011110010001100100011010111100101101001011010110100101001011010010001101011110010110100101101011010010100101101101001010010100101010100101110010100100011010111100101101001011010110100101001011011011010101010101001010100101010100101010100100101001101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010101010100010100101001010101001011100101101001010010100101010100101110010101011100111001010101011100000101011010100100101111010100100101101001101010010010110100110101000101001010100101010100101010100100101001101111001000101001010100101101001011010010110101101001010010100101110010100110110101001010000010101010101010010101001010101001010101001001010010011100101010101110");
    auto v1 = from_binary_str(s1);
    auto v2 = from_binary_str(s2);
    vector<uint64_t> v3(v1.size() + v2.size());
    vector<uint64_t> v4(v2.size() * 2);

    // 乘法
    hint::FHTMul(v3.data(), v1.data(), v1.size(), v2.data(), v2.size()); // 计算s1,s2乘积

    // 平方
    hint::FHTSquare(v4.data(), v2.data(), v2.size()); // 计算s2平方

    auto mul_res = to_binary_str(v3);
    auto square_res = to_binary_str(v4);

    cout << "mul_res: " << mul_res << std::endl;       // 101001011100111111000110101010110101110100000001101000110100110110001001010100101001110100011000001001000100011001010111011011011110101111010101000100010001101101100010011100110000111011110010010011011111001100111100010111011101000100010010101100011110110000010100100011001101110110111110100011110010111110010001011011000100011010110100011110111011001011001100111110110101001001110011100101001101101000010011011110000100100101110101001100111001111010100000010110111100010111001000010001101100000101000111010000111101000000001011110100010010100100000010100000010100110111001100111111110101000111011111110100100000011010001101010111101001111111101000100100100100000111010111110110011111101001001110111011100101100011111110011011011100001011010100101110110100000101011111111000110011100001001111001111110010011011001010000111111101000010001111011000010111111010010101110001011110011111010101110000101110010100000001101101101001110000111011100101100101110101100000011100111100111101001110010001100110100111001100001000001010001010011000101100111000101111001111111000111001111010101000100000000010011111101111110010011011011101101010001010000101001011100000010101111011110001101000001000101100010110000101111000111100100101000101101011000000001001110111010101011110011100001011001111110000101011001000010100100110011111000101100101110111100011100111100111000101000000111110001011100100111100001100011001110001100001111011110101001110111011100100110
    cout << "square_res: " << square_res << std::endl; // 101001011100111110111011001111100100000000100000000101100011010010111101100101111001000111101100101000111001111001110000101010110111101011011001111111110110011110100000001000100101111011111101100001000001000100010110001011101011111100000011010101111001010111111110101111001111111101000001000110001010010111001100011001100101101111100011010001110011100101101000001110011011101101100000111001001100000000100110011111100111001001110000001111010101110000011111010100000101010011111110000101101000001100100110101011010111100101011001111011111001111001011000010111011111010101101000001101101110100100000001100110000000111000000100111110011010011110010000100001111000101101111011101011000111000101000101011101000011011110100101010011000000001100111100101111100111011000110010000101000000001001110010101000111000110100010000010011000100011110001111000100011000111101111001111000110110000011010011010000000110000001000100111110011101101110000010011001011001111111000100111110011101111110110111010110111000111110000110001111001000101111001011110011100000101001010001011011001011010010001011111011000101001010001010001000001100100111101110101001010011111011110000010100101100011011010111101010101101010100101101100000100010111010100111101101000101101110101100000110101101001110010100000000110001101100110101111111010111101001111111011000110010110011001111110011100010100111100010001110001111010000111101110001100010110100001110011101100110000111001000100

    cout << "------------------------------------------------------------------------------------------\n";
    cout << "Performance:\n";
    // 性能测试
    string s3(8192, '1'); // 计算8192位二进制数的乘法和平方
    auto v5 = from_binary_str(s3);
    auto v6 = from_binary_str(s3);

    vector<uint64_t> v7(v5.size() + v6.size());
    vector<uint64_t> v8(v6.size() * 2);

    auto t1 = std::chrono::system_clock::now();
    hint::FHTMul(v7.data(), v5.data(), v5.size(), v6.data(), v6.size());
    auto t2 = std::chrono::system_clock::now();
    hint::FHTSquare(v8.data(), v6.data(), v6.size());
    auto t3 = std::chrono::system_clock::now();

    cout << to_binary_str(v7) << "\n";
    cout << to_binary_str(v8) << "\n";

    cout << "Multiply time:" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << "us\n";
    cout << "Square time:" << chrono::duration_cast<chrono::microseconds>(t3 - t2).count() << "us\n";
}