#include <iostream>
#include <complex>
#include <cstdint>
#include <immintrin.h>

// #define __AVX__

#ifndef HINT_SIMD_HPP
#define HINT_SIMD_HPP

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

    template <typename T, size_t ALIGN = 64>
    class AlignMem
    {
    public:
        using Ptr = T *;
        using ConstPtr = const T *;
        ~AlignMem()
        {
            if (ptr)
            {
                _mm_free(ptr);
            }
        };
        AlignMem() : ptr(nullptr), len(0) {}
        AlignMem(size_t n) : ptr(reinterpret_cast<Ptr>(_mm_malloc(n * sizeof(T), ALIGN))), len(n) {}
        AlignMem(const AlignMem &) = delete;
        AlignMem &operator=(const AlignMem &) = delete;
        T &operator[](size_t i)
        {
            return ptr[i];
        }
        const T &operator[](size_t i) const
        {
            return ptr[i];
        }
        Ptr begin()
        {
            return ptr;
        }
        Ptr end()
        {
            return ptr + len;
        }
        ConstPtr begin() const
        {
            return ptr;
        }
        ConstPtr end() const
        {
            return ptr + len;
        }
        size_t size() const
        {
            return len;
        }

    private:
        T *ptr;
        size_t len;
    };

#ifdef __AVX__
    template <typename YMM>
    inline void transpose64_2X4(YMM &row0, YMM &row1)
    {
        auto t0 = _mm256_unpacklo_pd(__m256d(row0), __m256d(row1)); // 0,1,2,3 4,5,6,7 -> 0,4,2,6
        auto t1 = _mm256_unpackhi_pd(__m256d(row0), __m256d(row1)); // 0,1,2,3 4,5,6,7 -> 1,5,3,7

        row0 = YMM(_mm256_permute2f128_pd(t0, t1, 0x20)); // 0,4,2,6 1,5,3,7 -> 0,4,1,5
        row1 = YMM(_mm256_permute2f128_pd(t0, t1, 0x31)); // 0,4,2,6 1,5,3,7 -> 2,6,3,7
    }
    template <typename YMM>
    inline void transpose64_4X2(YMM &row0, YMM &row1)
    {
        auto t0 = _mm256_permute2f128_pd(__m256d(row0), __m256d(row1), 0x20); // 0,1,2,3 4,5,6,7 -> 0,1,4,5
        auto t1 = _mm256_permute2f128_pd(__m256d(row0), __m256d(row1), 0x31); // 0,1,2,3 4,5,6,7 -> 2,3,6,7
        row0 = YMM(_mm256_unpacklo_pd(t0, t1));                               // 0,1,4,5 2,3,6,7 -> 0,4,2,6
        row1 = YMM(_mm256_unpackhi_pd(t0, t1));                               // 0,1,4,5 2,3,6,7 -> 1,5,3,7
    }

    template <typename YMM>
    inline void transpose64_4X4(YMM &row0, YMM &row1, YMM &row2, YMM &row3)
    {
        auto t0 = _mm256_unpacklo_pd(__m256d(row0), __m256d(row1)); // 0,1,2,3 4,5,6,7 -> 0,4,2,6
        auto t1 = _mm256_unpackhi_pd(__m256d(row0), __m256d(row1)); // 0,1,2,3 4,5,6,7 -> 1,5,3,7
        auto t2 = _mm256_unpacklo_pd(__m256d(row2), __m256d(row3)); // 8,9,10,11 12,13,14,15 -> 8,12,10,14
        auto t3 = _mm256_unpackhi_pd(__m256d(row2), __m256d(row3)); // 8,9,10,11 12,13,14,15 -> 9,13,11,15

        row0 = YMM(_mm256_permute2f128_pd(t0, t2, 0x20));
        row1 = YMM(_mm256_permute2f128_pd(t1, t3, 0x20));
        row2 = YMM(_mm256_permute2f128_pd(t0, t2, 0x31));
        row3 = YMM(_mm256_permute2f128_pd(t1, t3, 0x31));
    }

    class Float64X4
    {
    public:
        using F64 = double;
        using F64X4 = Float64X4;
        Float64X4() = default;
        Float64X4(__m256d in_data) : data(in_data) {}
        Float64X4(F64 in_data) : data(_mm256_set1_pd(in_data)) {}
        Float64X4(const F64 *in_data) : data(_mm256_load_pd(in_data)) {}

        F64X4 operator+(const F64X4 &other) const
        {
            return _mm256_add_pd(data, other.data);
        }
        F64X4 operator-(const F64X4 &other) const
        {
            return _mm256_sub_pd(data, other.data);
        }
        F64X4 operator*(const F64X4 &other) const
        {
            return _mm256_mul_pd(data, other.data);
        }
        F64X4 operator/(const F64X4 &other) const
        {
            return _mm256_div_pd(data, other.data);
        }

        F64X4 &operator+=(const F64X4 &other)
        {
            return *this = *this + other;
        }
        F64X4 &operator-=(const F64X4 &other)
        {
            return *this = *this - other;
        }
        F64X4 &operator*=(const F64X4 &other)
        {
            return *this = *this * other;
        }
        F64X4 &operator/=(const F64X4 &other)
        {
            return *this = *this / other;
        }
        F64X4 floor() const
        {
            return _mm256_floor_pd(data);
        }
        static F64X4 HalfSqrt2()
        {
            return broadcastFrom1(&HSQRT2);
        }
        static F64X4 NegHalfSqrt2()
        {
            return broadcastFrom1(&NHSQRT2);
        }
        // a * b + c
        static F64X4 fmadd(const F64X4 &a, const F64X4 &b, const F64X4 &c)
        {
#ifdef __FMA__
            return _mm256_fmadd_pd(a.data, b.data, c.data);
#else
#pragma message("No FMA support")
            return a * b + c;
#endif
        }
        // a * b - c
        static F64X4 fmsub(const F64X4 &a, const F64X4 &b, const F64X4 &c)
        {
#ifdef __FMA__
            return _mm256_fmsub_pd(a.data, b.data, c.data);
#else
#pragma message("No FMA support")
            return a * b - c;
#endif
        }
#ifdef __AVX2__
        template <int N>
        F64X4 permute4x64() const
        {
            return _mm256_permute4x64_pd(data, N);
        }
#else
        template <int N>
        F64X4 permute4x64() const
        {
            alignas(32) uint64_t arr[4];
            alignas(32) uint64_t dst[4];
            this->store(reinterpret_cast<F64 *>(arr));
            dst[0] = arr[(N >> 0) & 3];
            dst[1] = arr[(N >> 2) & 3];
            dst[2] = arr[(N >> 4) & 3];
            dst[3] = arr[(N >> 6) & 3];
            return fromMem(reinterpret_cast<const F64 *>(dst));
        }
#endif
        static F64X4 extractEven64X4(const F64X4 &in0, const F64X4 &in1)
        {
            F64X4 result = _mm256_unpacklo_pd(in0.data, in1.data); // 0,1,2,3 4,5,6,7 -> 0,4,2,6
            return result.permute4x64<0b11011000>();               // 0,4,2,6 -> 0,2,4,6
        }

        template <int N>
        F64X4 permute() const
        {
            return _mm256_permute_pd(data, N);
        }
        F64X4 reverse() const
        {
            return permute4x64<0b00011011>();
        }
        void load(const F64 *p)
        {
            data = _mm256_load_pd(p);
        }
        void loadu(const F64 *p)
        {
            data = _mm256_loadu_pd(p);
        }
        void load1(const F64 *p)
        {
            data = _mm256_broadcast_sd(p);
        }
        static F64X4 broadcastFrom1(const F64 *p)
        {
            return _mm256_broadcast_sd(p);
        }
        static F64X4 fromMem(const F64 *p)
        {
            return _mm256_load_pd(p);
        }
        static F64X4 fromUMem(const F64 *p)
        {
            return _mm256_loadu_pd(p);
        }
        void store(F64 *p) const
        {
            _mm256_store_pd(p, data);
        }
        void storeu(F64 *p) const
        {
            _mm256_storeu_pd(p, data);
        }
        operator __m256d() const
        {
            return data;
        }
#ifdef __AVX2__
        // Convert positive double to int64
        __m256i toI64X4() const
        {
            constexpr uint64_t mask = (uint64_t(1) << 52) - 1;
            constexpr uint64_t offset = (uint64_t(1) << 10) - 1;
            const __m256i f64bits = _mm256_castpd_si256(data);
            __m256i tail = _mm256_and_si256(f64bits, _mm256_set1_epi64x(mask));
            tail = _mm256_or_si256(tail, _mm256_set1_epi64x(mask + 1));
            __m256i exp = _mm256_srli_epi64(f64bits, 52);
            exp = _mm256_sub_epi64(_mm256_set1_epi64x(offset + 52), exp);
            return _mm256_srlv_epi64(tail, exp);
        }
#else
#pragma message("No AVX2 support")
        __m256i toI64X4() const
        {
            alignas(32) F64 arr[4];
            alignas(32) int64_t i64_arr[4];
            this->store(arr);
            i64_arr[0] = arr[0];
            i64_arr[1] = arr[1];
            i64_arr[2] = arr[2];
            i64_arr[3] = arr[3];
            return _mm256_load_si256(reinterpret_cast<const __m256i *>(i64_arr));
        }
#endif
        template <int N>
        F64 nthEle() const
        {
            union F64I64
            {
                int64_t i64;
                F64 f64;
            } temp;
            temp.i64 = _mm256_extract_epi64(__m256i(data), N);
            return temp.f64;
        }

        void print() const
        {
            std::cout << "[" << nthEle<0>() << "," << nthEle<1>()
                      << "," << nthEle<2>() << "," << nthEle<3>() << "]" << std::endl;
        }

    private:
        __m256d data;
        static constexpr F64 HSQRT2 = 0.70710678118654752440084436210485;
        static constexpr F64 NHSQRT2 = -HSQRT2;
    };

    constexpr Float64X4::F64 Float64X4::HSQRT2;
    constexpr Float64X4::F64 Float64X4::NHSQRT2;

    struct Complex64X4
    {
        using C64X4 = Complex64X4;
        using F64X4 = Float64X4;
        using F64 = double;
        Complex64X4() {}
        Complex64X4(F64X4 real, F64X4 imag) : real(real), imag(imag) {}
        Complex64X4(const F64 *p) : real(p), imag(p + 4) {}
        Complex64X4(const F64 *p_real, const F64 *p_imag) : real(p_real), imag(p_imag) {}
        C64X4 operator+(const C64X4 &other) const
        {
            return C64X4(real + other.real, imag + other.imag);
        }
        C64X4 operator-(const C64X4 &other) const
        {
            return C64X4(real - other.real, imag - other.imag);
        }
        C64X4 operator*(const F64X4 &other) const
        {
            return C64X4(real * other, imag * other);
        }
        C64X4 mul(const C64X4 &other) const
        {
            const F64X4 ii = imag * other.imag;
            const F64X4 ri = real * other.imag;
            const F64X4 r = F64X4::fmsub(real, other.real, ii);
            const F64X4 i = F64X4::fmadd(imag, other.real, ri);
            return C64X4(r, i);
        }
        C64X4 mulConj(const C64X4 &other) const
        {
            const F64X4 ii = imag * other.imag;
            const F64X4 ri = real * other.imag;
            const F64X4 r = F64X4::fmadd(real, other.real, ii);
            const F64X4 i = F64X4::fmsub(imag, other.real, ri);
            return C64X4(r, i);
        }

        C64X4 reverse() const
        {
            return C64X4(real.reverse(), imag.reverse());
        }
        // exp{i*theta*k},k in {0,1,2,3}
        static C64X4 omegaSeq0To3(F64 theta, F64 begin = 0)
        {
            F64 real_arr[4] = {cos(begin), cos(theta + begin), cos(2 * theta + begin), cos(3 * theta + begin)};
            F64 imag_arr[4] = {sin(begin), sin(theta + begin), sin(2 * theta + begin), sin(3 * theta + begin)};
            return C64X4(F64X4(real_arr), F64X4(imag_arr));
        }

        template <typename T>
        void load(const T *p, std::false_type)
        {
            this->load(p);
        }
        // From RIRI permutation
        template <typename T>
        void load(const T *p, std::true_type)
        {
            this->load(p);
            *this = this->toRRIIPermu();
        }
        template <typename T>
        void load(const T *p)
        {
            real.load(reinterpret_cast<const F64 *>(p));
            imag.load(reinterpret_cast<const F64 *>(p) + 4);
        }
        template <typename T>
        void loadu(const T *p)
        {
            real.loadu(reinterpret_cast<const F64 *>(p));
            imag.loadu(reinterpret_cast<const F64 *>(p) + 4);
        }
        void load1(const F64 *real_p, const F64 *imag_p)
        {
            real.load1(real_p);
            imag.load1(imag_p);
        }

        template <typename T>
        void store(T *p, std::false_type) const
        {
            this->store(p);
        }
        // To RIRI permutation
        template <typename T>
        void store(T *p, std::true_type) const
        {
            this->toRIRIPermu().store(p);
        }
        template <typename T>
        void store(T *p) const
        {
            real.store(reinterpret_cast<F64 *>(p));
            imag.store(reinterpret_cast<F64 *>(p) + 4);
        }
        template <typename T>
        void storeu(T *p) const
        {
            real.storeu(reinterpret_cast<F64 *>(p));
            imag.storeu(reinterpret_cast<F64 *>(p) + 4);
        }
        C64X4 square() const
        {
            const F64X4 ii = imag * imag;
            const F64X4 ri = real * imag;
            const F64X4 r = F64X4::fmsub(real, real, ii);
            const F64X4 i = ri + ri;
            return C64X4(r, i);
        }
        C64X4 cube() const
        {
            const F64X4 rr = real * real;
            const F64X4 ii = imag * imag;
            const F64X4 rr3 = rr + rr + rr;
            const F64X4 ii3 = ii + ii + ii;
            const F64X4 r = real * (rr - ii3);
            const F64X4 i = imag * (rr3 - ii);
            return C64X4(r, i);
        }
        C64X4 toRIRIPermu() const
        {
            C64X4 res = *this;
            transpose64_2X4(res.real, res.imag);
            return res;
        }
        C64X4 toRRIIPermu() const
        {
            C64X4 res = *this;
            transpose64_4X2(res.real, res.imag);
            return res;
        }
        void print() const
        {
            alignas(32) F64 real_arr[4]{}, imag_arr[4]{};
            real.storeu(real_arr);
            imag.storeu(imag_arr);
            std::cout << "[(" << real_arr[0] << ", " << imag_arr[0] << "), ("
                      << real_arr[1] << ", " << imag_arr[1] << "), ("
                      << real_arr[2] << ", " << imag_arr[2] << "), ("
                      << real_arr[3] << ", " << imag_arr[3] << ")]" << std::endl;
        }

        C64X4 transToI64(std::false_type) const
        {
            return *this;
        }
        C64X4 transToI64(std::true_type) const
        {
            constexpr int64_t F1_2 = 4602678819172646912; // magic::bit_cast<int64_t>(0.5);
            auto F1_2X4 = F64X4(__m256d(_mm256_set1_epi64x(F1_2)));
            auto real_i64 = (real + F1_2X4).toI64X4();
            auto imag_i64 = (imag + F1_2X4).toI64X4();
            return C64X4(__m256d(real_i64), __m256d(imag_i64));
        }

        F64X4 real, imag;
    };
#else
#pragma message("AVX is not supported")
#endif

#ifdef __AVX512F__

    template <typename ZMM>
    inline void transpose64_8X8(ZMM &row0, ZMM &row1, ZMM &row2, ZMM &row3, ZMM &row4, ZMM &row5, ZMM &row6, ZMM &row7)
    {
        auto t0 = _mm512_unpacklo_pd(row0, row1);
        auto t1 = _mm512_unpackhi_pd(row0, row1);
        auto t2 = _mm512_unpacklo_pd(row2, row3);
        auto t3 = _mm512_unpackhi_pd(row2, row3);
        auto t4 = _mm512_unpacklo_pd(row4, row5);
        auto t5 = _mm512_unpackhi_pd(row4, row5);
        auto t6 = _mm512_unpacklo_pd(row6, row7);
        auto t7 = _mm512_unpackhi_pd(row6, row7);

        auto u0 = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
        auto u1 = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(2, 0, 2, 0));
        auto u2 = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(3, 1, 3, 1));
        auto u3 = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(3, 1, 3, 1));
        auto u4 = _mm512_shuffle_f64x2(t4, t6, _MM_SHUFFLE(2, 0, 2, 0));
        auto u5 = _mm512_shuffle_f64x2(t5, t7, _MM_SHUFFLE(2, 0, 2, 0));
        auto u6 = _mm512_shuffle_f64x2(t4, t6, _MM_SHUFFLE(3, 1, 3, 1));
        auto u7 = _mm512_shuffle_f64x2(t5, t7, _MM_SHUFFLE(3, 1, 3, 1));

        row0 = _mm512_shuffle_f64x2(u0, u4, _MM_SHUFFLE(2, 0, 2, 0));
        row1 = _mm512_shuffle_f64x2(u1, u5, _MM_SHUFFLE(2, 0, 2, 0));
        row2 = _mm512_shuffle_f64x2(u2, u6, _MM_SHUFFLE(2, 0, 2, 0));
        row3 = _mm512_shuffle_f64x2(u3, u7, _MM_SHUFFLE(2, 0, 2, 0));
        row4 = _mm512_shuffle_f64x2(u0, u4, _MM_SHUFFLE(3, 1, 3, 1));
        row5 = _mm512_shuffle_f64x2(u1, u5, _MM_SHUFFLE(3, 1, 3, 1));
        row6 = _mm512_shuffle_f64x2(u2, u6, _MM_SHUFFLE(3, 1, 3, 1));
        row7 = _mm512_shuffle_f64x2(u3, u7, _MM_SHUFFLE(3, 1, 3, 1));
    }

    template <typename ZMM>
    inline void transpose64_2X8(ZMM &row0, ZMM &row1)
    {
        static const __m512i idx0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        static const __m512i idx1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        ZMM r0_copy = row0;
        row0 = _mm512_permutex2var_pd(r0_copy, idx0, row1);
        row1 = _mm512_permutex2var_pd(r0_copy, idx1, row1);
    }

    template <typename ZMM>
    inline void transpose64_8X2(ZMM &row0, ZMM &row1)
    {
        static const __m512i idx0 = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        static const __m512i idx1 = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        ZMM r0_copy = row0;
        row0 = _mm512_permutex2var_pd(r0_copy, idx0, row1);
        row1 = _mm512_permutex2var_pd(r0_copy, idx1, row1);
    }

    template <typename ZMM>
    inline ZMM extract64_even8(ZMM x0, ZMM x1)
    {
        static const __m512i idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        return _mm512_permutex2var_pd(x0, idx_even, x1);
    }

    template <typename ZMM>
    inline ZMM extract64_odd8(ZMM x0, ZMM x1)
    {
        static const __m512i idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        return _mm512_permutex2var_pd(x0, idx_odd, x1);
    }

    class Float64X8
    {
    public:
        using F64 = double;
        using F64X8 = Float64X8;
        Float64X8() = default;
        Float64X8(__m512d in_data) : data(in_data) {}
        Float64X8(F64 in_data) : data(_mm512_set1_pd(in_data)) {}
        Float64X8(const F64 *in_data) : data(_mm512_load_pd(in_data)) {}

        F64X8 operator+(const F64X8 &other) const
        {
            return _mm512_add_pd(data, other.data);
        }
        F64X8 operator-(const F64X8 &other) const
        {
            return _mm512_sub_pd(data, other.data);
        }
        F64X8 operator*(const F64X8 &other) const
        {
            return _mm512_mul_pd(data, other.data);
        }
        F64X8 operator/(const F64X8 &other) const
        {
            return _mm512_div_pd(data, other.data);
        }

        F64X8 &operator+=(const F64X8 &other)
        {
            return *this = *this + other;
        }
        F64X8 &operator-=(const F64X8 &other)
        {
            return *this = *this - other;
        }
        F64X8 &operator*=(const F64X8 &other)
        {
            return *this = *this * other;
        }
        F64X8 &operator/=(const F64X8 &other)
        {
            return *this = *this / other;
        }
        static F64X8 HalfSqrt2()
        {
            return broadcastFrom1(&HSQRT2);
        }
        static F64X8 NegHalfSqrt2()
        {
            return broadcastFrom1(&NHSQRT2);
        }
        // a * b + c
        static F64X8 fmadd(const F64X8 &a, const F64X8 &b, const F64X8 &c)
        {
            return _mm512_fmadd_pd(a.data, b.data, c.data);
        }
        // a * b - c
        static F64X8 fmsub(const F64X8 &a, const F64X8 &b, const F64X8 &c)
        {
            return _mm512_fmsub_pd(a.data, b.data, c.data);
        }
        static F64X8 extractEven64X4(const F64X8 &in0, const F64X8 &in1)
        {
            return extract64_even8(in0.data, in1.data);
        }
        template <int N>
        F64X8 permute() const
        {
            return _mm512_permute_pd(data, N);
        }
        F64X8 reverse() const
        {
            static const __m512i idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            return _mm512_permutexvar_pd(idx, data);
        }
        void load(const F64 *p)
        {
            data = _mm512_load_pd(p);
        }
        void loadu(const F64 *p)
        {
            data = _mm512_loadu_pd(p);
        }
        void load1(const F64 *p)
        {
            data = _mm512_set1_pd(*p);
        }
        static F64X8 broadcastFrom1(const F64 *p)
        {
            return _mm512_set1_pd(*p);
        }
        static F64X8 fromMem(const F64 *p)
        {
            return _mm512_load_pd(p);
        }
        static F64X8 fromUMem(const F64 *p)
        {
            return _mm512_loadu_pd(p);
        }
        void store(F64 *p) const
        {
            _mm512_store_pd(p, data);
        }
        void storeu(F64 *p) const
        {
            _mm512_storeu_pd(p, data);
        }
        operator __m512d() const
        {
            return data;
        }
        __m512i toI64X8() const
        {
            constexpr uint64_t mask = (uint64_t(1) << 52) - 1;
            constexpr uint64_t offset = (uint64_t(1) << 10) - 1;
            const __m512i f64bits = _mm512_castpd_si512(data);
            __m512i tail = _mm512_and_si512(f64bits, _mm512_set1_epi64(mask));
            tail = _mm512_or_si512(tail, _mm512_set1_epi64(mask + 1));
            __m512i exp = _mm512_srli_epi64(f64bits, 52);
            exp = _mm512_sub_epi64(_mm512_set1_epi64(offset + 52), exp);
            return _mm512_srlv_epi64(tail, exp);
        }

        void print() const
        {
            F64 arr[8];
            this->storeu(arr);
            std::cout << "[" << arr[0] << "," << arr[1] << "," << arr[2] << "," << arr[3]
                      << "," << arr[4] << "," << arr[5] << "," << arr[6] << "," << arr[7] << "]" << std::endl;
        }

    private:
        __m512d data;
        static constexpr F64 HSQRT2 = 0.70710678118654752440084436210485;
        static constexpr F64 NHSQRT2 = -HSQRT2;
    };

    constexpr Float64X8::F64 Float64X8::HSQRT2;
    constexpr Float64X8::F64 Float64X8::NHSQRT2;

    struct Complex64X8
    {
        using C64X8 = Complex64X8;
        using F64X8 = Float64X8;
        using F64 = double;
        Complex64X8() = default;
        Complex64X8(F64X8 real, F64X8 imag) : real(real), imag(imag) {}
        Complex64X8(const F64 *p) : real(p), imag(p + 4) {}
        Complex64X8(const F64 *p_real, const F64 *p_imag) : real(p_real), imag(p_imag) {}
        C64X8 operator+(const C64X8 &other) const
        {
            return C64X8(real + other.real, imag + other.imag);
        }
        C64X8 operator-(const C64X8 &other) const
        {
            return C64X8(real - other.real, imag - other.imag);
        }
        C64X8 operator*(const F64X8 &other) const
        {
            return C64X8(real * other, imag * other);
        }
        C64X8 mul(const C64X8 &other) const
        {
            const F64X8 ii = imag * other.imag;
            const F64X8 ri = real * other.imag;
            const F64X8 r = F64X8::fmsub(real, other.real, ii);
            const F64X8 i = F64X8::fmadd(imag, other.real, ri);
            return C64X8(r, i);
        }
        C64X8 mulConj(const C64X8 &other) const
        {
            const F64X8 ii = imag * other.imag;
            const F64X8 ri = real * other.imag;
            const F64X8 r = F64X8::fmadd(real, other.real, ii);
            const F64X8 i = F64X8::fmsub(imag, other.real, ri);
            return C64X8(r, i);
        }

        C64X8 reverse() const
        {
            return C64X8(real.reverse(), imag.reverse());
        }
        // exp{i*(theta+k)},k in {0,1,2,3,4,5,6,7}
        static C64X8 omegaSeq0To8(F64 theta, F64 begin = 0)
        {
            F64 real_arr[8] = {cos(begin), cos(theta + begin), cos(2 * theta + begin), cos(3 * theta + begin),
                               cos(4 * theta + begin), cos(5 * theta + begin), cos(6 * theta + begin), cos(7 * theta + begin)};
            F64 imag_arr[8] = {sin(begin), sin(theta + begin), sin(2 * theta + begin), sin(3 * theta + begin),
                               sin(4 * theta + begin), sin(5 * theta + begin), sin(6 * theta + begin), sin(7 * theta + begin)};
            return C64X8(F64X8(real_arr), F64X8(imag_arr));
        }

        template <typename T>
        void load(const T *p, std::false_type)
        {
            this->load(p);
        }
        // From RIRI permutation
        template <typename T>
        void load(const T *p, std::true_type)
        {
            this->load(p);
            *this = this->toRRIIPermu();
        }
        template <typename T>
        void load(const T *p)
        {
            real.load(reinterpret_cast<const F64 *>(p));
            imag.load(reinterpret_cast<const F64 *>(p) + 8);
        }
        template <typename T>
        void loadu(const T *p)
        {
            real.loadu(reinterpret_cast<const F64 *>(p));
            imag.loadu(reinterpret_cast<const F64 *>(p) + 8);
        }
        void load1(const F64 *real_p, const F64 *imag_p)
        {
            real.load1(real_p);
            imag.load1(imag_p);
        }

        template <typename T>
        void store(T *p, std::false_type) const
        {
            this->store(p);
        }
        // To RIRI permutation
        template <typename T>
        void store(T *p, std::true_type) const
        {
            this->toRIRIPermu().store(p);
        }
        template <typename T>
        void store(T *p) const
        {
            real.store(reinterpret_cast<F64 *>(p));
            imag.store(reinterpret_cast<F64 *>(p) + 8);
        }
        template <typename T>
        void storeu(T *p) const
        {
            real.storeu(reinterpret_cast<F64 *>(p));
            imag.storeu(reinterpret_cast<F64 *>(p) + 8);
        }
        C64X8 square() const
        {
            const F64X8 ii = imag * imag;
            const F64X8 ri = real * imag;
            const F64X8 r = F64X8::fmsub(real, real, ii);
            const F64X8 i = ri + ri;
            return C64X8(r, i);
        }
        C64X8 cubeUnit() const
        {
            constexpr uint64_t MAGIC_ONE = 4607182418800017408ull;   // 1
            constexpr uint64_t MAGIC_THREE = 4613937818241073152ull; // 3
            const F64X8 one = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_ONE));
            const F64X8 three = _mm512_castsi512_pd(_mm512_set1_epi64(MAGIC_THREE));
            const F64X8 rr = real * real;
            F64X8 rr4 = rr + rr;
            rr4 = rr4 + rr4;
            const F64X8 r = real * (rr4 - one);
            const F64X8 i = imag * (rr4 - three);
            return C64X8(r, i);
        }
        C64X8 toRIRIPermu() const
        {
            C64X8 res = *this;
            transpose64_2X8(res.real, res.imag);
            return res;
        }
        C64X8 toRRIIPermu() const
        {
            C64X8 res = *this;
            transpose64_8X2(res.real, res.imag);
            return res;
        }
        void print() const
        {
            F64 real_arr[8]{}, imag_arr[8]{};
            real.storeu(real_arr);
            imag.storeu(imag_arr);
            std::cout << "[("
                      << real_arr[0] << ", " << imag_arr[0] << "), ("
                      << real_arr[1] << ", " << imag_arr[1] << "), ("
                      << real_arr[2] << ", " << imag_arr[2] << "), ("
                      << real_arr[3] << ", " << imag_arr[3] << "), ("
                      << real_arr[4] << ", " << imag_arr[4] << "), ("
                      << real_arr[5] << ", " << imag_arr[5] << "), ("
                      << real_arr[6] << ", " << imag_arr[6] << "), ("
                      << real_arr[7] << ", " << imag_arr[7] << ")]"
                      << std::endl;
        }

        C64X8 transToI64(std::false_type) const
        {
            return *this;
        }
        C64X8 transToI64(std::true_type) const
        {
#ifdef __AVX512DQ__
#pragma message("AVX512DQ")
            auto real_i64 = _mm512_cvtpd_epi64(real);
            auto imag_i64 = _mm512_cvtpd_epi64(imag);
            return C64X8(_mm512_castsi512_pd(real_i64), _mm512_castsi512_pd(imag_i64));
#else
            constexpr int64_t F1_2 = 4602678819172646912; // magic::bit_cast<int64_t>(0.5);
            F64X8 F1_2X8 = _mm512_castsi512_pd(_mm512_set1_epi64(F1_2));
            auto real_i64 = (real + F1_2X8).toI64X8();
            auto imag_i64 = (imag + F1_2X8).toI64X8();
            return C64X8(_mm512_castsi512_pd(real_i64), _mm512_castsi512_pd(imag_i64));
#endif
        }

        F64X8 real, imag;
    };

#else
#pragma message("AVX512F is not supported")
#endif
}
#endif