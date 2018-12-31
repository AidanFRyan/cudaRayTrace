/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#if !defined(__CUDA_FP16_HPP__)
#define __CUDA_FP16_HPP__

/* C++11 header for std::move. 
 * In RTC mode, std::move is provided implicitly; don't include the header
*/
#if (__cplusplus >= 201103L) && !defined(__CUDACC_RTC__)
#include <utility>
#endif /* __cplusplus >= 201103L && !defined(__CUDACC_RTC__) */

/* Set up function decorations */
/* Set up function decorations */
#if defined(__CUDACC__)
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__
#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#define __CUDA_HOSTDEVICE__ __host__ __device__
#else /* !defined(__CUDACC__) */
#if defined(__GNUC__) /* || defined(__IBMC__) || defined(__clang__) || defined(__PGI) */
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __attribute__ ((unused))
#else
#define __CUDA_HOSTDEVICE_FP16_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE__
#endif /* defined(__CUDACC_) */

/* Set up structure-alignment attribute */
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if __cplusplus >= 201103L
#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* !(__cplusplus >= 201103L)*/
#if defined(__GNUC__) /* || defined(__IBMC__) || defined(__clang__) || defined(__PGI) */
#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER) /* || defined(__ICC) */
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* __cplusplus >= 201103L */
#endif /* defined(__CUDACC__) */


/* Macros to allow half & half2 to be used by inline assembly */
#define cuhalf_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define cuhalf_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define cuhalf_TO_VUS(var) *(reinterpret_cast<volatile unsigned short *>(&(var)))
#define cuhalf_TO_CVUS(var) *(reinterpret_cast<const volatile unsigned short *>(&(var)))
#define cuhalf2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define cuhalf2_TO_CUI(var) *(reinterpret_cast<const unsigned int *>(&(var)))

/* Type punning macros for host-side implementations */
#if defined(__CUDACC__)
#define __COPY_FLOAT_TO_UI(to, from) ((to) = *(reinterpret_cast<unsigned int *>(&(from))))
#define __COPY_UI_TO_FLOAT(to, from) ((to) = *(reinterpret_cast<float *>(&(from))))
#else
#include <string.h>
#define __COPY_FLOAT_TO_UI(to, from) memcpy(&(to), &(from), sizeof((to)))
#define __COPY_UI_TO_FLOAT(to, from) memcpy(&(to), &(from), sizeof((to)))
#endif

/**
* Types which allow static initialization of "half" and "half2" until
* these become an actual builtin. Note this initialization is as a
* bitfield representation of "half", and not a conversion from short->half.
* Such a representation will be deprecated in a future version of CUDA. 
* (Note these are visible to non-nvcc compilers, including C-only compilation)
*/
typedef struct __CUDA_ALIGN__(2) {
    unsigned short x;
} cuhalf_raw;

typedef struct __CUDA_ALIGN__(4) {
    unsigned short x, y;
} cuhalf2_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(__GNUC__) */

struct __CUDA_ALIGN__(2) cuhalf {
protected:
    unsigned short __x;

public:
#if __cplusplus >= 201103L
    cuhalf() = default;
#else
    __CUDA_HOSTDEVICE__ cuhalf() { }
#endif /* __cplusplus >= 201103L */

    /* Convert to/from cuhalf_raw */
    __CUDA_HOSTDEVICE__ cuhalf(const cuhalf_raw &hr) : __x(hr.x) { }
    __CUDA_HOSTDEVICE__ cuhalf &operator=(const cuhalf_raw &hr) { __x = hr.x; return *this; }
    __CUDA_HOSTDEVICE__ volatile cuhalf &operator=(const cuhalf_raw &hr) volatile { __x = hr.x; return *this; }
    __CUDA_HOSTDEVICE__ volatile cuhalf &operator=(const volatile cuhalf_raw &hr) volatile { __x = hr.x; return *this; }
    __CUDA_HOSTDEVICE__ operator cuhalf_raw() const { cuhalf_raw ret; ret.x = __x; return ret; }
    __CUDA_HOSTDEVICE__ operator volatile cuhalf_raw() const volatile { cuhalf_raw ret; ret.x = __x; return ret; }

#if !defined(__CUDA_NO_HALF_CONVERSIONS__)

    /* Construct from float/double */
    __CUDA_HOSTDEVICE__ cuhalf(float f) { __x = __float2half(f).__x;  printf("float: %f\n", f);}
    __CUDA_HOSTDEVICE__ cuhalf(double f) { __x = __float2half(static_cast<float>(f)).__x;  }

    __CUDA_HOSTDEVICE__ operator float() const { return cuhalf2float(*this); }
    __CUDA_HOSTDEVICE__ cuhalf &operator=(float f) { __x = __float2half(f).__x; return *this; }

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    __CUDA_HOSTDEVICE__ cuhalf &operator=(double f) { __x = __float2half(static_cast<float>(f)).__x; return *this; }

/* Member functions only available to nvcc compilation so far */
#if defined(__CUDACC__)
    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */
    __device__ cuhalf(short val) { __x = __short2half_rn(val).__x;  }
    __device__ cuhalf(unsigned short val) { __x = __ushort2half_rn(val).__x;  }
    __device__ cuhalf(int val) { __x = __int2half_rn(val).__x;  }
    __device__ cuhalf(unsigned int val) { __x = __uint2half_rn(val).__x;  }
    __device__ cuhalf(long long val) { __x = __ll2half_rn(val).__x;  }
    __device__ cuhalf(unsigned long long val) { __x = __ull2half_rn(val).__x; }

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */
    __device__ operator short() const { return cuhalf2short_rn(*this); }
    __device__ cuhalf &operator=(short val) { __x = __short2half_rn(val).__x; return *this; }

    __device__ operator unsigned short() const { return cuhalf2ushort_rn(*this); }
    __device__ cuhalf &operator=(unsigned short val) { __x = __ushort2half_rn(val).__x; return *this; }

    __device__ operator int() const { return cuhalf2int_rn(*this); }
    __device__ cuhalf &operator=(int val) { __x = __int2half_rn(val).__x; printf("int %d\n", val); return *this; }

    __device__ operator unsigned int() const { return cuhalf2uint_rn(*this); }
    __device__ cuhalf &operator=(unsigned int val) { __x = __uint2half_rn(val).__x; return *this; }

    __device__ operator long long() const { return cuhalf2ll_rn(*this); }
    __device__ cuhalf &operator=(long long val) { __x = __ll2half_rn(val).__x; return *this; }

    __device__ operator unsigned long long() const { return cuhalf2ull_rn(*this); }
    __device__ cuhalf &operator=(unsigned long long val) { __x = __ull2half_rn(val).__x; return *this; }

    /* Boolean conversion - note both 0 and -0 must return false */
    __device__ operator bool() const { return (__x & 0x7FFF) != 0; }
#endif /* defined(__CUDACC__) */
#endif /* !defined(__CUDA_NO_HALF_CONVERSIONS__) */
};

/* Global-space operator functions are only available to nvcc compilation */
#if defined(__CUDACC__)

/* Arithmetic FP16 operations only supported on arch >= 5.3 */
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#if !defined(__CUDA_NO_HALF_OPERATORS__)
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ cuhalf operator+(const cuhalf &lh, const cuhalf &rh) { return __hadd(lh, rh); }
__device__ __forceinline__ cuhalf operator+(const cuhalf& h, const float& f){ cuhalf fl = f; return __hadd(h, fl);}
__device__ __forceinline__ cuhalf operator+(const float& f, const cuhalf& h){ cuhalf fl = f; return __hadd(h, fl);}
__device__ __forceinline__ cuhalf operator-(const cuhalf &lh, const cuhalf &rh) { return __hsub(lh, rh); }
__device__ __forceinline__ cuhalf operator-(const cuhalf& h, const float& f){ cuhalf fl = f; return __hsub(h, fl);}
__device__ __forceinline__ cuhalf operator-(const float& f, const cuhalf& h){ cuhalf fl = f; return __hsub(h, fl);}
__device__ __forceinline__ cuhalf operator*(const cuhalf &lh, const cuhalf &rh) { return __hmul(lh, rh); }
__device__ __forceinline__ cuhalf operator*(const cuhalf& h, const float& f){ cuhalf fl = f; return __hmul(h, fl);}
__device__ __forceinline__ cuhalf operator*(const float& f, const cuhalf& h){ cuhalf fl = f; return __hmul(h, fl);}
__device__ __forceinline__ cuhalf operator/(const cuhalf &lh, const cuhalf &rh) { return __hdiv(lh, rh); }
__device__ __forceinline__ cuhalf operator/(const cuhalf& h, const float& f){ cuhalf fl = f; return __hdiv(h, fl);}
__device__ __forceinline__ cuhalf operator/(const float& f, const cuhalf& h){ cuhalf fl = f; return __hdiv(h, fl);}

__device__ __forceinline__ cuhalf &operator+=(cuhalf &lh, const cuhalf &rh) { lh = __hadd(lh, rh); return lh; }
__device__ __forceinline__ cuhalf &operator-=(cuhalf &lh, const cuhalf &rh) { lh = __hsub(lh, rh); return lh; }
__device__ __forceinline__ cuhalf &operator*=(cuhalf &lh, const cuhalf &rh) { lh = __hmul(lh, rh); return lh; }
__device__ __forceinline__ cuhalf &operator/=(cuhalf &lh, const cuhalf &rh) { lh = __hdiv(lh, rh); return lh; }

/* Note for increment and decrement we use the raw value 0x3C00 equating to half(1.0f), to avoid the extra conversion */
__device__ __forceinline__ cuhalf &operator++(cuhalf &h)      { cuhalf_raw one; one.x = 0x3C00; h += one; return h; }
__device__ __forceinline__ cuhalf &operator--(cuhalf &h)      { cuhalf_raw one; one.x = 0x3C00; h -= one; return h; }
__device__ __forceinline__ cuhalf  operator++(cuhalf &h, int) { cuhalf ret = h; cuhalf_raw one; one.x = 0x3C00; h += one; return ret; }
__device__ __forceinline__ cuhalf  operator--(cuhalf &h, int) { cuhalf ret = h; cuhalf_raw one; one.x = 0x3C00; h -= one; return ret; }

/* Unary plus and inverse operators */
__device__ __forceinline__ cuhalf operator+(const cuhalf &h) { return h; }
__device__ __forceinline__ cuhalf operator-(const cuhalf &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const cuhalf &lh, const cuhalf &rh) { return __heq(lh, rh); }
__device__ __forceinline__ bool operator==(const cuhalf &lh, const float &r) { cuhalf rh = r; return __heq(lh, rh); }
__device__ __forceinline__ bool operator==(const float &r, const cuhalf &lh) { cuhalf rh = r; return __heq(lh, rh); }
__device__ __forceinline__ bool operator!=(const cuhalf &lh, const cuhalf &rh) { return __hne(lh, rh); }
__device__ __forceinline__ bool operator!=(const cuhalf &lh, const float &r) { cuhalf rh = r; return __hne(lh, rh); }
__device__ __forceinline__ bool operator!=(const float &r, const cuhalf &lh) { cuhalf rh = r; return __hne(lh, rh); }
__device__ __forceinline__ bool operator> (const cuhalf &lh, const cuhalf &rh) { return __hgt(lh, rh); }
__device__ __forceinline__ bool operator> (const cuhalf &lh, const float &r) { cuhalf rh = r; return __hgt(lh, rh); }
__device__ __forceinline__ bool operator> (const float &r, const cuhalf &lh) { cuhalf rh = r; return __hgt(lh, rh); }
__device__ __forceinline__ bool operator< (const cuhalf &lh, const cuhalf &rh) { return __hlt(lh, rh); }
__device__ __forceinline__ bool operator< (const cuhalf &lh, const float &r) { cuhalf rh = r; return __hlt(lh, rh); }
__device__ __forceinline__ bool operator< (const float &r, const cuhalf &lh) { cuhalf rh = r; return __hlt(lh, rh); }
__device__ __forceinline__ bool operator>=(const cuhalf &lh, const cuhalf &rh) { return __hge(lh, rh); }
__device__ __forceinline__ bool operator>=(const cuhalf &lh, const float &r) { cuhalf rh = r; return __hge(lh, rh); }
__device__ __forceinline__ bool operator>=(const float &r, const cuhalf &lh) { cuhalf rh = r; return __hge(lh, rh); }
__device__ __forceinline__ bool operator<=(const cuhalf &lh, const cuhalf &rh) { return __hle(lh, rh); }
__device__ __forceinline__ bool operator<=(const cuhalf &lh, const float &r) { cuhalf rh = r; return __hle(lh, rh); }
__device__ __forceinline__ bool operator<=(const float &r, const cuhalf &lh) { cuhalf rh = r; return __hle(lh, rh); }
#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
#endif /* defined(__CUDACC__) */

/* cuhalf2 is visible to non-nvcc host compilers */
struct __CUDA_ALIGN__(4) cuhalf2 {
    cuhalf x, y;

    // All construct/copy/assign/move
public:
#if __cplusplus >= 201103L
    cuhalf2() = default;
    __CUDA_HOSTDEVICE__ cuhalf2(cuhalf2 &&src) { cuhalf2_TO_UI(*this) = std::move(cuhalf2_TO_CUI(src)); }
    __CUDA_HOSTDEVICE__ cuhalf2 &operator=(cuhalf2 &&src) { cuhalf2_TO_UI(*this) = std::move(cuhalf2_TO_CUI(src)); return *this; }
#else
    __CUDA_HOSTDEVICE__ cuhalf2() { }
#endif /* __cplusplus >= 201103L */
    __CUDA_HOSTDEVICE__ cuhalf2(const cuhalf &a, const cuhalf &b) : x(a), y(b) { }
    __CUDA_HOSTDEVICE__ cuhalf2(const cuhalf2 &src) { cuhalf2_TO_UI(*this) = cuhalf2_TO_CUI(src); }
    __CUDA_HOSTDEVICE__ cuhalf2 &operator=(const cuhalf2 &src) { cuhalf2_TO_UI(*this) = cuhalf2_TO_CUI(src); return *this; }

    /* Convert to/from cuhalf2_raw */
    __CUDA_HOSTDEVICE__ cuhalf2(const cuhalf2_raw &h2r ) { cuhalf2_TO_UI(*this) = cuhalf2_TO_CUI(h2r); }
    __CUDA_HOSTDEVICE__ cuhalf2 &operator=(const cuhalf2_raw &h2r) { cuhalf2_TO_UI(*this) = cuhalf2_TO_CUI(h2r); return *this; }
    __CUDA_HOSTDEVICE__ operator cuhalf2_raw() const { cuhalf2_raw ret; cuhalf2_TO_UI(ret) = cuhalf2_TO_CUI(*this); return ret; }
};

/* Global-space operator functions are only available to nvcc compilation */
#if defined(__CUDACC__)

/* Arithmetic FP16x2 operations only supported on arch >= 5.3 */
#if (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)) && !defined(__CUDA_NO_HALF2_OPERATORS__)

__device__ __forceinline__ cuhalf2 operator+(const cuhalf2 &lh, const cuhalf2 &rh) { return __hadd2(lh, rh); }
__device__ __forceinline__ cuhalf2 operator-(const cuhalf2 &lh, const cuhalf2 &rh) { return __hsub2(lh, rh); }
__device__ __forceinline__ cuhalf2 operator*(const cuhalf2 &lh, const cuhalf2 &rh) { return __hmul2(lh, rh); }
__device__ __forceinline__ cuhalf2 operator/(const cuhalf2 &lh, const cuhalf2 &rh) { return __h2div(lh, rh); }

__device__ __forceinline__ cuhalf2& operator+=(cuhalf2 &lh, const cuhalf2 &rh) { lh = __hadd2(lh, rh); return lh; }
__device__ __forceinline__ cuhalf2& operator-=(cuhalf2 &lh, const cuhalf2 &rh) { lh = __hsub2(lh, rh); return lh; }
__device__ __forceinline__ cuhalf2& operator*=(cuhalf2 &lh, const cuhalf2 &rh) { lh = __hmul2(lh, rh); return lh; }
__device__ __forceinline__ cuhalf2& operator/=(cuhalf2 &lh, const cuhalf2 &rh) { lh = __h2div(lh, rh); return lh; }

__device__ __forceinline__ cuhalf2 &operator++(cuhalf2 &h)      { cuhalf2_raw one; one.x = 0x3C00; one.y = 0x3C00; h = __hadd2(h, one); return h; }
__device__ __forceinline__ cuhalf2 &operator--(cuhalf2 &h)      { cuhalf2_raw one; one.x = 0x3C00; one.y = 0x3C00; h = __hsub2(h, one); return h; }
__device__ __forceinline__ cuhalf2  operator++(cuhalf2 &h, int) { cuhalf2 ret = h; cuhalf2_raw one; one.x = 0x3C00; one.y = 0x3C00; h = __hadd2(h, one); return ret; }
__device__ __forceinline__ cuhalf2  operator--(cuhalf2 &h, int) { cuhalf2 ret = h; cuhalf2_raw one; one.x = 0x3C00; one.y = 0x3C00; h = __hsub2(h, one); return ret; }

__device__ __forceinline__ cuhalf2 operator+(const cuhalf2 &h) { return h; }
__device__ __forceinline__ cuhalf2 operator-(const cuhalf2 &h) { return __hneg2(h); }

__device__ __forceinline__ bool operator==(const cuhalf2 &lh, const cuhalf2 &rh) { return __hbeq2(lh, rh); }
__device__ __forceinline__ bool operator!=(const cuhalf2 &lh, const cuhalf2 &rh) { return __hbne2(lh, rh); }
__device__ __forceinline__ bool operator>(const cuhalf2 &lh, const cuhalf2 &rh) { return __hbgt2(lh, rh); }
__device__ __forceinline__ bool operator<(const cuhalf2 &lh, const cuhalf2 &rh) { return __hblt2(lh, rh); }
__device__ __forceinline__ bool operator>=(const cuhalf2 &lh, const cuhalf2 &rh) { return __hbge2(lh, rh); }
__device__ __forceinline__ bool operator<=(const cuhalf2 &lh, const cuhalf2 &rh) { return __hble2(lh, rh); }

#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
#endif /* defined(__CUDACC__) */

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( pop )
#endif

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __CUDA_HOSTDEVICE__
#undef __CUDA_ALIGN__

#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
static unsigned short __internal_float2half(float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int x, u, shift, exponent, mantissa;
    memcpy(&x, &f, sizeof(f));
    u = (x & 0x7fffffffU);
    sign = ((x >> 16) & 0x8000U);
    // NaN/+Inf/-Inf
    if (u >= 0x7f800000U) {
        remainder = 0;
        return static_cast<unsigned short>((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
    }
    // Overflows
    if (u > 0x477fefffU) {
        remainder = 0x80000000U;
        return static_cast<unsigned short>(sign | 0x7bffU);
    }
    // Normal numbers
    if (u >= 0x38800000U) {
        remainder = u << 19;
        u -= 0x38000000U;
        return static_cast<unsigned short>(sign | (u >> 13));
    }
    // +0/-0
    if (u < 0x33000001U) {
        remainder = u;
        return static_cast<unsigned short>(sign);
    }
    // Denormal numbers
    exponent = u >> 23;
    mantissa = (u & 0x7fffffU);
    shift = 0x7eU - exponent;
    mantissa |= 0x800000U;
    remainder = mantissa << (32 - shift);
    return static_cast<unsigned short>(sign | (mantissa >> shift));
}
#endif  /* #if !defined(__CUDACC_RTC__) */

__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf __float2half(const float f)
{
    cuhalf val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(cuhalf_TO_US(val)) : "f"(f));
#else
    cuhalf_raw r;
    unsigned int sign, remainder;
    r.x = __internal_float2half(f, sign, remainder);
    if (remainder > 0x80000000U || (remainder == 0x80000000U && (r.x & 0x1)))
        r.x++;
    val = r;
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf __float2half_rn(const float f)
{
    cuhalf val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(cuhalf_TO_US(val)) : "f"(f));
#else
    cuhalf_raw r;
    unsigned int sign, remainder;
    r.x = __internal_float2half(f, sign, remainder);
    if (remainder > 0x80000000U || (remainder == 0x80000000U && (r.x & 0x1)))
        r.x++;
    val = r;
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf __float2half_rz(const float f)
{
    cuhalf val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.rz.f16.f32 %0, %1;}\n" : "=h"(cuhalf_TO_US(val)) : "f"(f));
#else
    cuhalf_raw r;
    unsigned int sign, remainder;
    r.x = __internal_float2half(f, sign, remainder);
    val = r;
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf __float2half_rd(const float f)
{
    cuhalf val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.rm.f16.f32 %0, %1;}\n" : "=h"(cuhalf_TO_US(val)) : "f"(f));
#else
    cuhalf_raw r;
    unsigned int sign, remainder;
    r.x = __internal_float2half(f, sign, remainder);
    if (remainder && sign)
        r.x++;
    val = r;
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf __float2half_ru(const float f)
{
    cuhalf val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.rp.f16.f32 %0, %1;}\n" : "=h"(cuhalf_TO_US(val)) : "f"(f));
#else
    cuhalf_raw r;
    unsigned int sign, remainder;
    r.x = __internal_float2half(f, sign, remainder);
    if (remainder && !sign)
        r.x++;
    val = r;
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf2 __float2half2_rn(const float f)
{
    cuhalf2 val;
#if defined(__CUDA_ARCH__)
    asm("{.reg .f16 low;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(cuhalf2_TO_UI(val)) : "f"(f));
#else
    val = cuhalf2(__float2half_rn(f), __float2half_rn(f));
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf2 __floats2half2_rn(const float f1, const float f2)
{
    cuhalf2 val;
#if defined(__CUDA_ARCH__)
    asm("{.reg .f16 low,high;\n"
        "  cvt.rn.f16.f32 low, %1;\n"
        "  cvt.rn.f16.f32 high, %2;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "f"(f1), "f"(f2));
#else
    val = cuhalf2(__float2half_rn(f1), __float2half_rn(f2));
#endif
    return val;
}

#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
static float __internal_half2float(unsigned short h)
{
    unsigned int sign = ((h >> 15) & 1);
    unsigned int exponent = ((h >> 10) & 0x1f);
    unsigned int mantissa = ((h & 0x3ff) << 13);
    float f;
    if (exponent == 0x1fU) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffffU) : 0);
        exponent = 0xffU;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    unsigned int u = ((sign << 31) | (exponent << 23) | mantissa);
    memcpy(&f, &u, sizeof(u));
    return f;
}
#endif  /* !defined(__CUDACC_RTC__) */

__CUDA_HOSTDEVICE_FP16_DECL__ float cuhalf2float(const cuhalf h)
{
    float val;
#if defined(__CUDA_ARCH__)
    asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(cuhalf_TO_CUS(h)));
#else
    val = __internal_half2float(static_cast<cuhalf_raw>(h).x);
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const cuhalf2 l)
{
    float val;
#if defined(__CUDA_ARCH__)
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(val) : "r"(cuhalf2_TO_CUI(l)));
#else
    val = __internal_half2float(static_cast<cuhalf2_raw>(l).x);
#endif
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const cuhalf2 l)
{
    float val;
#if defined(__CUDA_ARCH__)
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(val) : "r"(cuhalf2_TO_CUI(l)));
#else
    val = __internal_half2float(static_cast<cuhalf2_raw>(l).y);
#endif
    return val;
}

/* Intrinsic functions only available to nvcc compilers */
#if defined(__CUDACC__)

/* CUDA vector-types compatible vector creation function (note returns cuhalf2, not half2) */
__VECTOR_FUNCTIONS_DECL__ cuhalf2 make_half2(cuhalf x, cuhalf y)
{
    cuhalf2 t; t.x = x; t.y = y; return t;
}
#undef __VECTOR_FUNCTIONS_DECL__


/* Definitions of intrinsics */
__CUDA_HOSTDEVICE_FP16_DECL__ cuhalf2 __float22half2_rn(const float2 f)
{
    cuhalf2 val = __floats2half2_rn(f.x, f.y);
    return val;
}
__CUDA_HOSTDEVICE_FP16_DECL__ float2 cuhalf22float2(const cuhalf2 l)
{
    float hi_float;
    float lo_float;
#if defined(__CUDA_ARCH__)
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(cuhalf2_TO_CUI(l)));

    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high},%1;\n"
        "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(cuhalf2_TO_CUI(l)));
#else
    lo_float = __internal_half2float(((cuhalf2_raw)l).x);
    hi_float = __internal_half2float(((cuhalf2_raw)l).y);
#endif
    return make_float2(lo_float, hi_float);
}
__CUDA_FP16_DECL__ int cuhalf2int_rn(cuhalf h)
{
    int i;
    asm("cvt.rni.s32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int cuhalf2int_rz(cuhalf h)
{
    int i;
    asm("cvt.rzi.s32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int cuhalf2int_rd(cuhalf h)
{
    int i;
    asm("cvt.rmi.s32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ int cuhalf2int_ru(cuhalf h)
{
    int i;
    asm("cvt.rpi.s32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __int2half_rn(int i)
{
    cuhalf h;
    asm("cvt.rn.f16.s32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __int2half_rz(int i)
{
    cuhalf h;
    asm("cvt.rz.f16.s32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __int2half_rd(int i)
{
    cuhalf h;
    asm("cvt.rm.f16.s32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __int2half_ru(int i)
{
    cuhalf h;
    asm("cvt.rp.f16.s32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}

__CUDA_FP16_DECL__ short int cuhalf2short_rn(cuhalf h)
{
    short int i;
    asm("cvt.rni.s16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int cuhalf2short_rz(cuhalf h)
{
    short int i;
    asm("cvt.rzi.s16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int cuhalf2short_rd(cuhalf h)
{
    short int i;
    asm("cvt.rmi.s16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ short int cuhalf2short_ru(cuhalf h)
{
    short int i;
    asm("cvt.rpi.s16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __short2half_rn(short int i)
{
    cuhalf h;
    asm("cvt.rn.f16.s16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __short2half_rz(short int i)
{
    cuhalf h;
    asm("cvt.rz.f16.s16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __short2half_rd(short int i)
{
    cuhalf h;
    asm("cvt.rm.f16.s16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __short2half_ru(short int i)
{
    cuhalf h;
    asm("cvt.rp.f16.s16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned int cuhalf2uint_rn(cuhalf h)
{
    unsigned int i;
    asm("cvt.rni.u32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int cuhalf2uint_rz(cuhalf h)
{
    unsigned int i;
    asm("cvt.rzi.u32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int cuhalf2uint_rd(cuhalf h)
{
    unsigned int i;
    asm("cvt.rmi.u32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned int cuhalf2uint_ru(cuhalf h)
{
    unsigned int i;
    asm("cvt.rpi.u32.f16 %0, %1;" : "=r"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __uint2half_rn(unsigned int i)
{
    cuhalf h;
    asm("cvt.rn.f16.u32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __uint2half_rz(unsigned int i)
{
    cuhalf h;
    asm("cvt.rz.f16.u32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __uint2half_rd(unsigned int i)
{
    cuhalf h;
    asm("cvt.rm.f16.u32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __uint2half_ru(unsigned int i)
{
    cuhalf h;
    asm("cvt.rp.f16.u32 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "r"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned short int cuhalf2ushort_rn(cuhalf h)
{
    unsigned short int i;
    asm("cvt.rni.u16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int cuhalf2ushort_rz(cuhalf h)
{
    unsigned short int i;
    asm("cvt.rzi.u16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int cuhalf2ushort_rd(cuhalf h)
{
    unsigned short int i;
    asm("cvt.rmi.u16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned short int cuhalf2ushort_ru(cuhalf h)
{
    unsigned short int i;
    asm("cvt.rpi.u16.f16 %0, %1;" : "=h"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __ushort2half_rn(unsigned short int i)
{
    cuhalf h;
    asm("cvt.rn.f16.u16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ushort2half_rz(unsigned short int i)
{
    cuhalf h;
    asm("cvt.rz.f16.u16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ushort2half_rd(unsigned short int i)
{
    cuhalf h;
    asm("cvt.rm.f16.u16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ushort2half_ru(unsigned short int i)
{
    cuhalf h;
    asm("cvt.rp.f16.u16 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "h"(i));
    return h;
}

__CUDA_FP16_DECL__ unsigned long long int cuhalf2ull_rn(cuhalf h)
{
    unsigned long long int i;
    asm("cvt.rni.u64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int cuhalf2ull_rz(cuhalf h)
{
    unsigned long long int i;
    asm("cvt.rzi.u64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int cuhalf2ull_rd(cuhalf h)
{
    unsigned long long int i;
    asm("cvt.rmi.u64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ unsigned long long int cuhalf2ull_ru(cuhalf h)
{
    unsigned long long int i;
    asm("cvt.rpi.u64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __ull2half_rn(unsigned long long int i)
{
    cuhalf h;
    asm("cvt.rn.f16.u64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ull2half_rz(unsigned long long int i)
{
    cuhalf h;
    asm("cvt.rz.f16.u64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ull2half_rd(unsigned long long int i)
{
    cuhalf h;
    asm("cvt.rm.f16.u64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ull2half_ru(unsigned long long int i)
{
    cuhalf h;
    asm("cvt.rp.f16.u64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}

__CUDA_FP16_DECL__ long long int cuhalf2ll_rn(cuhalf h)
{
    long long int i;
    asm("cvt.rni.s64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int cuhalf2ll_rz(cuhalf h)
{
    long long int i;
    asm("cvt.rzi.s64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int cuhalf2ll_rd(cuhalf h)
{
    long long int i;
    asm("cvt.rmi.s64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ long long int cuhalf2ll_ru(cuhalf h)
{
    long long int i;
    asm("cvt.rpi.s64.f16 %0, %1;" : "=l"(i) : "h"(cuhalf_TO_US(h)));
    return i;
}
__CUDA_FP16_DECL__ cuhalf __ll2half_rn(long long int i)
{
    cuhalf h;
    asm("cvt.rn.f16.s64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ll2half_rz(long long int i)
{
    cuhalf h;
    asm("cvt.rz.f16.s64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ll2half_rd(long long int i)
{
    cuhalf h;
    asm("cvt.rm.f16.s64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ll2half_ru(long long int i)
{
    cuhalf h;
    asm("cvt.rp.f16.s64 %0, %1;" : "=h"(cuhalf_TO_US(h)) : "l"(i));
    return h;
}

__CUDA_FP16_DECL__ cuhalf htrunc(const cuhalf h)
{
    cuhalf r;
    asm("cvt.rzi.f16.f16 %0, %1;" : "=h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf hceil(const cuhalf h)
{
    cuhalf r;
    asm("cvt.rpi.f16.f16 %0, %1;" : "=h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf hfloor(const cuhalf h)
{
    cuhalf r;
    asm("cvt.rmi.f16.f16 %0, %1;" : "=h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(h)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf hrint(const cuhalf h)
{
    cuhalf r;
    asm("cvt.rni.f16.f16 %0, %1;" : "=h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(h)));
    return r;
}

__CUDA_FP16_DECL__ cuhalf2 h2trunc(const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rzi.f16.f16 low, low;\n"
        "  cvt.rzi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2ceil(const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rpi.f16.f16 low, low;\n"
        "  cvt.rpi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2floor(const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rmi.f16.f16 low, low;\n"
        "  cvt.rmi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2rint(const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rni.f16.f16 low, low;\n"
        "  cvt.rni.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 __lows2half2(const cuhalf2 l, const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {alow,blow};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(l)), "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 __highs2half2(const cuhalf2 l, const cuhalf2 h)
{
    cuhalf2 val;
    asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
        "  mov.b32 {alow,ahigh}, %1;\n"
        "  mov.b32 {blow,bhigh}, %2;\n"
        "  mov.b32 %0, {ahigh,bhigh};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(l)), "r"(cuhalf2_TO_CUI(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf __low2half(const cuhalf2 h)
{
    cuhalf ret;
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, low;}" : "=h"(cuhalf_TO_US(ret)) : "r"(cuhalf2_TO_CUI(h)));
    return ret;
}
__CUDA_FP16_DECL__ int __hisinf(const cuhalf a)
{
    if (cuhalf_TO_CUS(a) == 0xFC00)
        return -1;
    if (cuhalf_TO_CUS(a) == 0x7C00)
        return 1;
    return 0;
}
__CUDA_FP16_DECL__ cuhalf2 __low2half2(const cuhalf2 l)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {low,low};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 __high2half2(const cuhalf2 l)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,high};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(l)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf __high2half(const cuhalf2 h)
{
    cuhalf ret;
    asm("{.reg .f16 low,high;\n"
        " mov.b32 {low,high}, %1;\n"
        " mov.b16 %0, high;}" : "=h"(cuhalf_TO_US(ret)) : "r"(cuhalf2_TO_CUI(h)));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf2 __halves2half2(const cuhalf l, const cuhalf h)
{
    cuhalf2 val;
    asm("{  mov.b32 %0, {%1,%2};}\n"
        : "=r"(cuhalf2_TO_UI(val)) : "h"(cuhalf_TO_CUS(l)), "h"(cuhalf_TO_CUS(h)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 cuhalf2half2(const cuhalf lh)
{
    cuhalf2 val;
    asm("{  mov.b32 %0, {%1,%1};}\n"
        : "=r"(cuhalf2_TO_UI(val)) : "h"(cuhalf_TO_CUS(lh)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 __lowhigh2highlow(const cuhalf2 lh)
{
    cuhalf2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  mov.b32 %0, {high,low};}\n" : "=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(lh)));
    return val;
}
__CUDA_FP16_DECL__ short int cuhalf_as_short(const cuhalf h)
{
    return (short int)cuhalf_TO_CUS(h);
}
__CUDA_FP16_DECL__ unsigned short int cuhalf_as_ushort(const cuhalf h)
{
    return cuhalf_TO_CUS(h);
}
__CUDA_FP16_DECL__ cuhalf __short_as_half(const short int i)
{
    cuhalf h;
    cuhalf_TO_US(h) = (unsigned short int)i;
    return h;
}
__CUDA_FP16_DECL__ cuhalf __ushort_as_half(const unsigned short int i)
{
    cuhalf h;
    cuhalf_TO_US(h) = i;
    return h;
}

#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
/******************************************************************************
*                           cuhalf, cuhalf2 warp shuffle                     *
******************************************************************************/
#define __SHUFFLE_HALF2_MACRO(name) do {\
   cuhalf2 r; \
   asm("{"#name" %0,%1,%2,%3;\n}" \
       :"=r"(cuhalf2_TO_UI(r)): "r"(cuhalf2_TO_CUI(var)), "r"(delta), "r"(c)); \
   return r; \
} while(0)

#define __SHUFFLE_SYNC_HALF2_MACRO(name) do {\
   cuhalf2 r; \
   asm("{"#name" %0,%1,%2,%3,%4;\n}" \
       :"=r"(cuhalf2_TO_UI(r)): "r"(cuhalf2_TO_CUI(var)), "r"(delta), "r"(c), "r"(mask)); \
   return r; \
} while(0)

__CUDA_FP16_DECL__ cuhalf2 __shfl(cuhalf2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.idx.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_up(cuhalf2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = (warpSize - width) << 8;
    __SHUFFLE_HALF2_MACRO(shfl.up.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_down(cuhalf2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.down.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_xor(cuhalf2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_HALF2_MACRO(shfl.bfly.b32);
}

__CUDA_FP16_DECL__ cuhalf2 __shfl_sync(unsigned mask, cuhalf2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.idx.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_up_sync(unsigned mask, cuhalf2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = (warpSize - width) << 8;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.up.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_down_sync(unsigned mask, cuhalf2 var, unsigned int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.down.b32);
}
__CUDA_FP16_DECL__ cuhalf2 __shfl_xor_sync(unsigned mask, cuhalf2 var, int delta, int width)
{
    int warpSize;
    asm("{mov.u32 %0, WARP_SZ;\n}" : "=r"(warpSize));
    int c = ((warpSize - width) << 8) | 0x1f;
    __SHUFFLE_SYNC_HALF2_MACRO(shfl.sync.bfly.b32);
}

#undef __SHUFFLE_HALF2_MACRO
#undef __SHUFFLE_SYNC_HALF2_MACRO

__CUDA_FP16_DECL__ cuhalf __shfl(cuhalf var, int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_up(cuhalf var, unsigned int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_up(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_down(cuhalf var, unsigned int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_down(temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_xor(cuhalf var, int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_xor(temp1, delta, width);
    return __low2half(temp2);
}

__CUDA_FP16_DECL__ cuhalf __shfl_sync(unsigned mask, cuhalf var, int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_up_sync(unsigned mask, cuhalf var, unsigned int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_down_sync(unsigned mask, cuhalf var, unsigned int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__CUDA_FP16_DECL__ cuhalf __shfl_xor_sync(unsigned mask, cuhalf var, int delta, int width)
{
    cuhalf2 temp1 = __halves2half2(var, var);
    cuhalf2 temp2 = __shfl_xor_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}

#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)*/
/******************************************************************************
*               cuhalf and cuhalf2 __ldg,__ldcg,__ldca,__ldcs                *
******************************************************************************/

#if defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
__CUDA_FP16_DECL__ cuhalf2 __ldg(const  cuhalf2 *ptr)
{
    cuhalf2 ret;
    asm ("ld.global.nc.b32 %0, [%1];"  : "=r"(cuhalf2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf __ldg(const cuhalf *ptr)
{
    cuhalf ret;
    asm ("ld.global.nc.b16 %0, [%1];"  : "=h"(cuhalf_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf2 __ldcg(const  cuhalf2 *ptr)
{
    cuhalf2 ret;
    asm ("ld.global.cg.b32 %0, [%1];"  : "=r"(cuhalf2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf __ldcg(const cuhalf *ptr)
{
    cuhalf ret;
    asm ("ld.global.cg.b16 %0, [%1];"  : "=h"(cuhalf_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf2 __ldca(const  cuhalf2 *ptr)
{
    cuhalf2 ret;
    asm ("ld.global.ca.b32 %0, [%1];"  : "=r"(cuhalf2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf __ldca(const cuhalf *ptr)
{
    cuhalf ret;
    asm ("ld.global.ca.b16 %0, [%1];"  : "=h"(cuhalf_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf2 __ldcs(const  cuhalf2 *ptr)
{
    cuhalf2 ret;
    asm ("ld.global.cs.b32 %0, [%1];"  : "=r"(cuhalf2_TO_UI(ret)) : __LDG_PTR(ptr));
    return ret;
}
__CUDA_FP16_DECL__ cuhalf __ldcs(const cuhalf *ptr)
{
    cuhalf ret;
    asm ("ld.global.cs.b16 %0, [%1];"  : "=h"(cuhalf_TO_US(ret)) : __LDG_PTR(ptr));
    return ret;
}
#undef __LDG_PTR
#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
/******************************************************************************
*                             cuhalf2 comparison                             *
******************************************************************************/
#define __COMPARISON_OP_HALF2_MACRO(name) do {\
   cuhalf2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)),"r"(cuhalf2_TO_CUI(b))); \
   return val; \
} while(0)
__CUDA_FP16_DECL__ cuhalf2 __heq2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ cuhalf2 __hne2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ cuhalf2 __hle2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ cuhalf2 __hge2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ cuhalf2 __hlt2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ cuhalf2 __hgt2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ cuhalf2 __hequ2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ cuhalf2 __hneu2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ cuhalf2 __hleu2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ cuhalf2 __hgeu2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ cuhalf2 __hltu2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ cuhalf2 __hgtu2(const cuhalf2 a, const cuhalf2 b)
{
    __COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef __COMPARISON_OP_HALF2_MACRO
#define __BOOL_COMPARISON_OP_HALF2_MACRO(name) do {\
   cuhalf2 val; \
   asm( "{ "#name".f16x2.f16x2 %0,%1,%2;\n}" \
        :"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)),"r"(cuhalf2_TO_CUI(b))); \
   if (cuhalf2_TO_CUI(val) == 0x3C003C00) \
      return true; \
   else  \
      return false; \
} while(0)
__CUDA_FP16_DECL__ bool __hbeq2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.eq);
}
__CUDA_FP16_DECL__ bool __hbne2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ne);
}
__CUDA_FP16_DECL__ bool __hble2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.le);
}
__CUDA_FP16_DECL__ bool __hbge2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ge);
}
__CUDA_FP16_DECL__ bool __hblt2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.lt);
}
__CUDA_FP16_DECL__ bool __hbgt2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.gt);
}
__CUDA_FP16_DECL__ bool __hbequ2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.equ);
}
__CUDA_FP16_DECL__ bool __hbneu2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.neu);
}
__CUDA_FP16_DECL__ bool __hbleu2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.leu);
}
__CUDA_FP16_DECL__ bool __hbgeu2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.geu);
}
__CUDA_FP16_DECL__ bool __hbltu2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.ltu);
}
__CUDA_FP16_DECL__ bool __hbgtu2(const cuhalf2 a, const cuhalf2 b)
{
    __BOOL_COMPARISON_OP_HALF2_MACRO(set.gtu);
}
#undef __BOOL_COMPARISON_OP_HALF2_MACRO
/******************************************************************************
*                             cuhalf comparison                              *
******************************************************************************/
#define __COMPARISON_OP_HALF_MACRO(name) do {\
   unsigned short val; \
   asm( "{ .reg .pred __$temp3;\n" \
        "  setp."#name".f16  __$temp3, %1, %2;\n" \
        "  selp.u16 %0, 1, 0, __$temp3;}" \
        : "=h"(val) : "h"(cuhalf_TO_CUS(a)), "h"(cuhalf_TO_CUS(b))); \
   return val ? true : false; \
} while(0)
__CUDA_FP16_DECL__ bool __heq(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(eq);
}
__CUDA_FP16_DECL__ bool __hne(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(ne);
}
__CUDA_FP16_DECL__ bool __hle(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(le);
}
__CUDA_FP16_DECL__ bool __hge(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(ge);
}
__CUDA_FP16_DECL__ bool __hlt(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(lt);
}
__CUDA_FP16_DECL__ bool __hgt(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(gt);
}
__CUDA_FP16_DECL__ bool __hequ(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(equ);
}
__CUDA_FP16_DECL__ bool __hneu(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(neu);
}
__CUDA_FP16_DECL__ bool __hleu(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(leu);
}
__CUDA_FP16_DECL__ bool __hgeu(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(geu);
}
__CUDA_FP16_DECL__ bool __hltu(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(ltu);
}
__CUDA_FP16_DECL__ bool __hgtu(const cuhalf a, const cuhalf b)
{
    __COMPARISON_OP_HALF_MACRO(gtu);
}
#undef __COMPARISON_OP_HALF_MACRO
/******************************************************************************
*                            cuhalf2 arithmetic                             *
******************************************************************************/
#define __BINARY_OP_HALF2_MACRO(name) do {\
   cuhalf2 val; \
   asm( "{"#name".f16x2 %0,%1,%2;\n}" \
        :"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)),"r"(cuhalf2_TO_CUI(b))); \
   return val; \
} while(0)

__CUDA_FP16_DECL__ cuhalf2 __hadd2(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(add);
}
__CUDA_FP16_DECL__ cuhalf2 __hsub2(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(sub);
}
__CUDA_FP16_DECL__ cuhalf2 __hmul2(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(mul);
}
__CUDA_FP16_DECL__ cuhalf2 __hadd2_sat(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(add.sat);
}
__CUDA_FP16_DECL__ cuhalf2 __hsub2_sat(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ cuhalf2 __hmul2_sat(const cuhalf2 a, const cuhalf2 b)
{
    __BINARY_OP_HALF2_MACRO(mul.sat);
}
#undef __BINARY_OP_HALF2_MACRO
#define __TERNARY_OP_HALF2_MACRO(name) do {\
   cuhalf2 val; \
   asm( "{"#name".f16x2 %0,%1,%2,%3;\n}" \
        :"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)),"r"(cuhalf2_TO_CUI(b)),"r"(cuhalf2_TO_CUI(c))); \
   return val; \
} while(0)
__CUDA_FP16_DECL__ cuhalf2 __hfma2(const cuhalf2 a, const cuhalf2 b, const cuhalf2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ cuhalf2 __hfma2_sat(const cuhalf2 a, const cuhalf2 b, const cuhalf2 c)
{
    __TERNARY_OP_HALF2_MACRO(fma.rn.sat);
}
#undef __TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ cuhalf2 __h2div(cuhalf2 a, cuhalf2 b) {
    cuhalf ha, hb;

    ha = __low2half(a);
    hb = __low2half(b);

    cuhalf v1 = __hdiv(ha, hb);

    ha = __high2half(a);
    hb = __high2half(b);

    cuhalf v2 = __hdiv(ha, hb);

    return __halves2half2(v1, v2);
}
/******************************************************************************
*                             cuhalf arithmetic                             *
******************************************************************************/
#define __BINARY_OP_HALF_MACRO(name) do {\
   cuhalf val; \
   asm( "{"#name".f16 %0,%1,%2;\n}" \
        :"=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)),"h"(cuhalf_TO_CUS(b))); \
   return val; \
} while(0)
__CUDA_FP16_DECL__ cuhalf __hadd(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(add);
}
__CUDA_FP16_DECL__ cuhalf __hsub(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(sub);
}
__CUDA_FP16_DECL__ cuhalf __hmul(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(mul);
}
__CUDA_FP16_DECL__ cuhalf __hadd_sat(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(add.sat);
}
__CUDA_FP16_DECL__ cuhalf __hsub_sat(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(sub.sat);
}
__CUDA_FP16_DECL__ cuhalf __hmul_sat(const cuhalf a, const cuhalf b)
{
    __BINARY_OP_HALF_MACRO(mul.sat);
}
#undef __BINARY_OP_HALF_MACRO
#define __TERNARY_OP_HALF_MACRO(name) do {\
   cuhalf val; \
   asm( "{"#name".f16 %0,%1,%2,%3;\n}" \
        :"=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)),"h"(cuhalf_TO_CUS(b)),"h"(cuhalf_TO_CUS(c))); \
   return val; \
} while(0)
__CUDA_FP16_DECL__ cuhalf __hfma(const cuhalf a, const cuhalf b, const cuhalf c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn);
}
__CUDA_FP16_DECL__ cuhalf __hfma_sat(const cuhalf a, const cuhalf b, const cuhalf c)
{
    __TERNARY_OP_HALF_MACRO(fma.rn.sat);
}
#undef __TERNARY_OP_HALF2_MACRO
__CUDA_FP16_DECL__ cuhalf __hdiv(cuhalf a, cuhalf b) {
    cuhalf v, abs, den;
    cuhalf_TO_US(den) = 0x008F;
    float fa, fb, fv, rcp;

    fa = cuhalf2float(a);
    fb = cuhalf2float(b);

    asm("{rcp.approx.f32 %0, %1;\n}" :"=f"(rcp) : "f"(fb));

    fv = rcp * fa;

    v = __float2half(fv);
    cuhalf_TO_US(abs) = (unsigned short)(((unsigned int)cuhalf_TO_CUS(v)) & 0x00007FFF);
    if (__hlt(abs, den) && (!(cuhalf_TO_CUS(abs) == 0x0000))) {
        float err = __fmaf_rn(-fb, fv, fa);
        fv = __fmaf_rn(rcp, err, fv);
        v = __float2half(fv);
    }
    return v;
}

/******************************************************************************
*                             cuhalf2 functions                  *
******************************************************************************/
#define __SPEC_CASE2(i,r, spc, ulp) \
   "{.reg.b32 spc, ulp, p;\n"\
   "  mov.b32 spc,"#spc";\n"\
   "  mov.b32 ulp,"#ulp";\n"\
   "  set.eq.f16x2.f16x2 p,"#i", spc;\n"\
   "  fma.rn.f16x2 "#r",p,ulp,"#r";\n}\n"
#define __SPEC_CASE(i,r, spc, ulp) \
   "{.reg.b16 spc, ulp, p;\n"\
   "  mov.b16 spc,"#spc";\n"\
   "  mov.b16 ulp,"#ulp";\n"\
   "  set.eq.f16.f16 p,"#i", spc;\n"\
   "  fma.rn.f16 "#r",p,ulp,"#r";\n}\n"
#define __APPROX_FCAST(fun) do {\
   cuhalf val;\
   asm("{.reg.b32         f;        \n"\
                " .reg.b16         r;        \n"\
                "  mov.b16         r,%1;     \n"\
                "  cvt.f32.f16     f,r;      \n"\
                "  "#fun".approx.f32   f,f;  \n"\
                "  cvt.rn.f16.f32      r,f;  \n"\
                "  mov.b16         %0,r;     \n"\
                "}": "=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));\
   return val;\
} while(0)
#define __APPROX_FCAST2(fun) do {\
   cuhalf2 val;\
   asm("{.reg.b16         hl, hu;         \n"\
                " .reg.b32         fl, fu;         \n"\
                "  mov.b32         {hl, hu}, %1;   \n"\
                "  cvt.f32.f16     fl, hl;         \n"\
                "  cvt.f32.f16     fu, hu;         \n"\
                "  "#fun".approx.f32   fl, fl;     \n"\
                "  "#fun".approx.f32   fu, fu;     \n"\
                "  cvt.rn.f16.f32      hl, fl;     \n"\
                "  cvt.rn.f16.f32      hu, fu;     \n"\
                "  mov.b32         %0, {hl, hu};   \n"\
                "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));       \
   return val;\
} while(0)
static __device__ __forceinline__ float __float_simpl_sinf(float);
static __device__ __forceinline__ float __float_simpl_cosf(float);
__CUDA_FP16_DECL__ cuhalf __hsin_internal(const cuhalf a) {
    float f = cuhalf2float(a);
    f = __float_simpl_sinf(f);
    return __float2half_rn(f);
}
__CUDA_FP16_DECL__ cuhalf hsin(const cuhalf a) {
    cuhalf r = __hsin_internal(a);
    asm("{\n\t"
        "  .reg.b16 i,r,t;     \n\t"
        "  mov.b16 r, %0;      \n\t"
        "  mov.b16 i, %1;      \n\t"
        "  mov.b16 t, 0x8000;  \n\t"
        "  and.b16 t,r,t;      \n\t"
        __SPEC_CASE(i, r, 0X32B3, 0x0800)
        __SPEC_CASE(i, r, 0X5CB0, 0x1000)
        __SPEC_CASE(i, r, 0XB2B3, 0x8800)
        __SPEC_CASE(i, r, 0XDCB0, 0x9000)
        "  or.b16  r,r,t;      \n\t"
        "  mov.b16 %0, r;      \n"
        "}\n" : "+h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf2 h2sin(const cuhalf2 a) {
    cuhalf l = __low2half(a);
    cuhalf h = __high2half(a);
    cuhalf2 r = __halves2half2(__hsin_internal(l), __hsin_internal(h));
    asm("{\n\t"
        "  .reg.b32 i,r,t;             \n\t"
        "  mov.b32 r, %0;              \n\t"
        "  mov.b32 i, %1;              \n\t"
        "  and.b32 t, r, 0x80008000;   \n\t"
        __SPEC_CASE2(i, r, 0X32B332B3, 0x08000800)
        __SPEC_CASE2(i, r, 0X5CB05CB0, 0x10001000)
        __SPEC_CASE2(i, r, 0XB2B3B2B3, 0x88008800)
        __SPEC_CASE2(i, r, 0XDCB0DCB0, 0x90009000)
        "  or.b32  r, r, t;            \n\t"
        "  mov.b32 %0, r;              \n"
        "}\n" : "+r"(cuhalf2_TO_UI(r)) : "r"(cuhalf2_TO_CUI(a)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf __hcos_internal(const cuhalf a) {
    float f = cuhalf2float(a);
    f = __float_simpl_cosf(f);
    return __float2half_rn(f);
}
__CUDA_FP16_DECL__ cuhalf hcos(const cuhalf a) {
    cuhalf r = __hcos_internal(a);
    asm("{\n\t"
        "  .reg.b16 i,r;        \n\t"
        "  mov.b16 r, %0;       \n\t"
        "  mov.b16 i, %1;       \n\t"
        __SPEC_CASE(i, r, 0X2B7C, 0x1000)
        __SPEC_CASE(i, r, 0XAB7C, 0x1000)
        "  mov.b16 %0, r;       \n"
        "}\n" : "+h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(a)));
    return r;
}
__CUDA_FP16_DECL__ cuhalf2 h2cos(const cuhalf2 a) {
    cuhalf l = __low2half(a);
    cuhalf h = __high2half(a);
    cuhalf2 r = __halves2half2(__hcos_internal(l), __hcos_internal(h));
    asm("{\n\t"
        "  .reg.b32 i,r;   \n\t"
        "  mov.b32 r, %0;  \n\t"
        "  mov.b32 i, %1;  \n\t"
        __SPEC_CASE2(i, r, 0X2B7C2B7C, 0x10001000)
        __SPEC_CASE2(i, r, 0XAB7CAB7C, 0x10001000)
        "  mov.b32 %0, r;  \n"
        "}\n" : "+r"(cuhalf2_TO_UI(r)) : "r"(cuhalf2_TO_CUI(a)));
    return r;
}
static __device__ __forceinline__ float __internal_trig_reduction_kernel(float a, int *quadrant)
{
    float j, t;
    int q;
    q = __float2int_rn(a * 0.636619772f);
    j = (float)q;
    t = __fmaf_rn(-j, 1.5707962512969971e+000f, a);
    t = __fmaf_rn(-j, 7.5497894158615964e-008f, t);
    *quadrant = q;
    return t;
}
static __device__ __forceinline__ float __internal_sin_cos_kernel(float x, int i)
{
    float x2, z;
    x2 = x*x;

    if (i & 1) {
        z = 2.44331571e-5f;
        z = __fmaf_rn(z, x2, -1.38873163e-3f);
    }
    else {
        z = -1.95152959e-4f;
        z = __fmaf_rn(z, x2, 8.33216087e-3f);
    }
    if (i & 1) {
        z = __fmaf_rn(z, x2, 4.16666457e-2f);
        z = __fmaf_rn(z, x2, -5.00000000e-1f);
    }
    else {
        z = __fmaf_rn(z, x2, -1.66666546e-1f);
        z = __fmaf_rn(z, x2, 0.0f);
    }
    x = __fmaf_rn(z, x, x);
    if (i & 1) x = __fmaf_rn(z, x2, 1.0f);
    if (i & 2) x = __fmaf_rn(x, -1.0f, 0.0f);
    return x;
}
static __device__ __forceinline__ float __float_simpl_sinf(float a)
{
    float z;
    int i;
    if (isinf(a)) {
        a = a * 0.0f;
    }
    a = __internal_trig_reduction_kernel(a, &i);
    z = __internal_sin_cos_kernel(a, i);
    return z;
}
static __device__ __forceinline__ float __float_simpl_cosf(float a)
{
    float z;
    int i;
    if (isinf(a)) {
        a = a * 0.0f;
    }
    a = __internal_trig_reduction_kernel(a, &i);
    i++;
    z = __internal_sin_cos_kernel(a, i);
    return z;
}
__CUDA_FP16_DECL__ cuhalf hexp(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         h,r;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  mov.b32         C, 0x3fb8aa3b;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f,f;        \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X1F79, 0x9400)
        __SPEC_CASE(h, r, 0X25CF, 0x9400)
        __SPEC_CASE(h, r, 0XC13B, 0x0400)
        __SPEC_CASE(h, r, 0XC1EF, 0x0200)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2exp(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x3fb8aa3b;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0X1F791F79, 0x94009400)
        __SPEC_CASE2(h, r, 0X25CF25CF, 0x94009400)
        __SPEC_CASE2(h, r, 0XC13BC13B, 0x04000400)
        __SPEC_CASE2(h, r, 0XC1EFC1EF, 0x02000200)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf hexp2(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b32         f, ULP;         \n"
        " .reg.b16         r;              \n"
        "  mov.b16         r,%1;           \n"
        "  cvt.f32.f16     f,r;            \n"
        "  ex2.approx.f32      f,f;        \n"
        "  mov.b32         ULP, 0x33800000;\n"
        "  fma.rn.f32      f,f,ULP,f;      \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}": "=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2exp2(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, ULP;    \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  mov.b32         ULP, 0x33800000;\n"
        "  fma.rn.f32      fl,fl,ULP,fl;   \n"
        "  fma.rn.f32      fu,fu,ULP,fu;   \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         %0, {hl, hu};   \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf hexp10(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b16         h,r;            \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  mov.b32         C, 0x40549A78;  \n"
        "  mul.f32         f,f,C;          \n"
        "  ex2.approx.f32      f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x34DE, 0x9800)
        __SPEC_CASE(h, r, 0x9766, 0x9000)
        __SPEC_CASE(h, r, 0x9972, 0x1000)
        __SPEC_CASE(h, r, 0xA5C4, 0x1000)
        __SPEC_CASE(h, r, 0xBF0A, 0x8100)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2exp10(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         h,r,fl,fu, C;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  mov.b32         h, %1;          \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  mov.b32         C, 0x40549A78;  \n"
        "  mul.f32         fl,fl,C;        \n"
        "  mul.f32         fu,fu,C;        \n"
        "  ex2.approx.f32      fl, fl;     \n"
        "  ex2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(h, r, 0x34DE34DE, 0x98009800)
        __SPEC_CASE2(h, r, 0x97669766, 0x90009000)
        __SPEC_CASE2(h, r, 0x99729972, 0x10001000)
        __SPEC_CASE2(h, r, 0xA5C4A5C4, 0x10001000)
        __SPEC_CASE2(h, r, 0xBF0ABF0A, 0x81008100)
        "  mov.b32         %0, r;  \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf hlog2(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f;              \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.f32      f, f;       \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(r, r, 0xA2E2, 0x8080)
        __SPEC_CASE(r, r, 0xBF46, 0x9400)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2log2(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, r, p;   \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  lg2.approx.f32      fl, fl;     \n"
        "  lg2.approx.f32      fu, fu;     \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         r, {hl, hu};    \n"
        __SPEC_CASE2(r, r, 0xA2E2A2E2, 0x80808080)
        __SPEC_CASE2(r, r, 0xBF46BF46, 0x94009400)
        "  mov.b32         %0, r;          \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf hlog(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b32         f, C;           \n"
        " .reg.b16         r,h;            \n"
        "  mov.b16         h,%1;           \n"
        "  cvt.f32.f16     f,h;            \n"
        "  lg2.approx.f32      f,f;        \n"
        "  mov.b32         C, 0x3f317218;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        __SPEC_CASE(h, r, 0X160D, 0x9C00)
        __SPEC_CASE(h, r, 0X3BFE, 0x8010)
        __SPEC_CASE(h, r, 0X3C0B, 0x8080)
        __SPEC_CASE(h, r, 0X6051, 0x1C00)
        "  mov.b16         %0,r;           \n"
        "}": "=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2log(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3f317218;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0X160D160D, 0x9C009C00)
        __SPEC_CASE2(h, r, 0X3BFE3BFE, 0x80108010)
        __SPEC_CASE2(h, r, 0X3C0B3C0B, 0x80808080)
        __SPEC_CASE2(h, r, 0X60516051, 0x1C001C00)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf hlog10(const cuhalf a) {
    cuhalf val;
    asm("{.reg.b16         h, r;           \n"
        " .reg.b32         f, C;           \n"
        "  mov.b16         h, %1;          \n"
        "  cvt.f32.f16     f, h;           \n"
        "  lg2.approx.f32      f, f;       \n"
        "  mov.b32         C, 0x3E9A209B;  \n"
        "  mul.f32         f,f,C;          \n"
        "  cvt.rn.f16.f32      r, f;       \n"
        __SPEC_CASE(h, r, 0x338F, 0x1000)
        __SPEC_CASE(h, r, 0x33F8, 0x9000)
        __SPEC_CASE(h, r, 0x57E1, 0x9800)
        __SPEC_CASE(h, r, 0x719D, 0x9C00)
        "  mov.b16         %0, r;          \n"
        "}":"=h"(cuhalf_TO_US(val)) : "h"(cuhalf_TO_CUS(a)));
    return val;
}
__CUDA_FP16_DECL__ cuhalf2 h2log10(const cuhalf2 a) {
    cuhalf2 val;
    asm("{.reg.b16         hl, hu;             \n"
        " .reg.b32         r, fl, fu, C, h;    \n"
        "  mov.b32         {hl, hu}, %1;       \n"
        "  mov.b32         h, %1;              \n"
        "  cvt.f32.f16     fl, hl;             \n"
        "  cvt.f32.f16     fu, hu;             \n"
        "  lg2.approx.f32      fl, fl;         \n"
        "  lg2.approx.f32      fu, fu;         \n"
        "  mov.b32         C, 0x3E9A209B;      \n"
        "  mul.f32         fl,fl,C;            \n"
        "  mul.f32         fu,fu,C;            \n"
        "  cvt.rn.f16.f32      hl, fl;         \n"
        "  cvt.rn.f16.f32      hu, fu;         \n"
        "  mov.b32         r, {hl, hu};        \n"
        __SPEC_CASE2(h, r, 0x338F338F, 0x10001000)
        __SPEC_CASE2(h, r, 0x33F833F8, 0x90009000)
        __SPEC_CASE2(h, r, 0x57E157E1, 0x98009800)
        __SPEC_CASE2(h, r, 0x719D719D, 0x9C009C00)
        "  mov.b32         %0, r;              \n"
        "}":"=r"(cuhalf2_TO_UI(val)) : "r"(cuhalf2_TO_CUI(a)));
    return val;
}
#undef __SPEC_CASE2
#undef __SPEC_CASE
__CUDA_FP16_DECL__ cuhalf2 h2rcp(const cuhalf2 a) {
    __APPROX_FCAST2(rcp);
}
__CUDA_FP16_DECL__ cuhalf hrcp(const cuhalf a) {
    __APPROX_FCAST(rcp);
}
__CUDA_FP16_DECL__ cuhalf2 h2rsqrt(const cuhalf2 a) {
    __APPROX_FCAST2(rsqrt);
}
__CUDA_FP16_DECL__ cuhalf hrsqrt(const cuhalf a) {
    __APPROX_FCAST(rsqrt);
}
__CUDA_FP16_DECL__ cuhalf2 h2sqrt(const cuhalf2 a) {
    __APPROX_FCAST2(sqrt);
}
__CUDA_FP16_DECL__ cuhalf hsqrt(const cuhalf a) {
    __APPROX_FCAST(sqrt);
}
#undef __APPROX_FCAST
#undef __APPROX_FCAST2
__CUDA_FP16_DECL__ cuhalf2 __hisnan2(const cuhalf2 a)
{
    cuhalf2 r;
    asm("{set.nan.f16x2.f16x2 %0,%1,%2;\n}"
        :"=r"(cuhalf2_TO_UI(r)) : "r"(cuhalf2_TO_CUI(a)), "r"(cuhalf2_TO_CUI(a)));
    return r;
}
__CUDA_FP16_DECL__ bool __hisnan(const cuhalf a)
{
    cuhalf r;
    asm("{set.nan.f16.f16 %0,%1,%2;\n}"
        :"=h"(cuhalf_TO_US(r)) : "h"(cuhalf_TO_CUS(a)), "h"(cuhalf_TO_CUS(a)));
    if (cuhalf_TO_CUS(r) == 0)
        return false;
    else return true;
}
__CUDA_FP16_DECL__ cuhalf2 __hneg2(const cuhalf2 a)
{
    cuhalf2 zero = __float2half2_rn(0.0);
    return __hsub2(zero, a);
}
__CUDA_FP16_DECL__ cuhalf __hneg(const cuhalf a)
{
    cuhalf zero;
    zero = __float2half(0.0);
    return __hsub(zero, a);
}
#endif /*__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/

/* Define __PTR for atomicAdd prototypes below, undef after done */
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

__CUDA_FP16_DECL__  cuhalf2 atomicAdd(cuhalf2 *address, cuhalf2 val) {
    cuhalf2 r;
    asm volatile ("{ atom.add.noftz.f16x2 %0,[%1],%2; }\n"
                  : "=r"(cuhalf2_TO_UI(r)) : __PTR(address), "r"(cuhalf2_TO_CUI(val))
                  : "memory");
   return r;
}

#endif /*!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600*/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

__CUDA_FP16_DECL__  cuhalf atomicAdd(cuhalf *address, cuhalf val) {
    cuhalf r;
    asm volatile ("{ atom.add.noftz.f16 %0,[%1],%2; }\n"
                  : "=h"(*(reinterpret_cast<unsigned short int *>(&r)))
                  : __PTR(address), "h"(*(reinterpret_cast<const unsigned short int *>(&val)))
                  : "memory");
   return r;
}

#endif /*!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700*/

#undef __PTR

#undef __CUDA_FP16_DECL__
#endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */

#undef __CUDA_HOSTDEVICE_FP16_DECL__
#undef __CUDA_FP16_DECL__
#undef cuhalf_TO_US
#undef cuhalf_TO_CUS
#undef cuhalf2_TO_UI
#undef cuhalf2_TO_CUI
#undef __COPY_FLOAT_TO_UI
#undef __COPY_UI_TO_FLOAT

/* Define first-class types "half" and "half2", unless user specifies otherwise via "#define CUDA_NO_HALF" */
/* C cannot ever have these types defined here, because cuhalf and cuhalf2 are C++ classes */
//  #if defined(__cplusplus) && !defined(CUDA_NO_HALF)
//  typedef cuhalf half;
//  typedef cuhalf2 half2;
//  #endif /* defined(__cplusplus) && !defined(CUDA_NO_HALF) */

#endif /* end of include guard: __CUDA_FP16_HPP__ */
