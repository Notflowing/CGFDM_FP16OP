#ifndef OVERLOAD_HALF2_H
#define OVERLOAD_HALF2_H

#pragma once

// define and overload some operations between half, half2 and basic type

#include <cuda_fp16.h>

// overload half-precision operator+
__device__ __forceinline__ __half2 operator+(const __half2 &l, const int &r)                { return __hadd2(l, make_half2(__int2half_rn(r), __int2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const unsigned int &r)       { return __hadd2(l, make_half2(__uint2half_rn(r), __uint2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const short &r)              { return __hadd2(l, make_half2(__short2half_rn(r), __short2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const unsigned short &r)     { return __hadd2(l, make_half2(__ushort2half_rn(r), __ushort2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const long long &r)          { return __hadd2(l, make_half2(__ll2half_rn(r), __ll2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const unsigned long long &r) { return __hadd2(l, make_half2(__ull2half_rn(r), __ull2half_rn(r))); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const float &r)              { return __hadd2(l, __float2half2_rn(r)); }
__device__ __forceinline__ __half2 operator+(const __half2 &l, const double &r)             { return __hadd2(l, __float2half2_rn(static_cast<float>(r))); }

__device__ __forceinline__ __half2 operator+(const int &l, const __half2 &r)                { return __hadd2(make_half2(__int2half_rn(l), __int2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const unsigned int &l, const __half2 &r)       { return __hadd2(make_half2(__uint2half_rn(l), __uint2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const short &l, const __half2 &r)              { return __hadd2(make_half2(__short2half_rn(l), __short2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const unsigned short &l, const __half2 &r)     { return __hadd2(make_half2(__ushort2half_rn(l), __ushort2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const long long &l, const __half2 &r)          { return __hadd2(make_half2(__ll2half_rn(l), __ll2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const unsigned long long &l, const __half2 &r) { return __hadd2(make_half2(__ull2half_rn(l), __ull2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator+(const float &l, const __half2 &r)              { return __hadd2(__float2half2_rn(l), r); }
__device__ __forceinline__ __half2 operator+(const double &l, const __half2 &r)             { return __hadd2(__float2half2_rn(static_cast<float>(l)), r); }

// overload half-precision operator-
__device__ __forceinline__ __half2 operator-(const __half2 &l, const int &r)                { return __hsub2(l, make_half2(__int2half_rn(r), __int2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const unsigned int &r)       { return __hsub2(l, make_half2(__uint2half_rn(r), __uint2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const short &r)              { return __hsub2(l, make_half2(__short2half_rn(r), __short2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const unsigned short &r)     { return __hsub2(l, make_half2(__ushort2half_rn(r), __ushort2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const long long &r)          { return __hsub2(l, make_half2(__ll2half_rn(r), __ll2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const unsigned long long &r) { return __hsub2(l, make_half2(__ull2half_rn(r), __ull2half_rn(r))); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const float &r)              { return __hsub2(l, __float2half2_rn(r)); }
__device__ __forceinline__ __half2 operator-(const __half2 &l, const double &r)             { return __hsub2(l, __float2half2_rn(static_cast<float>(r))); }

__device__ __forceinline__ __half2 operator-(const int &l, const __half2 &r)                { return __hsub2(make_half2(__int2half_rn(l), __int2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const unsigned int &l, const __half2 &r)       { return __hsub2(make_half2(__uint2half_rn(l), __uint2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const short &l, const __half2 &r)              { return __hsub2(make_half2(__short2half_rn(l), __short2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const unsigned short &l, const __half2 &r)     { return __hsub2(make_half2(__ushort2half_rn(l), __ushort2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const long long &l, const __half2 &r)          { return __hsub2(make_half2(__ll2half_rn(l), __ll2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const unsigned long long &l, const __half2 &r) { return __hsub2(make_half2(__ull2half_rn(l), __ull2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator-(const float &l, const __half2 &r)              { return __hsub2(__float2half2_rn(l), r); }
__device__ __forceinline__ __half2 operator-(const double &l, const __half2 &r)             { return __hsub2(__float2half2_rn(static_cast<float>(l)), r); }

// overload half-precision operator*
__device__ __forceinline__ __half2 operator*(const __half2 &l, const int &r)                { return __hmul2(l, make_half2(__int2half_rn(r), __int2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const unsigned int &r)       { return __hmul2(l, make_half2(__uint2half_rn(r), __uint2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const short &r)              { return __hmul2(l, make_half2(__short2half_rn(r), __short2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const unsigned short &r)     { return __hmul2(l, make_half2(__ushort2half_rn(r), __ushort2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const long long &r)          { return __hmul2(l, make_half2(__ll2half_rn(r), __ll2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const unsigned long long &r) { return __hmul2(l, make_half2(__ull2half_rn(r), __ull2half_rn(r))); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const float &r)              { return __hmul2(l, __float2half2_rn(r)); }
__device__ __forceinline__ __half2 operator*(const __half2 &l, const double &r)             { return __hmul2(l, __float2half2_rn(static_cast<float>(r))); }

__device__ __forceinline__ __half2 operator*(const int &l, const __half2 &r)                { return __hmul2(make_half2(__int2half_rn(l), __int2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const unsigned int &l, const __half2 &r)       { return __hmul2(make_half2(__uint2half_rn(l), __uint2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const short &l, const __half2 &r)              { return __hmul2(make_half2(__short2half_rn(l), __short2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const unsigned short &l, const __half2 &r)     { return __hmul2(make_half2(__ushort2half_rn(l), __ushort2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const long long &l, const __half2 &r)          { return __hmul2(make_half2(__ll2half_rn(l), __ll2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const unsigned long long &l, const __half2 &r) { return __hmul2(make_half2(__ull2half_rn(l), __ull2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator*(const float &l, const __half2 &r)              { return __hmul2(__float2half2_rn(l), r); }
__device__ __forceinline__ __half2 operator*(const double &l, const __half2 &r)             { return __hmul2(__float2half2_rn(static_cast<float>(l)), r); }

// overload half-precision operator/
__device__ __forceinline__ __half2 operator/(const __half2 &l, const int &r)                { return __h2div(l, make_half2(__int2half_rn(r), __int2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const unsigned int &r)       { return __h2div(l, make_half2(__uint2half_rn(r), __uint2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const short &r)              { return __h2div(l, make_half2(__short2half_rn(r), __short2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const unsigned short &r)     { return __h2div(l, make_half2(__ushort2half_rn(r), __ushort2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const long long &r)          { return __h2div(l, make_half2(__ll2half_rn(r), __ll2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const unsigned long long &r) { return __h2div(l, make_half2(__ull2half_rn(r), __ull2half_rn(r))); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const float &r)              { return __h2div(l, __float2half2_rn(r)); }
__device__ __forceinline__ __half2 operator/(const __half2 &l, const double &r)             { return __h2div(l, __float2half2_rn(static_cast<float>(r))); }

__device__ __forceinline__ __half2 operator/(const int &l, const __half2 &r)                { return __h2div(make_half2(__int2half_rn(l), __int2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const unsigned int &l, const __half2 &r)       { return __h2div(make_half2(__uint2half_rn(l), __uint2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const short &l, const __half2 &r)              { return __h2div(make_half2(__short2half_rn(l), __short2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const unsigned short &l, const __half2 &r)     { return __h2div(make_half2(__ushort2half_rn(l), __ushort2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const long long &l, const __half2 &r)          { return __h2div(make_half2(__ll2half_rn(l), __ll2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const unsigned long long &l, const __half2 &r) { return __h2div(make_half2(__ull2half_rn(l), __ull2half_rn(l)), r); }
__device__ __forceinline__ __half2 operator/(const float &l, const __half2 &r)              { return __h2div(__float2half2_rn(l), r); }
__device__ __forceinline__ __half2 operator/(const double &l, const __half2 &r)             { return __h2div(__float2half2_rn(static_cast<float>(l)), r); }

// some basic arithmetic operations and  increment and decrement operators
// +=, -=, *=, /=, ++, --

// unary plus and inverse operators
// +, -

// some basic comparison operations
// ==, !=, >, <, >=, <=


#endif