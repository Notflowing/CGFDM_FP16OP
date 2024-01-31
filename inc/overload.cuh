#ifndef OVERLOAD_H
#define OVERLOAD_H

#pragma once

// define and overload some operations between half, half2 and basic type

#include <cuda_fp16.h>

// overload half-precision operator+
__device__ __forceinline__ __half operator+(const __half &l, const int &r)                  { return __hadd(l, __int2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const unsigned int &r)         { return __hadd(l, __uint2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const short &r)                { return __hadd(l, __short2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const unsigned short &r)       { return __hadd(l, __ushort2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const long long &r)            { return __hadd(l, __ll2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const unsigned long long &r)   { return __hadd(l, __ull2half_rn(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const float &r)                { return __hadd(l, __float2half(r)); }
__device__ __forceinline__ __half operator+(const __half &l, const double &r)               { return __hadd(l, __float2half(static_cast<float>(r))); }

__device__ __forceinline__ __half operator+(const int &l, const __half &r)                  { return __hadd(__int2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const unsigned int &l, const __half &r)         { return __hadd(__uint2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const short &l, const __half &r)                { return __hadd(__short2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const unsigned short &l, const __half &r)       { return __hadd(__ushort2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const long long &l, const __half &r)            { return __hadd(__ll2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const unsigned long long &l, const __half &r)   { return __hadd(__ull2half_rn(l), r); }
__device__ __forceinline__ __half operator+(const float &l, const __half &r)                { return __hadd(__float2half(l), r); }
__device__ __forceinline__ __half operator+(const double &l, const __half &r)               { return __hadd(__float2half(static_cast<float>(l)), r); }

// overload half-precision operator-
__device__ __forceinline__ __half operator-(const __half &l, const int &r)                  { return __hsub(l, __int2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const unsigned int &r)         { return __hsub(l, __uint2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const short &r)                { return __hsub(l, __short2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const unsigned short &r)       { return __hsub(l, __ushort2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const long long &r)            { return __hsub(l, __ll2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const unsigned long long &r)   { return __hsub(l, __ull2half_rn(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const float &r)                { return __hsub(l, __float2half(r)); }
__device__ __forceinline__ __half operator-(const __half &l, const double &r)               { return __hsub(l, __float2half(static_cast<float>(r))); }

__device__ __forceinline__ __half operator-(const int &l, const __half &r)                  { return __hsub(__int2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const unsigned int &l, const __half &r)         { return __hsub(__uint2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const short &l, const __half &r)                { return __hsub(__short2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const unsigned short &l, const __half &r)       { return __hsub(__ushort2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const long long &l, const __half &r)            { return __hsub(__ll2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const unsigned long long &l, const __half &r)   { return __hsub(__ull2half_rn(l), r); }
__device__ __forceinline__ __half operator-(const float &l, const __half &r)                { return __hsub(__float2half(l), r); }
__device__ __forceinline__ __half operator-(const double &l, const __half &r)               { return __hsub(__float2half(static_cast<float>(l)), r); }

// overload half-precision operator*
__device__ __forceinline__ __half operator*(const __half &l, const int &r)                  { return __hmul(l, __int2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const unsigned int &r)         { return __hmul(l, __uint2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const short &r)                { return __hmul(l, __short2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const unsigned short &r)       { return __hmul(l, __ushort2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const long long &r)            { return __hmul(l, __ll2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const unsigned long long &r)   { return __hmul(l, __ull2half_rn(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const float &r)                { return __hmul(l, __float2half(r)); }
__device__ __forceinline__ __half operator*(const __half &l, const double &r)               { return __hmul(l, __float2half(static_cast<float>(r))); }

__device__ __forceinline__ __half operator*(const int &l, const __half &r)                  { return __hmul(__int2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const unsigned int &l, const __half &r)         { return __hmul(__uint2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const short &l, const __half &r)                { return __hmul(__short2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const unsigned short &l, const __half &r)       { return __hmul(__ushort2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const long long &l, const __half &r)            { return __hmul(__ll2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const unsigned long long &l, const __half &r)   { return __hmul(__ull2half_rn(l), r); }
__device__ __forceinline__ __half operator*(const float &l, const __half &r)                { return __hmul(__float2half(l), r); }
__device__ __forceinline__ __half operator*(const double &l, const __half &r)               { return __hmul(__float2half(static_cast<float>(l)), r); }

// overload half-precision operator/
__device__ __forceinline__ __half operator/(const __half &l, const int &r)                  { return __hdiv(l, __int2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const unsigned int &r)         { return __hdiv(l, __uint2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const short &r)                { return __hdiv(l, __short2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const unsigned short &r)       { return __hdiv(l, __ushort2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const long long &r)            { return __hdiv(l, __ll2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const unsigned long long &r)   { return __hdiv(l, __ull2half_rn(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const float &r)                { return __hdiv(l, __float2half(r)); }
__device__ __forceinline__ __half operator/(const __half &l, const double &r)               { return __hdiv(l, __float2half(static_cast<float>(r))); }

__device__ __forceinline__ __half operator/(const int &l, const __half &r)                  { return __hdiv(__int2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const unsigned int &l, const __half &r)         { return __hdiv(__uint2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const short &l, const __half &r)                { return __hdiv(__short2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const unsigned short &l, const __half &r)       { return __hdiv(__ushort2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const long long &l, const __half &r)            { return __hdiv(__ll2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const unsigned long long &l, const __half &r)   { return __hdiv(__ull2half_rn(l), r); }
__device__ __forceinline__ __half operator/(const float &l, const __half &r)                { return __hdiv(__float2half(l), r); }
__device__ __forceinline__ __half operator/(const double &l, const __half &r)               { return __hdiv(__float2half(static_cast<float>(l)), r); }

// some basic arithmetic operations and  increment and decrement operators
// +=, -=, *=, /=, ++, --

// unary plus and inverse operators
// +, -

// some basic comparison operations
// ==, !=, >, <, >=, <=


#endif