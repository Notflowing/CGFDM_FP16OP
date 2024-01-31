#ifndef __MACRO__
#define __MACRO__
#pragma once


// HOST for host(CPU), DEVICE for device(GPU)
typedef enum HeterArch {HOST, DEVICE} HeterArch;
// XAXIS for x direction, YAXIS for y direction, ZAXIS for z direction
typedef enum AXIS {XAXIS, YAXIS, ZAXIS} AXIS;
// scaling factor
typedef enum SCALE {CO, CV, CS} SCALE;
//
typedef enum VTF {velocity, stress, freesurf} VTF;

// radius of gauss smooth
#define nGauss 3

#define A0  0   // ( -0.481231743137f )
#define A1	-1  // ( -1.049562606709f )
#define A2	-1  // ( -1.602529574275f )
#define A3	-1  // ( -1.778267193916f )
 


#define B0 0.333333333f     // 0.412253292915f
#define B1 0.75f            // 0.440216963931f
#define B2 0.66666667f      // 1.426311463224f
#define B3 0.25f            // 0.197876053732f



#define HALO 3
#define MIN(x, y) ( (x) < (y) ? (x) : (y) )
#define MAX(x, y) ( (x) > (y) ? (x) : (y) )

#define FOR_LOOP2D( i, j, startI, endI, startJ, endJ ) \
for ( j = startJ; j < endJ; j ++ ) {    \
for ( i = startI; i < endI; i ++ ) {    \

#define END_LOOP2D( ) }}


#define FOR_LOOP3D( i, j, k, startI, endI, startJ, endJ, startK, endK ) \
for ( k = startK; k < endK; k ++ ) {     \
for ( j = startJ; j < endJ; j ++ ) {     \
for ( i = startI; i < endI; i ++ ) {     \

#define END_LOOP3D( ) }}}


#define WAVESIZE 9      // 9 Wave components: Vx Vy Vz Txx Tyy Tzz Txy Txz Tyz
#define COORDSIZE 3     // 3 coordinate components: coordX coordY coordZ
#define CONTRASIZE 9    // contravariant components
#define MEDIUMSIZE 3    // 3 medium components: Vs Vp rho ( lam mu bouyancy )
#define MOMENTSIZE 6    // 6 moment rate components: Mxx, Myy, Mzz, Mxy, Mxz, Myz


#define PI 3.141592657f
#define RADIAN2DEGREE (180.0 / PI)
#define DEGREE2RADIAN (PI / 180.0)


#define POW2(x) ( (x) * (x) )
#define GAUSS_FUN(t, a, t0) ( exp(-POW2( ((t) - (t0)) / (a) )) / (a * 1.772453850905516) )


#define THREE 3

// forward difference coefficient for DRP/opt MacCormack scheme
#define af_1 (-0.30874f)
#define af0  (-0.6326f )
#define af1  ( 1.2330f )
#define af2  (-0.3334f )
#define af3  ( 0.04168f)
// backward difference coefficient for DRP/opt MacCormack scheme
#define ab_1 ( 0.30874f)
#define ab0  ( 0.6326f )
#define ab1  (-1.2330f )
#define ab2  ( 0.3334f )
#define ab3  (-0.04168f)

#define alpha1 0.0f
#define alpha2 0.5f
#define alpha3 0.5f
#define alpha4 1.0f

#define beta1 0.16666667f
#define beta2 0.33333333f
#define beta3 0.33333333f
#define beta4 0.16666667f

// coefficient for free surface surrounding points(the following 2 points)
#define Cf1 (-1.16666667f)
#define Cf2 ( 1.33333333f)
#define Cf3 (-0.16666667f)
#define Cb1 ( 1.16666667f)
#define Cb2 (-1.33333333f)
#define Cb3 ( 0.16666667f)


// define X_FAST
// grid: x, y, z
#define Index3D( i, j, k, nx, ny, nz ) ( (i) + (j) * (nx) + (k) * ((nx) * (ny)) )
#define Index2D( i, j, nx, ny ) ( (i) + (j) * (nx) )
// grid with 2 * HALO
#define INDEX( i, j, k ) ( (i) + (j) * (_nx_) + (k) * ((_nx_) * (_ny_)) )

// generate index of adjacent point: Up/Down/Left/Right
#define INDEX_xi(i, j, k, offset) ( ((i) + (offset)) + (j) * (_nx_) + (k) * (_nx_) * (_ny_) )
#define INDEX_et(i, j, k, offset) ( (i) + ((j) + (offset)) * (_nx_) + (k) * (_nx_) * (_ny_) )
#define INDEX_zt(i, j, k, offset) ( (i) + (j) * (_nx_) + ((k) + (offset)) * (_nx_) * (_ny_) )

// DRP/opt MacCormack scheme
#define MacCormack(W, FB, SUB)  ( (rDH) * (FB) * (af_1 * W[INDEX_##SUB(i, j, k, (FB) * (-1))] + \
                                                   af0 * W[INDEX_##SUB(i, j, k, (FB) *  (0))] + \
                                                   af1 * W[INDEX_##SUB(i, j, k, (FB) *  (1))] + \
                                                   af2 * W[INDEX_##SUB(i, j, k, (FB) *  (2))] + \
                                                   af3 * W[INDEX_##SUB(i, j, k, (FB) *  (3))]) )
// 2.5D tile
#define tile_blockDimX 32
#define tile_blockDimY 16
#define MacCormack_xi(tile) ( (rDH) * (FB1) * (af_1 * tile[ty][tx + (FB1) * (-1)] + \
								   				af0 * tile[ty][tx + (FB1) * ( 0)] + \
								   				af1 * tile[ty][tx + (FB1) * ( 1)] + \
								   				af2 * tile[ty][tx + (FB1) * ( 2)] + \
								   				af3 * tile[ty][tx + (FB1) * ( 3)]) )
#define MacCormack_et(tile) ( (rDH) * (FB2) * (af_1 * tile[ty + (FB2) * (-1)][tx] + \
								   				af0 * tile[ty + (FB2) * ( 0)][tx] + \
								   				af1 * tile[ty + (FB2) * ( 1)][tx] + \
								   				af2 * tile[ty + (FB2) * ( 2)][tx] + \
								   				af3 * tile[ty + (FB2) * ( 3)][tx]) )
#define MacCormack_zt(zval) ( (rDH) * (FB3) * (af_1 * zval[2 + (FB3) * (-2)] + \
								   				af0 * zval[2 + (FB3) * (-1)] + \
								   				af1 * zval[2 + (FB3) * ( 0)] + \
								   				af2 * zval[2 + (FB3) * ( 1)] + \
								   				af3 * zval[2 + (FB3) * ( 2)]) )
// 2.5D tile

// DRP/opt MacCormack for free surface
#define MacCormack_freesurf(J_T, FB) ( (rDH) * (FB) * (af_1 * J_T[HALO + (FB) * (-1)] + \
                                                        af0 * J_T[HALO + (FB) *  (0)] + \
                                                        af1 * J_T[HALO + (FB) *  (1)] + \
                                                        af2 * J_T[HALO + (FB) *  (2)] + \
                                                        af3 * J_T[HALO + (FB) *  (3)]) )
// free surface surrounding points(the following 2 points)
#define L2( W, FB, SUB ) ( (rDH) * (FB) * ( W[INDEX_##SUB( i, j, k, (FB) * (1) )] - W[index] ) )
#define L3( W, FB, SUB ) ( (rDH) * (FB) * ( Cf1 * W[index] + Cf2 * W[INDEX_##SUB( i, j, k, (FB) *  (1) )] + \
                                            Cf3 * W[INDEX_##SUB( i, j, k, (FB) *  (2) )] ) )

// dot product
#define DOT_PRODUCT3D(A1, A2, A3, B1, B2, B3) ( (A1) * (B1) + (A2) * (B2) + (A3) * (B3) )
#define DOT_PRODUCT2D(A1, A2, B1, B2) ( (A1) * (B1) + (A2) * (B2) )

#endif //__MACRO__
