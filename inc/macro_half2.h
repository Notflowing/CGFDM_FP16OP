#ifndef __MACRO_HALF2__
#define __MACRO_HALF2__
#pragma once

#define INDEX_HALF2( i, j, k ) ( (i) + (j) * (_nx_2) + (k) * ((_nx_2) * (_ny_)) )

#define MacCormack_X_x(W, FB)  ( (rDH) * (FB) * (af_1 * W[INDEX((2 * i + (FB) * (-1)), j, k)] + \
                                                  af0 * W[INDEX((2 * i + (FB) *  (0)), j, k)] + \
                                                  af1 * W[INDEX((2 * i + (FB) *  (1)), j, k)] + \
                                                  af2 * W[INDEX((2 * i + (FB) *  (2)), j, k)] + \
                                                  af3 * W[INDEX((2 * i + (FB) *  (3)), j, k)]) )
#define MacCormack_X_y(W, FB)  ( (rDH) * (FB) * (af_1 * W[INDEX((2 * i + 1 + (FB) * (-1)), j, k)] + \
                                                  af0 * W[INDEX((2 * i + 1 + (FB) *  (0)), j, k)] + \
                                                  af1 * W[INDEX((2 * i + 1 + (FB) *  (1)), j, k)] + \
                                                  af2 * W[INDEX((2 * i + 1 + (FB) *  (2)), j, k)] + \
                                                  af3 * W[INDEX((2 * i + 1 + (FB) *  (3)), j, k)]) )

// generate index of adjacent point: Up/Down/Left/Right
#define INDEX_HALF_et(i, j, k, offset) ( (i) + ((j) + (offset)) * (_nx_2) + (k) * (_nx_2) * (_ny_) )
#define INDEX_HALF_zt(i, j, k, offset) ( (i) + (j) * (_nx_2) + ((k) + (offset)) * (_nx_2) * (_ny_) )

// DRP/opt MacCormack scheme
#define MacCormack_HALF2(W, FB, SUB)  ( (rDH) * (FB) * (af_1 * W[INDEX_HALF_##SUB(i, j, k, (FB) * (-1))] + \
                                                         af0 * W[INDEX_HALF_##SUB(i, j, k, (FB) *  (0))] + \
                                                         af1 * W[INDEX_HALF_##SUB(i, j, k, (FB) *  (1))] + \
                                                         af2 * W[INDEX_HALF_##SUB(i, j, k, (FB) *  (2))] + \
                                                         af3 * W[INDEX_HALF_##SUB(i, j, k, (FB) *  (3))]) )

#endif