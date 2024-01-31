#include "header.h"

// thread ---  int --> long long   overflow


typedef void (*WAVE_RK_FUNC_FLOAT )( FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long num );

__global__ void wave_rk0( FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	// FLOAT h_w, w, t_w, m_w;
	if ( i >= 0 && i < WStride )
	{
		m_W[i] = W[i];
		t_W[i] = m_W[i] + beta1  * h_W[i];
		  W[i] = m_W[i] + alpha2 * h_W[i];
	}
}


__global__ void wave_rk1( FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride)
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	// FLOAT h_w, w, t_w, m_w;
	if ( i >= 0 && i < WStride )
	{
		t_W[i] = t_W[i] + beta2  * h_W[i];
		  W[i] = m_W[i] + alpha3 * h_W[i];
	}
}


__global__ void wave_rk2( FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	// FLOAT h_w, w, t_w, m_w;
	if ( i >= 0 && i < WStride )
	{
		t_W[i] = t_W[i] + beta3 * h_W[i];
		  W[i] = m_W[i] + h_W[i];
	}
}


__global__ void wave_rk3( FLOAT * h_W, FLOAT * W, FLOAT * t_W, FLOAT * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	// FLOAT h_w, w, t_w, m_w;
	if ( i >= 0 && i < WStride )
	{
		W[i] = t_W[i] + beta4 * h_W[i];
	}
}

void waveRk( const GRID * const grid, const int irk, FLOAT * const h_W, FLOAT * const W, FLOAT * const t_W, FLOAT * const m_W )
{
	WAVE_RK_FUNC_FLOAT wave_rk[4] = { wave_rk0, wave_rk1, wave_rk2, wave_rk3 };
	long long num = grid->_nx_ * grid->_ny_ * grid->_nz_ * WAVESIZE;

	dim3 threads( 1024, 1, 1 );
	dim3 blocks;
	blocks.x = ( num + threads.x - 1 ) / threads.x;
	blocks.y = 1;
	blocks.z = 1;
	wave_rk[irk]<<< blocks, threads >>>( h_W, W, t_W, m_W, num );
	//wave_rk<<< blocks, threads >>>( h_W, W, t_W, m_W, num, DT );
	checkCudaErrors( cudaDeviceSynchronize( ) );

}



__global__ void new_wave_rk( FLOAT * const h_W, FLOAT * const W, const long long WStride, const float B, const float DT )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	FLOAT hw = 0.0f;
	FLOAT w = 0.0f;
	if ( i >= 0 && i < WStride )
	{
		w = W[i];
		hw = h_W[i];
		w = w + B * hw;
		W[i] = w;
	}

}

// LSRK wave Runge-Kutta
void newRk( const GRID * const grid, FLOAT * const h_W, FLOAT * const W, const float B, const float DT )
{
	long long num = grid->_nx_ * grid->_ny_ * grid->_nz_ * WAVESIZE;

	dim3 threads( 1024, 1, 1 );
	dim3 blocks;
	blocks.x = ( num + threads.x - 1 ) / threads.x;
	blocks.y = 1;
	blocks.z = 1;
	new_wave_rk <<< blocks, threads >>> ( h_W, W, num, B, DT );
	checkCudaErrors( cudaDeviceSynchronize() );

}