#include "header.h"


typedef void (*WAVE_RK_FUNC )( half2 * h_W, half2 * W, half2 * t_W, half2 * m_W, long long num );


__global__ void wave_pml_rk0( half2 * h_W, half2 * W, half2 * t_W, half2 * m_W, long long WStride )
{

	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	if ( i >= 0 && i < WStride )
	{
		//if( i == WStride )
		//	printf( "WStride = %d\n", WStride );
		m_W[i] = W[i];
		t_W[i] = m_W[i] + beta1  * h_W[i];
		  W[i] = m_W[i] + alpha2 * h_W[i];
	}
}


__global__ void wave_pml_rk1( half2 * h_W, half2 * W, half2 * t_W, half2 * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	if ( i >= 0 && i < WStride )
	{
		t_W[i] = t_W[i] + beta2  * h_W[i];
		  W[i] = m_W[i] + alpha3 * h_W[i];
	}
}


__global__ void wave_pml_rk2( half2 * h_W, half2 * W, half2 * t_W, half2 * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	if ( i >= 0 && i < WStride )
	{
		t_W[i] = t_W[i] + beta3 * h_W[i];
		  W[i] = m_W[i] + h_W[i];
	}
}


__global__ void wave_pml_rk3( half2 * h_W, half2 * W, half2 * t_W, half2 * m_W, long long WStride )
{
	long long i = threadIdx.x + blockIdx.x * blockDim.x;

	if ( i >= 0 && i < WStride )
	{
		W[i] = t_W[i] + beta4 * h_W[i];
	}
}


void pmlRk( const GRID * const grid, const MPI_BORDER border, const int irk, const AUX4 Aux4_1, const AUX4 Aux4_2 )
{
	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;
	
	int nPML = grid->nPML;

	long long num = 0;
	
	WAVE_RK_FUNC pml_rk[4] = { wave_pml_rk0, wave_pml_rk1, wave_pml_rk2, wave_pml_rk3 };
	long long numx = nPML * ny * nz * WAVESIZE / 2;
	long long numy = nPML * nx * nz * WAVESIZE / 2;
	long long numz = nPML * nx * ny * WAVESIZE / 2;

                        
	dim3 thread( 512, 1, 1 );
	dim3 blockX;

	blockX.x = ( numx + thread.x - 1 ) / thread.x;
	blockX.y = 1;
	blockX.z = 1;

	dim3 blockY;
	blockY.x = ( numy + thread.x - 1 ) / thread.x;
	blockY.y = 1;
	blockY.z = 1;

	dim3 blockZ;
	blockZ.x = ( numz + thread.x - 1 ) / thread.x;
	blockZ.y = 1;
	blockZ.z = 1;

	if ( border.isx1 )	pml_rk[irk]<<< blockX, thread >>>( (half2 *)Aux4_1.h_Aux_x.Vx, (half2 *)Aux4_1.Aux_x.Vx, (half2 *)Aux4_1.t_Aux_x.Vx, (half2 *)Aux4_1.m_Aux_x.Vx, numx );
	if ( border.isy1 )  pml_rk[irk]<<< blockY, thread >>>( (half2 *)Aux4_1.h_Aux_y.Vx, (half2 *)Aux4_1.Aux_y.Vx, (half2 *)Aux4_1.t_Aux_y.Vx, (half2 *)Aux4_1.m_Aux_y.Vx, numy );
	if ( border.isz1 )  pml_rk[irk]<<< blockZ, thread >>>( (half2 *)Aux4_1.h_Aux_z.Vx, (half2 *)Aux4_1.Aux_z.Vx, (half2 *)Aux4_1.t_Aux_z.Vx, (half2 *)Aux4_1.m_Aux_z.Vx, numz );

	if ( border.isx2 )	pml_rk[irk]<<< blockX, thread >>>( (half2 *)Aux4_2.h_Aux_x.Vx, (half2 *)Aux4_2.Aux_x.Vx, (half2 *)Aux4_2.t_Aux_x.Vx, (half2 *)Aux4_2.m_Aux_x.Vx, numx );
	if ( border.isy2 )  pml_rk[irk]<<< blockY, thread >>>( (half2 *)Aux4_2.h_Aux_y.Vx, (half2 *)Aux4_2.Aux_y.Vx, (half2 *)Aux4_2.t_Aux_y.Vx, (half2 *)Aux4_2.m_Aux_y.Vx, numy );

#ifndef FREE_SURFACE
	if ( border.isz2 )  pml_rk[irk]<<< blockZ, thread >>>( (half2 *)Aux4_2.h_Aux_z.Vx, (half2 *)Aux4_2.Aux_z.Vx, (half2 *)Aux4_2.t_Aux_z.Vx, (half2 *)Aux4_2.m_Aux_z.Vx, numz );
#endif

}



__global__ void new_pml_rk( half2 * const h_W, half2 * const W, const long long WStride, const float B, const float DT )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if ( i >= 0 && i < WStride )
	{
		W[i]   = W[i] + B * h_W[i];
	}
}

// LSRK pml Runge-Kutta
void newPmlRk( const GRID * const grid, const MPI_BORDER border, const int irk,
			   const AUX4 Aux4_1, const AUX4 Aux4_2, const float B, const float DT )
{
	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;
	
	int nPML = grid->nPML;

	long long num = 0;
	long long numx = nPML * ny * nz * WAVESIZE / 2;
	long long numy = nPML * nx * nz * WAVESIZE / 2;
	long long numz = nPML * nx * ny * WAVESIZE / 2;
                        
	dim3 thread( 512, 1, 1 );
	dim3 blockX;

	blockX.x = ( numx + thread.x - 1 ) / thread.x;
	blockX.y = 1;
	blockX.z = 1;

	dim3 blockY;
	blockY.x = ( numy + thread.x - 1 ) / thread.x;
	blockY.y = 1;
	blockY.z = 1;

	dim3 blockZ;
	blockZ.x = ( numz + thread.x - 1 ) / thread.x;
	blockZ.y = 1;
	blockZ.z = 1;

	if ( border.isx1 )	new_pml_rk <<< blockX, thread >>> ( (half2 *)Aux4_1.h_Aux_x.Vx, (half2 *)Aux4_1.Aux_x.Vx, numx, B, DT );
	if ( border.isy1 )  new_pml_rk <<< blockY, thread >>> ( (half2 *)Aux4_1.h_Aux_y.Vx, (half2 *)Aux4_1.Aux_y.Vx, numy, B, DT );
	if ( border.isz1 )  new_pml_rk <<< blockZ, thread >>> ( (half2 *)Aux4_1.h_Aux_z.Vx, (half2 *)Aux4_1.Aux_z.Vx, numz, B, DT );
	if ( border.isx2 )	new_pml_rk <<< blockX, thread >>> ( (half2 *)Aux4_2.h_Aux_x.Vx, (half2 *)Aux4_2.Aux_x.Vx, numx, B, DT );
	if ( border.isy2 )  new_pml_rk <<< blockY, thread >>> ( (half2 *)Aux4_2.h_Aux_y.Vx, (half2 *)Aux4_2.Aux_y.Vx, numy, B, DT );
                                                                                                  
#ifndef FREE_SURFACE
	if ( border.isz2 )  new_pml_rk <<< blockZ, thread >>> ( (half2 *)Aux4_2.h_Aux_z.Vx, (half2 *)Aux4_2.Aux_z.Vx, numz, B, DT );
#endif

}
