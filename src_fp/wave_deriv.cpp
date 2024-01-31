#include "header.h"

#ifdef PML
#define TIMES_PML_BETA_X * pml_beta_x
#define TIMES_PML_BETA_Y * pml_beta_y
#define TIMES_PML_BETA_Z * pml_beta_z
#else
#define TIMES_PML_BETA_X 
#define TIMES_PML_BETA_Y 
#define TIMES_PML_BETA_Z 
#endif

// Allocate device FLOAT memory for wave filed wave
void allocWave( const GRID * const grid, WAVE * h_W, WAVE * W, WAVE * t_W, WAVE * m_W )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	//printf( "_nx_ = %d, _ny_ = %d, _nz_ = %d\n", _nx_, _ny_, _nz_  );
	long long num = _nx_ * _ny_ * _nz_; 
		
	FLOAT * pWave = NULL;
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;

	checkCudaErrors( cudaMalloc((void **)&pWave, size) );
	checkCudaErrors( cudaMemset(pWave, 0, size) ); 

	h_W->Vx  = pWave + 0 * num;
	h_W->Vy  = pWave + 1 * num;
	h_W->Vz  = pWave + 2 * num;
	h_W->Txx = pWave + 3 * num;
	h_W->Tyy = pWave + 4 * num;
	h_W->Tzz = pWave + 5 * num;
	h_W->Txy = pWave + 6 * num;
	h_W->Txz = pWave + 7 * num;
	h_W->Tyz = pWave + 8 * num;

	pWave 	 = pWave + 9 * num;
	
	W->Vx    = pWave + 0 * num;
	W->Vy    = pWave + 1 * num;
	W->Vz    = pWave + 2 * num;
	W->Txx   = pWave + 3 * num;
	W->Tyy   = pWave + 4 * num;
	W->Tzz   = pWave + 5 * num;
	W->Txy   = pWave + 6 * num;
	W->Txz   = pWave + 7 * num;
	W->Tyz   = pWave + 8 * num;

	pWave    = pWave + 9 * num;

	t_W->Vx  = pWave + 0 * num;
	t_W->Vy  = pWave + 1 * num;
	t_W->Vz  = pWave + 2 * num;
	t_W->Txx = pWave + 3 * num;
	t_W->Tyy = pWave + 4 * num;
	t_W->Tzz = pWave + 5 * num;
	t_W->Txy = pWave + 6 * num;
	t_W->Txz = pWave + 7 * num;
	t_W->Tyz = pWave + 8 * num;

	pWave 	 = pWave + 9 * num;

	m_W->Vx  = pWave + 0 * num;
	m_W->Vy  = pWave + 1 * num;
	m_W->Vz  = pWave + 2 * num;
	m_W->Txx = pWave + 3 * num;
	m_W->Tyy = pWave + 4 * num;
	m_W->Tzz = pWave + 5 * num;
	m_W->Txy = pWave + 6 * num;
	m_W->Txz = pWave + 7 * num;
	m_W->Tyz = pWave + 8 * num;
//	printf( "Alloc: h_W = %p,", h_W->Vx );
//	printf( " W = %p,",			  W->Vx );
//	printf( " t_W = %p,",		t_W->Vx );
//	printf( " m_W = %p\n",		m_W->Vx );

}


void freeWave( WAVE h_W, WAVE W, WAVE t_W, WAVE m_W )
{
	checkCudaErrors( cudaFree(h_W.Vx) );
}

															
__global__ void wave_deriv( const WAVE h_W, const WAVE W,
							const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
#ifdef PML
							const PML_BETA_FLOAT pml_beta,
#endif
							const int _nx_, const int _ny_, const int _nz_, const float rDH, const int FB1,
							const int FB2, const int FB3 )
{
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO; 
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + HALO;

	long long index;

#ifdef PML
	FLOAT pml_beta_x;
	FLOAT pml_beta_y;
	FLOAT pml_beta_z;
#endif

	FLOAT mu;
	FLOAT lambda;
	FLOAT buoyancy;

	FLOAT xi_x;		FLOAT xi_y;		FLOAT xi_z;
	FLOAT et_x;		FLOAT et_y;		FLOAT et_z;
	FLOAT zt_x;		FLOAT zt_y;		FLOAT zt_z;
	
	FLOAT Txx_xi;	FLOAT Tyy_xi;	FLOAT Txy_xi;
	FLOAT Txx_et;	FLOAT Tyy_et;	FLOAT Txy_et;
	FLOAT Txx_zt;	FLOAT Tyy_zt;	FLOAT Txy_zt;

	FLOAT Txz_xi;	FLOAT Tyz_xi;	FLOAT Tzz_xi;
	FLOAT Txz_et;	FLOAT Tyz_et;	FLOAT Tzz_et;
	FLOAT Txz_zt;	FLOAT Tyz_zt;	FLOAT Tzz_zt;

	FLOAT Vx_xi;	FLOAT Vx_et;	FLOAT Vx_zt;
	FLOAT Vy_xi;	FLOAT Vy_et;	FLOAT Vy_zt;
	FLOAT Vz_xi;	FLOAT Vz_et;	FLOAT Vz_zt;

	FLOAT Vx1;		FLOAT Vx2;		FLOAT Vx3;
	FLOAT Vy1;		FLOAT Vy2;		FLOAT Vy3;
	FLOAT Vz1;		FLOAT Vz2;		FLOAT Vz3;
	FLOAT Txx1;		FLOAT Txx2;		FLOAT Txx3;
	FLOAT Tyy1;		FLOAT Tyy2;		FLOAT Tyy3;
	FLOAT Tzz1;		FLOAT Tzz2;		FLOAT Tzz3;
	FLOAT Txy1;		FLOAT Txy2;		FLOAT Txy3;
	FLOAT Txz1;		FLOAT Txz2;		FLOAT Txz3;
	FLOAT Tyz1;		FLOAT Tyz2;		FLOAT Tyz3;

	if ( i >= HALO && i < _nx && j >= HALO && j < _ny && k >= HALO && k < _nz )
	{
		index = INDEX( i, j, k );
		mu       = medium.mu[index];
		lambda   = medium.lambda[index];
		buoyancy = medium.buoyancy[index];
#ifdef PML
		pml_beta_x = pml_beta.x[i];
		pml_beta_y = pml_beta.y[j];
		pml_beta_z = pml_beta.z[k];
#endif
		xi_x = con.xi_x[index];
		xi_y = con.xi_y[index];
		xi_z = con.xi_z[index];
		et_x = con.et_x[index];
		et_y = con.et_y[index];
		et_z = con.et_z[index];
		zt_x = con.zt_x[index];
		zt_y = con.zt_y[index];
		zt_z = con.zt_z[index];

		 Vx_xi = MacCormack( W.Vx , FB1, xi ) TIMES_PML_BETA_X;
		 Vy_xi = MacCormack( W.Vy , FB1, xi ) TIMES_PML_BETA_X;
		 Vz_xi = MacCormack( W.Vz , FB1, xi ) TIMES_PML_BETA_X;
		Txx_xi = MacCormack( W.Txx, FB1, xi ) TIMES_PML_BETA_X;
		Tyy_xi = MacCormack( W.Tyy, FB1, xi ) TIMES_PML_BETA_X;
		Tzz_xi = MacCormack( W.Tzz, FB1, xi ) TIMES_PML_BETA_X;
		Txy_xi = MacCormack( W.Txy, FB1, xi ) TIMES_PML_BETA_X;
		Txz_xi = MacCormack( W.Txz, FB1, xi ) TIMES_PML_BETA_X;
		Tyz_xi = MacCormack( W.Tyz, FB1, xi ) TIMES_PML_BETA_X;
		 Vx_et = MacCormack( W.Vx , FB2, et ) TIMES_PML_BETA_Y;
		 Vy_et = MacCormack( W.Vy , FB2, et ) TIMES_PML_BETA_Y;
		 Vz_et = MacCormack( W.Vz , FB2, et ) TIMES_PML_BETA_Y;
		Txx_et = MacCormack( W.Txx, FB2, et ) TIMES_PML_BETA_Y;
		Tyy_et = MacCormack( W.Tyy, FB2, et ) TIMES_PML_BETA_Y;
		Tzz_et = MacCormack( W.Tzz, FB2, et ) TIMES_PML_BETA_Y;
		Txy_et = MacCormack( W.Txy, FB2, et ) TIMES_PML_BETA_Y;
		Txz_et = MacCormack( W.Txz, FB2, et ) TIMES_PML_BETA_Y;
		Tyz_et = MacCormack( W.Tyz, FB2, et ) TIMES_PML_BETA_Y;
   		 Vx_zt = MacCormack( W.Vx , FB3, zt ) TIMES_PML_BETA_Z;
   		 Vy_zt = MacCormack( W.Vy , FB3, zt ) TIMES_PML_BETA_Z;
   		 Vz_zt = MacCormack( W.Vz , FB3, zt ) TIMES_PML_BETA_Z;
  		Txx_zt = MacCormack( W.Txx, FB3, zt ) TIMES_PML_BETA_Z;
  		Tyy_zt = MacCormack( W.Tyy, FB3, zt ) TIMES_PML_BETA_Z;
  		Tzz_zt = MacCormack( W.Tzz, FB3, zt ) TIMES_PML_BETA_Z;
  		Txy_zt = MacCormack( W.Txy, FB3, zt ) TIMES_PML_BETA_Z;
  		Txz_zt = MacCormack( W.Txz, FB3, zt ) TIMES_PML_BETA_Z;
  		Tyz_zt = MacCormack( W.Tyz, FB3, zt ) TIMES_PML_BETA_Z;

		Vx1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txx_xi, Txy_xi, Txz_xi ) * buoyancy;
		Vx2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txx_et, Txy_et, Txz_et ) * buoyancy;
		Vx3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txx_zt, Txy_zt, Txz_zt ) * buoyancy;
		Vy1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txy_xi, Tyy_xi, Tyz_xi ) * buoyancy;
		Vy2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txy_et, Tyy_et, Tyz_et ) * buoyancy;
		Vy3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txy_zt, Tyy_zt, Tyz_zt ) * buoyancy;
		Vz1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txz_xi, Tyz_xi, Tzz_xi ) * buoyancy;
		Vz2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txz_et, Tyz_et, Tzz_et ) * buoyancy;
		Vz3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txz_zt, Tyz_zt, Tzz_zt ) * buoyancy;

		Txx1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_x * Vx_xi );
		Txx2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_x * Vx_et );
		Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
		Tyy1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_y * Vy_xi );
		Tyy2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_y * Vy_et );
		Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
		Tzz1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_z * Vz_xi );
		Tzz2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_z * Vz_et );
		Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

		Txy1 = DOT_PRODUCT2D( xi_y, xi_x, Vx_xi, Vy_xi ) * mu;
		Txy2 = DOT_PRODUCT2D( et_y, et_x, Vx_et, Vy_et ) * mu;
		Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
		Txz1 = DOT_PRODUCT2D( xi_z, xi_x, Vx_xi, Vz_xi ) * mu;
		Txz2 = DOT_PRODUCT2D( et_z, et_x, Vx_et, Vz_et ) * mu;
		Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
		Tyz1 = DOT_PRODUCT2D( xi_z, xi_y, Vy_xi, Vz_xi ) * mu;
		Tyz2 = DOT_PRODUCT2D( et_z, et_y, Vy_et, Vz_et ) * mu;
		Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;						

		h_W.Vx [index] = Vx1  + Vx2  + Vx3 ;
		h_W.Vy [index] = Vy1  + Vy2  + Vy3 ;
		h_W.Vz [index] = Vz1  + Vz2  + Vz3 ;
		h_W.Txx[index] = Txx1 + Txx2 + Txx3;
		h_W.Tyy[index] = Tyy1 + Tyy2 + Tyy3;
		h_W.Tzz[index] = Tzz1 + Tzz2 + Tzz3;
		h_W.Txy[index] = Txy1 + Txy2 + Txy3;
		h_W.Txz[index] = Txz1 + Txz2 + Txz3;
		h_W.Tyz[index] = Tyz1 + Tyz2 + Tyz3;
	}

}			


// wave propagate derive
void waveDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
				const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
#ifdef PML
				const PML_BETA_FLOAT pml_beta,
#endif
				const int irk, const int FB1, const int FB2, const int FB3,
				const float DT, const int IsFreeSurface )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
#ifdef FREE_SURFACE
	if ( IsFreeSurface )
		_nz_ = _nz_ - HALO;
#endif

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;
	float rDH = grid->rDH;

	dim3 threads( 32, 4, 4);
	dim3 blocks;
	blocks.x = ( nx + threads.x - 1 ) / threads.x;
	blocks.y = ( ny + threads.y - 1 ) / threads.y;
	blocks.z = ( nz + threads.z - 1 ) / threads.z;

	wave_deriv <<< blocks, threads >>> 
	( h_W, W, con, medium,
#ifdef PML
	  pml_beta,
#endif
	  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3 );


}

