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
	half2 *h_W_half2_Vx  = (half2 *)h_W.Vx;
	half2 *h_W_half2_Vy  = (half2 *)h_W.Vy;
	half2 *h_W_half2_Vz  = (half2 *)h_W.Vz;
	half2 *h_W_half2_Txx = (half2 *)h_W.Txx;
	half2 *h_W_half2_Tyy = (half2 *)h_W.Tyy;
	half2 *h_W_half2_Tzz = (half2 *)h_W.Tzz;
	half2 *h_W_half2_Txy = (half2 *)h_W.Txy;
	half2 *h_W_half2_Txz = (half2 *)h_W.Txz;
	half2 *h_W_half2_Tyz = (half2 *)h_W.Tyz;

	half2 *W_half2_Vx  = (half2 *)W.Vx;
	half2 *W_half2_Vy  = (half2 *)W.Vy;
	half2 *W_half2_Vz  = (half2 *)W.Vz;
	half2 *W_half2_Txx = (half2 *)W.Txx;
	half2 *W_half2_Tyy = (half2 *)W.Tyy;
	half2 *W_half2_Tzz = (half2 *)W.Tzz;
	half2 *W_half2_Txy = (half2 *)W.Txy;
	half2 *W_half2_Txz = (half2 *)W.Txz;
	half2 *W_half2_Tyz = (half2 *)W.Tyz;

	half2 *con_half2_xi_x = (half2 *)con.xi_x;
	half2 *con_half2_xi_y = (half2 *)con.xi_y;
	half2 *con_half2_xi_z = (half2 *)con.xi_z;
	half2 *con_half2_et_x = (half2 *)con.et_x;
	half2 *con_half2_et_y = (half2 *)con.et_y;
	half2 *con_half2_et_z = (half2 *)con.et_z;
	half2 *con_half2_zt_x = (half2 *)con.zt_x;
	half2 *con_half2_zt_y = (half2 *)con.zt_y;
	half2 *con_half2_zt_z = (half2 *)con.zt_z;

	half2 *medium_half2_mu       = (half2 *)medium.mu;
	half2 *medium_half2_lambda   = (half2 *)medium.lambda;
	half2 *medium_half2_buoyancy = (half2 *)medium.buoyancy;
	
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	int nx = _nx - HALO;
	int _nx_2 = _nx_ / 2;

	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO / 2;
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + HALO;

	long long index;

#ifdef PML
	half2 pml_beta_x;
	half2 pml_beta_y;
	half2 pml_beta_z;
#endif

	half2 mu;
	half2 lambda;
	half2 buoyancy;

	half2 xi_x;		half2 xi_y;		half2 xi_z;
	half2 et_x;		half2 et_y;		half2 et_z;
	half2 zt_x;		half2 zt_y;		half2 zt_z;
	
	half2 Txx_xi;	half2 Tyy_xi;	half2 Txy_xi;
	half2 Txx_et;	half2 Tyy_et;	half2 Txy_et;
	half2 Txx_zt;	half2 Tyy_zt;	half2 Txy_zt;

	half2 Txz_xi;	half2 Tyz_xi;	half2 Tzz_xi;
	half2 Txz_et;	half2 Tyz_et;	half2 Tzz_et;
	half2 Txz_zt;	half2 Tyz_zt;	half2 Tzz_zt;

	half2 Vx_xi;	half2 Vx_et;	half2 Vx_zt;
	half2 Vy_xi;	half2 Vy_et;	half2 Vy_zt;
	half2 Vz_xi;	half2 Vz_et;	half2 Vz_zt;

	half2 Vx1;		half2 Vx2;		half2 Vx3;
	half2 Vy1;		half2 Vy2;		half2 Vy3;
	half2 Vz1;		half2 Vz2;		half2 Vz3;
	half2 Txx1;		half2 Txx2;		half2 Txx3;
	half2 Tyy1;		half2 Tyy2;		half2 Tyy3;
	half2 Tzz1;		half2 Tzz2;		half2 Tzz3;
	half2 Txy1;		half2 Txy2;		half2 Txy3;
	half2 Txz1;		half2 Txz2;		half2 Txz3;
	half2 Tyz1;		half2 Tyz2;		half2 Tyz3;


	if ( i >= HALO / 2 && i <= (nx / 2 + 1) && j >= HALO && j < _ny && k >= HALO && k < _nz )
	{
		index = INDEX_HALF2( i, j, k );
		mu       = medium_half2_mu[index];
		lambda   = medium_half2_lambda[index];
		buoyancy = medium_half2_buoyancy[index];
#ifdef PML
		pml_beta_x = ((half2 *)pml_beta.x)[i];
		pml_beta_y = __half2half2(pml_beta.y[j]);
		pml_beta_z = __half2half2(pml_beta.z[k]);
#endif
		xi_x = con_half2_xi_x[index];
		xi_y = con_half2_xi_y[index];
		xi_z = con_half2_xi_z[index];
		et_x = con_half2_et_x[index];
		et_y = con_half2_et_y[index];
		et_z = con_half2_et_z[index];
		zt_x = con_half2_zt_x[index];
		zt_y = con_half2_zt_y[index];
		zt_z = con_half2_zt_z[index];

		if (i == HALO / 2)
		{
			 Vx_xi.x = 0.0f;
			 Vy_xi.x = 0.0f;
			 Vz_xi.x = 0.0f;
			Txx_xi.x = 0.0f;
			Tyy_xi.x = 0.0f;
			Tzz_xi.x = 0.0f;
			Txy_xi.x = 0.0f;
			Txz_xi.x = 0.0f;
			Tyz_xi.x = 0.0f;
		}
		else
		{
			 Vx_xi.x = MacCormack_X_x(W.Vx , FB1);
			 Vy_xi.x = MacCormack_X_x(W.Vy , FB1);
			 Vz_xi.x = MacCormack_X_x(W.Vz , FB1);
			Txx_xi.x = MacCormack_X_x(W.Txx, FB1);
			Tyy_xi.x = MacCormack_X_x(W.Tyy, FB1);
			Tzz_xi.x = MacCormack_X_x(W.Tzz, FB1);
			Txy_xi.x = MacCormack_X_x(W.Txy, FB1);
			Txz_xi.x = MacCormack_X_x(W.Txz, FB1);
			Tyz_xi.x = MacCormack_X_x(W.Tyz, FB1);
		}

		if (i == (nx / 2 + 1))
		{
			 Vx_xi.y = 0.0f;
			 Vy_xi.y = 0.0f;
			 Vz_xi.y = 0.0f;
			Txx_xi.y = 0.0f;
			Tyy_xi.y = 0.0f;
			Tzz_xi.y = 0.0f;
			Txy_xi.y = 0.0f;
			Txz_xi.y = 0.0f;
			Tyz_xi.y = 0.0f;
		}
		else
		{
			 Vx_xi.y = MacCormack_X_y(W.Vx , FB1);
			 Vy_xi.y = MacCormack_X_y(W.Vy , FB1);
			 Vz_xi.y = MacCormack_X_y(W.Vz , FB1);
			Txx_xi.y = MacCormack_X_y(W.Txx, FB1);
			Tyy_xi.y = MacCormack_X_y(W.Tyy, FB1);
			Tzz_xi.y = MacCormack_X_y(W.Tzz, FB1);
			Txy_xi.y = MacCormack_X_y(W.Txy, FB1);
			Txz_xi.y = MacCormack_X_y(W.Txz, FB1);
			Tyz_xi.y = MacCormack_X_y(W.Tyz, FB1);
		}

		 Vx_xi =  Vx_xi TIMES_PML_BETA_X;
		 Vy_xi =  Vy_xi TIMES_PML_BETA_X;
		 Vz_xi =  Vz_xi TIMES_PML_BETA_X;
		Txx_xi = Txx_xi TIMES_PML_BETA_X;
		Tyy_xi = Tyy_xi TIMES_PML_BETA_X;
		Tzz_xi = Tzz_xi TIMES_PML_BETA_X;
		Txy_xi = Txy_xi TIMES_PML_BETA_X;
		Txz_xi = Txz_xi TIMES_PML_BETA_X;
		Tyz_xi = Tyz_xi TIMES_PML_BETA_X;
		
		 Vx_et = MacCormack_HALF2( W_half2_Vx , FB2, et ) TIMES_PML_BETA_Y;
		 Vy_et = MacCormack_HALF2( W_half2_Vy , FB2, et ) TIMES_PML_BETA_Y;
		 Vz_et = MacCormack_HALF2( W_half2_Vz , FB2, et ) TIMES_PML_BETA_Y;
		Txx_et = MacCormack_HALF2( W_half2_Txx, FB2, et ) TIMES_PML_BETA_Y;
		Tyy_et = MacCormack_HALF2( W_half2_Tyy, FB2, et ) TIMES_PML_BETA_Y;
		Tzz_et = MacCormack_HALF2( W_half2_Tzz, FB2, et ) TIMES_PML_BETA_Y;
		Txy_et = MacCormack_HALF2( W_half2_Txy, FB2, et ) TIMES_PML_BETA_Y;
		Txz_et = MacCormack_HALF2( W_half2_Txz, FB2, et ) TIMES_PML_BETA_Y;
		Tyz_et = MacCormack_HALF2( W_half2_Tyz, FB2, et ) TIMES_PML_BETA_Y;

   		 Vx_zt = MacCormack_HALF2( W_half2_Vx , FB3, zt ) TIMES_PML_BETA_Z;
   		 Vy_zt = MacCormack_HALF2( W_half2_Vy , FB3, zt ) TIMES_PML_BETA_Z;
   		 Vz_zt = MacCormack_HALF2( W_half2_Vz , FB3, zt ) TIMES_PML_BETA_Z;
  		Txx_zt = MacCormack_HALF2( W_half2_Txx, FB3, zt ) TIMES_PML_BETA_Z;
  		Tyy_zt = MacCormack_HALF2( W_half2_Tyy, FB3, zt ) TIMES_PML_BETA_Z;
  		Tzz_zt = MacCormack_HALF2( W_half2_Tzz, FB3, zt ) TIMES_PML_BETA_Z;
  		Txy_zt = MacCormack_HALF2( W_half2_Txy, FB3, zt ) TIMES_PML_BETA_Z;
  		Txz_zt = MacCormack_HALF2( W_half2_Txz, FB3, zt ) TIMES_PML_BETA_Z;
  		Tyz_zt = MacCormack_HALF2( W_half2_Tyz, FB3, zt ) TIMES_PML_BETA_Z;

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

		h_W_half2_Vx [index] = Vx1  + Vx2  + Vx3 ;
		h_W_half2_Vy [index] = Vy1  + Vy2  + Vy3 ;
		h_W_half2_Vz [index] = Vz1  + Vz2  + Vz3 ;
		h_W_half2_Txx[index] = Txx1 + Txx2 + Txx3;
		h_W_half2_Tyy[index] = Tyy1 + Tyy2 + Tyy3;
		h_W_half2_Tzz[index] = Tzz1 + Tzz2 + Tzz3;
		h_W_half2_Txy[index] = Txy1 + Txy2 + Txy3;
		h_W_half2_Txz[index] = Txz1 + Txz2 + Txz3;
		h_W_half2_Tyz[index] = Tyz1 + Tyz2 + Tyz3;
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
	blocks.x = ( nx / 2 + 1 + threads.x - 1 ) / threads.x;
	blocks.y = ( ny + threads.y - 1 ) / threads.y;
	blocks.z = ( nz + threads.z - 1 ) / threads.z;

	wave_deriv <<< blocks, threads >>> 
	( h_W, W, con, medium,
#ifdef PML
	  pml_beta,
#endif
	  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3 );
	// checkCudaErrors( cudaDeviceSynchronize() );

}

