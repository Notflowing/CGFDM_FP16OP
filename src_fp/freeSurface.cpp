#include "header.h"

#ifdef PML
#define times_pml_beta_x * pml_beta_x
#define times_pml_beta_y * pml_beta_y
#define times_pml_beta_z * pml_beta_z
#else
#define times_pml_beta_x 
#define times_pml_beta_y 
#define times_pml_beta_z 
#endif


__global__ void free_surface_deriv( const WAVE h_W, const WAVE W, const CONTRAVARIANT_FLOAT con,
									const MEDIUM_FLOAT medium, const FLOAT * Jac,
									const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY, 
#ifdef PML
									const PML_BETA_FLOAT pml_beta,
#endif
									const int _nx_, const int _ny_, const int _nz_, const float rDH,
									const int FB1, const int FB2, const int FB3 )
{																	
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO;
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + _nz - HALO;

	long long index;

	FLOAT mu;				
	FLOAT lambda;		
	FLOAT buoyancy;

#ifdef PML
	FLOAT pml_beta_x;
	FLOAT pml_beta_y;
#endif
	
	FLOAT xi_x;		FLOAT xi_y; 	FLOAT xi_z;
	FLOAT et_x; 	FLOAT et_y; 	FLOAT et_z;
	FLOAT zt_x; 	FLOAT zt_y; 	FLOAT zt_z;

	FLOAT Vx_xi;	FLOAT Vx_et;	FLOAT Vx_zt;
	FLOAT Vy_xi;	FLOAT Vy_et;	FLOAT Vy_zt;
	FLOAT Vz_xi;	FLOAT Vz_et;	FLOAT Vz_zt;

	FLOAT Jinv;
	FLOAT jacb;
	FLOAT J_T1x[7];	FLOAT J_T2x[7];	FLOAT J_T3x[7];
	FLOAT J_T1y[7];	FLOAT J_T2y[7];	FLOAT J_T3y[7];
	FLOAT J_T1z[7];	FLOAT J_T2z[7];	FLOAT J_T3z[7];

	FLOAT Vx1;		FLOAT Vx2;		FLOAT Vx3;
	FLOAT Vy1;		FLOAT Vy2;		FLOAT Vy3;
	FLOAT Vz1;		FLOAT Vz2;		FLOAT Vz3;

	FLOAT Txx1;		FLOAT Txx2;		FLOAT Txx3;
	FLOAT Tyy1;		FLOAT Tyy2;		FLOAT Tyy3;
	FLOAT Tzz1;		FLOAT Tzz2;		FLOAT Tzz3;
	FLOAT Txy1;		FLOAT Txy2;		FLOAT Txy3;
	FLOAT Txz1;		FLOAT Txz2;		FLOAT Txz3;
	FLOAT Tyz1;		FLOAT Tyz2;		FLOAT Tyz3;

	// FLOAT h_WVx ;
	// FLOAT h_WVy ;
	// FLOAT h_WVz ;
	// FLOAT h_WTxx;
	// FLOAT h_WTyy;
	// FLOAT h_WTzz;
	// FLOAT h_WTxy;
	// FLOAT h_WTxz;
	// FLOAT h_WTyz;

	int l = 0;
	int k_s = 0; /*relative index on the surface*/
	int pos = 0;
	
	int indexOnSurf;
	if ( i >= HALO && i < _nx && j >= HALO && j < _ny && k >= (_nz - HALO) && k < _nz )
	{
		index = INDEX( i, j, k );
		mu       = medium.mu[index];
		lambda   = medium.lambda[index];
		buoyancy = medium.buoyancy[index];

#ifdef PML
		pml_beta_x = pml_beta.x[i];
		pml_beta_y = pml_beta.y[j];
#endif
		k_s = ( _nz - 1 ) - k + HALO; /*relative index on the surface*/
		for ( l = 0; l <= ( 2 * HALO ); l ++ )
		{
			pos = INDEX( i + ( l - HALO ), j, k );
			xi_x = con.xi_x[pos];		xi_y = con.xi_y[pos];		xi_z = con.xi_z[pos];
			jacb = Jac[pos];
			J_T1x[l] = ( xi_x * W.Txx[pos] + xi_y * W.Txy[pos] + xi_z * W.Txz[pos] ) * jacb;
			J_T2x[l] = ( xi_x * W.Txy[pos] + xi_y * W.Tyy[pos] + xi_z * W.Tyz[pos] ) * jacb;
			J_T3x[l] = ( xi_x * W.Txz[pos] + xi_y * W.Tyz[pos] + xi_z * W.Tzz[pos] ) * jacb;

			pos = INDEX( i, j + ( l - HALO ), k );
			et_x = con.et_x[pos];		et_y = con.et_y[pos];		et_z = con.et_z[pos];
			jacb = Jac[pos];
			J_T1y[l] = ( et_x * W.Txx[pos] + et_y * W.Txy[pos] + et_z * W.Txz[pos] ) * jacb;
			J_T2y[l] = ( et_x * W.Txy[pos] + et_y * W.Tyy[pos] + et_z * W.Tyz[pos] ) * jacb;
			J_T3y[l] = ( et_x * W.Txz[pos] + et_y * W.Tyz[pos] + et_z * W.Tzz[pos] ) * jacb;
		}
		for ( l = 0; l < k_s; l ++ )
		{
			pos = INDEX( i, j, k + ( l - HALO ) );
			zt_x = con.zt_x[pos]; 		zt_y = con.zt_y[pos]; 		zt_z = con.zt_z[pos];
			jacb = Jac[pos];
			J_T1z[l] = ( zt_x * W.Txx[pos] + zt_y * W.Txy[pos] + zt_z * W.Txz[pos] ) * jacb;
			J_T2z[l] = ( zt_x * W.Txy[pos] + zt_y * W.Tyy[pos] + zt_z * W.Tyz[pos] ) * jacb;
			J_T3z[l] = ( zt_x * W.Txz[pos] + zt_y * W.Tyz[pos] + zt_z * W.Tzz[pos] ) * jacb;
		}
		/*The T on the surface is 0.*/
		J_T1z[k_s] = 0.0f;
		J_T2z[k_s] = 0.0f;
		J_T3z[k_s] = 0.0f;
		for ( l = k_s + 1; l <= 2 * HALO; l ++ )	
		{									
			J_T1z[l] = - J_T1z[2 * k_s - l];
			J_T2z[l] = - J_T2z[2 * k_s - l];
			J_T3z[l] = - J_T3z[2 * k_s - l];
		}									
		jacb = Jac[index];
		Jinv = 1.0f / jacb;
		
		h_W.Vx [index] = buoyancy * Jinv * ( MacCormack_freesurf( J_T1x, FB1 ) times_pml_beta_x
								  		   + MacCormack_freesurf( J_T1y, FB2 ) times_pml_beta_y
								  		   + MacCormack_freesurf( J_T1z, FB3 ) );
		h_W.Vy [index] = buoyancy * Jinv * ( MacCormack_freesurf( J_T2x, FB1 ) times_pml_beta_x
								  		   + MacCormack_freesurf( J_T2y, FB2 ) times_pml_beta_y
								  		   + MacCormack_freesurf( J_T2z, FB3 ) );
		h_W.Vz [index] = buoyancy * Jinv * ( MacCormack_freesurf( J_T3x, FB1 ) times_pml_beta_x
								  		   + MacCormack_freesurf( J_T3y, FB2 ) times_pml_beta_y
								  		   + MacCormack_freesurf( J_T3z, FB3 ) );

		Vx_xi = MacCormack( W.Vx, FB1, xi ) times_pml_beta_x;	Vx_et = MacCormack( W.Vx, FB2, et ) times_pml_beta_y;
		Vy_xi = MacCormack( W.Vy, FB1, xi ) times_pml_beta_x;	Vy_et = MacCormack( W.Vy, FB2, et ) times_pml_beta_y;
		Vz_xi = MacCormack( W.Vz, FB1, xi ) times_pml_beta_x;	Vz_et = MacCormack( W.Vz, FB2, et ) times_pml_beta_y;

		xi_x = con.xi_x[index];		xi_y = con.xi_y[index]; 	xi_z = con.xi_z[index];
		et_x = con.et_x[index]; 	et_y = con.et_y[index]; 	et_z = con.et_z[index];
		zt_x = con.zt_x[index]; 	zt_y = con.zt_y[index]; 	zt_z = con.zt_z[index];

//=======================================================
//When change the HALO, BE CAREFUL!!!!
//=======================================================
											
		if ( k == _nz - 1 )					
		{ 									
			indexOnSurf = INDEX( i, j, 0 );	
			Vx_zt = DOT_PRODUCT3D( _rDZ_DX.M11[indexOnSurf], _rDZ_DX.M12[indexOnSurf], _rDZ_DX.M13[indexOnSurf], Vx_xi, Vy_xi, Vz_xi ) 	
				  + DOT_PRODUCT3D( _rDZ_DY.M11[indexOnSurf], _rDZ_DY.M12[indexOnSurf], _rDZ_DY.M13[indexOnSurf], Vx_et, Vy_et, Vz_et );	
			Vy_zt = DOT_PRODUCT3D( _rDZ_DX.M21[indexOnSurf], _rDZ_DX.M22[indexOnSurf], _rDZ_DX.M23[indexOnSurf], Vx_xi, Vy_xi, Vz_xi ) 	
				  + DOT_PRODUCT3D( _rDZ_DY.M21[indexOnSurf], _rDZ_DY.M22[indexOnSurf], _rDZ_DY.M23[indexOnSurf], Vx_et, Vy_et, Vz_et );	
			Vz_zt = DOT_PRODUCT3D( _rDZ_DX.M31[indexOnSurf], _rDZ_DX.M32[indexOnSurf], _rDZ_DX.M33[indexOnSurf], Vx_xi, Vy_xi, Vz_xi ) 	
				  + DOT_PRODUCT3D( _rDZ_DY.M31[indexOnSurf], _rDZ_DY.M32[indexOnSurf], _rDZ_DY.M33[indexOnSurf], Vx_et, Vy_et, Vz_et );	
		} 								
		if ( k == _nz - 2 ) 				
		{								
			Vx_zt =	L2( W.Vx, FB3, zt );
			Vy_zt =	L2( W.Vy, FB3, zt );
			Vz_zt =	L2( W.Vz, FB3, zt );
		}								
		if ( k == _nz - 3 ) 				
		{								
			Vx_zt =	L3( W.Vx, FB3, zt );
			Vy_zt =	L3( W.Vy, FB3, zt );
			Vz_zt =	L3( W.Vz, FB3, zt );
		}								
		
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

		h_W.Txx[index] 	= Txx1 + Txx2 + Txx3;
		h_W.Tyy[index] 	= Tyy1 + Tyy2 + Tyy3;
		h_W.Tzz[index] 	= Tzz1 + Tzz2 + Tzz3;
		h_W.Txy[index] 	= Txy1 + Txy2 + Txy3;
		h_W.Txz[index] 	= Txz1 + Txz2 + Txz3;
		h_W.Tyz[index] 	= Tyz1 + Tyz2 + Tyz3;
	}

}

// Free surface derive using traction imaging method
void freeSurfaceDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
					   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
					   const FLOAT * Jac, const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,
#ifdef PML
					   PML_BETA_FLOAT pml_beta,
#endif
					   const int irk, const int FB1, const int FB2, const int FB3, const float DT )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	float rDH = grid->rDH;

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;

	dim3 threads( 32, 4, 1);
	dim3 blocks;
	
	blocks.x = ( nx + threads.x - 1 ) / threads.x;
	blocks.y = ( ny + threads.y - 1 ) / threads.y;
	blocks.z = HALO / threads.z;

	free_surface_deriv <<< blocks, threads >>>
	( h_W, W, con, medium, Jac, _rDZ_DX, _rDZ_DY, 
#ifdef PML
	  pml_beta,
#endif
	  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3 );

}

