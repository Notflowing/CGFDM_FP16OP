#include "header.h"


__global__ void pml_free_surface_x( const WAVE h_W, const WAVE W, const AUXILIARY h_Aux_x, const AUXILIARY Aux_x,			
									const FLOAT * ZT_X, const FLOAT * ZT_Y, const FLOAT * ZT_Z, MEDIUM_FLOAT medium,		
									const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,	const FLOAT * pml_d_x,
									const int nPML, const int _nx_, const int _ny_, const int _nz_,
									const int FLAG, const float rDH, const int FB1, const float DT )
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;

	int k0;
	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	int indexOnSurf;
	FLOAT mu;
	FLOAT lambda;

	FLOAT d_x;

	FLOAT Vx_xi;		FLOAT Vy_xi;		FLOAT Vz_xi;
	FLOAT Vx_zt;		FLOAT Vy_zt;		FLOAT Vz_zt;
	FLOAT zt_x ;		FLOAT zt_y ;		FLOAT zt_z ;

	FLOAT Txx3;
	FLOAT Tyy3;
	FLOAT Tzz3;
	FLOAT Txy3;
	FLOAT Txz3;
	FLOAT Tyz3;


	int stride = FLAG * ( nx - nPML );
	k0 = nz - 1;
	if ( i0 >= 0 && i0 < nPML && j0 >= 0 && j0 < ny )
	{
		i = i0 + HALO + stride;
		j = j0 + HALO;
		k = k0 + HALO;
		index = INDEX(i, j, k);
		indexOnSurf = INDEX( i, j, 0 );
		pos	= Index3D( i0, j0, k0, nPML, ny, nz );	//i0 + j0 * nPML + k0 * nPML * ny;

		mu     = medium.mu[index];
		lambda = medium.lambda[index];

		d_x = pml_d_x[i];
		zt_x = ZT_X[index];		zt_y = ZT_Y[index];		zt_z = ZT_Z[index];

		 Vx_xi = MacCormack( W.Vx, FB1, xi ) * d_x;
		 Vy_xi = MacCormack( W.Vy, FB1, xi ) * d_x;
		 Vz_xi = MacCormack( W.Vz, FB1, xi ) * d_x;

		Vx_zt = DOT_PRODUCT3D( _rDZ_DX.M11[indexOnSurf], _rDZ_DX.M12[indexOnSurf], _rDZ_DX.M13[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );
		Vy_zt = DOT_PRODUCT3D( _rDZ_DX.M21[indexOnSurf], _rDZ_DX.M22[indexOnSurf], _rDZ_DX.M23[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );
		Vz_zt = DOT_PRODUCT3D( _rDZ_DX.M31[indexOnSurf], _rDZ_DX.M32[indexOnSurf], _rDZ_DX.M33[indexOnSurf], Vx_xi, Vy_xi, Vz_xi );
		
		Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
		Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
		Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

		Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
		Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
		Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;

		h_Aux_x.Txx[pos] = h_Aux_x.Txx[pos] + Txx3;																									
		h_Aux_x.Tyy[pos] = h_Aux_x.Tyy[pos] + Tyy3;																									
		h_Aux_x.Tzz[pos] = h_Aux_x.Tzz[pos] + Tzz3;																									
		h_Aux_x.Txy[pos] = h_Aux_x.Txy[pos] + Txy3;																									
		h_Aux_x.Txz[pos] = h_Aux_x.Txz[pos] + Txz3;																									
		h_Aux_x.Tyz[pos] = h_Aux_x.Tyz[pos] + Tyz3;																									
									
	}

}


__global__ void pml_free_surface_y( const WAVE h_W, const WAVE W, const AUXILIARY h_Aux_y, const AUXILIARY Aux_y,			
									const FLOAT * ZT_X, const FLOAT * ZT_Y, const FLOAT * ZT_Z, MEDIUM_FLOAT medium,		
									const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,	const FLOAT * pml_d_y,
									const int nPML, const int _nx_, const int _ny_, const int _nz_,
									const int FLAG, const float rDH, const int FB2, const float DT )
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;

	int k0;
	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	int indexOnSurf;
	FLOAT mu;
	FLOAT lambda;

	FLOAT d_y;

	FLOAT Vx_et;		FLOAT Vy_et;		FLOAT Vz_et;
	FLOAT Vx_zt;		FLOAT Vy_zt;		FLOAT Vz_zt;
	FLOAT zt_x ;		FLOAT zt_y ;		FLOAT zt_z ;
		
	FLOAT Txx3;
	FLOAT Tyy3;
	FLOAT Tzz3;
	FLOAT Txy3;
	FLOAT Txz3;
	FLOAT Tyz3;


	int stride = FLAG * ( ny - nPML );
	k0 = nz - 1;
	if ( i0 >= 0 && i0 < nx && j0 >= 0 && j0 < nPML )
	{
		i = i0 + HALO;
		j = j0 + HALO + stride;
		k = k0 + HALO;
		index = INDEX(i, j, k);
		indexOnSurf = INDEX( i, j, 0 );
		pos	= Index3D( i0, j0, k0, nx, nPML, nz );	//i0 + j0 * nx + k0 * nx * nPML;

		mu     = medium.mu[index];
		lambda = medium.lambda[index];

		d_y = pml_d_y[j];
		zt_x = ZT_X[index];		zt_y = ZT_Y[index];		zt_z = ZT_Z[index];

		 Vx_et = MacCormack( W.Vx, FB2, et ) * d_y;
		 Vy_et = MacCormack( W.Vy, FB2, et ) * d_y;
		 Vz_et = MacCormack( W.Vz, FB2, et ) * d_y;

		Vx_zt = DOT_PRODUCT3D( _rDZ_DY.M11[indexOnSurf], _rDZ_DY.M12[indexOnSurf], _rDZ_DY.M13[indexOnSurf], Vx_et, Vy_et, Vz_et );
		Vy_zt = DOT_PRODUCT3D( _rDZ_DY.M21[indexOnSurf], _rDZ_DY.M22[indexOnSurf], _rDZ_DY.M23[indexOnSurf], Vx_et, Vy_et, Vz_et );
		Vz_zt = DOT_PRODUCT3D( _rDZ_DY.M31[indexOnSurf], _rDZ_DY.M32[indexOnSurf], _rDZ_DY.M33[indexOnSurf], Vx_et, Vy_et, Vz_et );

		Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
		Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
		Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

		Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
		Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
		Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;


		h_Aux_y.Txx[pos] = h_Aux_y.Txx[pos] + Txx3;
		h_Aux_y.Tyy[pos] = h_Aux_y.Tyy[pos] + Tyy3;
		h_Aux_y.Tzz[pos] = h_Aux_y.Tzz[pos] + Tzz3;
		h_Aux_y.Txy[pos] = h_Aux_y.Txy[pos] + Txy3;
		h_Aux_y.Txz[pos] = h_Aux_y.Txz[pos] + Txz3;
		h_Aux_y.Tyz[pos] = h_Aux_y.Tyz[pos] + Tyz3;

	}

}

// PML for free surface derive
void pmlFreeSurfaceDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
						  const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
						  const AUX4 Aux4_1, const AUX4 Aux4_2,
						  const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY, const PML_D_FLOAT pml_d,
						  const MPI_BORDER border, const int FB1, const int FB2, const float DT )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	int nPML = grid->nPML;
	float rDH = grid->rDH;

	FLOAT * ZT_X = con.zt_x;	FLOAT * ZT_Y = con.zt_y;	FLOAT * ZT_Z = con.zt_z;

	FLOAT * pml_d_x = pml_d.x;
	FLOAT * pml_d_y = pml_d.y;

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	dim3 thread( 8, 8, 1);
	dim3 blockX;
	blockX.x = ( nPML + thread.x - 1 ) / thread.x;
	blockX.y = ( ny   + thread.y - 1 ) / thread.y;
	blockX.z = 1;

	dim3 blockY;
	blockY.x = ( nx   + thread.x - 1 ) / thread.x;
	blockY.y = ( nPML + thread.y - 1 ) / thread.y;
	blockY.z = 1;

	if ( border.isx1 )
	{
		pml_free_surface_x <<< blockX, thread >>>	
		( h_W, W, Aux4_1.h_Aux_x, Aux4_1.Aux_x,
		  ZT_X, ZT_Y, ZT_Z, medium, _rDZ_DX, _rDZ_DY,
		  pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT );
	}
	if ( border.isy1 )
	{
		pml_free_surface_y <<< blockY, thread >>>
		( h_W, W, Aux4_1.h_Aux_y, Aux4_1.Aux_y,
		  ZT_X, ZT_Y, ZT_Z, medium, _rDZ_DX, _rDZ_DY,
		  pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT );
	}


	if ( border.isx2 )
	{
		pml_free_surface_x <<< blockX, thread >>>
	    ( h_W, W, Aux4_2.h_Aux_x, Aux4_2.Aux_x,
		  ZT_X, ZT_Y, ZT_Z, medium, _rDZ_DX, _rDZ_DY,
		  pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT );
	}
	if ( border.isy2 )
	{
		pml_free_surface_y <<< blockY, thread >>>	
		( h_W, W, Aux4_2.h_Aux_y, Aux4_2.Aux_y,	
		  ZT_X, ZT_Y, ZT_Z, medium, _rDZ_DX, _rDZ_DY,				
		  pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT );
	}

}

