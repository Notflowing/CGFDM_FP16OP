#include "header.h"

// Allocate device FLOAT memory for PML(LSRK)
void allocAuxPML( const int nPML, const int N1, const int N2, \
				  AUX * h_Aux, AUX * Aux, AUX * t_Aux, AUX * m_Aux )
{
	long long num = nPML * N1 * N2; 
		
	FLOAT * pAux = NULL;
	long long size = sizeof( FLOAT ) * num * WAVESIZE * 4;
	//printf("num = %ld, nPML = %d, N1 = %d, N2 = %d\n", num, nPML, N1, N2  );

	checkCudaErrors( cudaMalloc((void **)&pAux, size) );
	checkCudaErrors( cudaMemset( pAux, 0, size) );
	//printf("pAux1 = %d\n", pAux );

	h_Aux->Vx  = pAux + 0 * num;
	h_Aux->Vy  = pAux + 1 * num;
	h_Aux->Vz  = pAux + 2 * num;
	h_Aux->Txx = pAux + 3 * num;
	h_Aux->Tyy = pAux + 4 * num;
	h_Aux->Tzz = pAux + 5 * num;
	h_Aux->Txy = pAux + 6 * num;
	h_Aux->Txz = pAux + 7 * num;
	h_Aux->Tyz = pAux + 8 * num;

	pAux 	 = pAux + 9 * num;
	//printf("pAux2 = %d\n", pAux );

	Aux->Vx  = pAux + 0 * num;
	Aux->Vy  = pAux + 1 * num;
	Aux->Vz  = pAux + 2 * num;
	Aux->Txx = pAux + 3 * num;
	Aux->Tyy = pAux + 4 * num;
	Aux->Tzz = pAux + 5 * num;
	Aux->Txy = pAux + 6 * num;
	Aux->Txz = pAux + 7 * num;
	Aux->Tyz = pAux + 8 * num;

	pAux    = pAux + 9 * num;
	//printf("pAux3 = %d\n", pAux );

	t_Aux->Vx  = pAux + 0 * num;
	t_Aux->Vy  = pAux + 1 * num;
	t_Aux->Vz  = pAux + 2 * num;
	t_Aux->Txx = pAux + 3 * num;
	t_Aux->Tyy = pAux + 4 * num;
	t_Aux->Tzz = pAux + 5 * num;
	t_Aux->Txy = pAux + 6 * num;
	t_Aux->Txz = pAux + 7 * num;
	t_Aux->Tyz = pAux + 8 * num;

	pAux 	 = pAux + 9 * num;
	//printf("pAux4 = %d\n", pAux );

	m_Aux->Vx  = pAux + 0 * num;
	m_Aux->Vy  = pAux + 1 * num;
	m_Aux->Vz  = pAux + 2 * num;
	m_Aux->Txx = pAux + 3 * num;
	m_Aux->Tyy = pAux + 4 * num;
	m_Aux->Tzz = pAux + 5 * num;
	m_Aux->Txy = pAux + 6 * num;
	m_Aux->Txz = pAux + 7 * num;
	m_Aux->Tyz = pAux + 8 * num;
	//printf("pAux4 = %d\n", pAux );

}

// Allocate device FLOAT memory for PML
void allocPML( const GRID * const grid, AUX4 *Aux4_1, AUX4 *Aux4_2, const MPI_BORDER border )
{
	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;

	int nPML = grid->nPML;
	//printf( "nx = %d, nPML = %d\n", nx, nPML );

	memset( ( void *)Aux4_1, 0, sizeof(AUX4) );
	memset( ( void *)Aux4_2, 0, sizeof(AUX4) );


	if ( border.isx1 && nPML >= nx ) { printf( "The PML layer(nPML) just bigger than nx(%d)\n", nPML, nx );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isy1 && nPML >= ny ) { printf( "The PML layer(nPML) just bigger than ny(%d)\n", nPML, ny );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isz1 && nPML >= nz ) { printf( "The PML layer(nPML) just bigger than nz(%d)\n", nPML, nz );  MPI_Abort( MPI_COMM_WORLD, 130 );}
                                                                                                                                     
	if ( border.isx2 && nPML >= nx ) { printf( "The PML layer(nPML) just bigger than nx(%d)\n", nPML, nx );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isy2 && nPML >= ny ) { printf( "The PML layer(nPML) just bigger than ny(%d)\n", nPML, ny );  MPI_Abort( MPI_COMM_WORLD, 130 );}
	if ( border.isz2 && nPML >= nz ) { printf( "The PML layer(nPML) just bigger than nz(%d)\n", nPML, nz );  MPI_Abort( MPI_COMM_WORLD, 130 );}

	if ( border.isx1 ) allocAuxPML( nPML, ny, nz, &( Aux4_1->h_Aux_x ), &( Aux4_1->Aux_x ), &( Aux4_1->t_Aux_x ), &( Aux4_1->m_Aux_x ) );
	if ( border.isy1 ) allocAuxPML( nPML, nx, nz, &( Aux4_1->h_Aux_y ), &( Aux4_1->Aux_y ), &( Aux4_1->t_Aux_y ), &( Aux4_1->m_Aux_y ) );
	if ( border.isz1 ) allocAuxPML( nPML, nx, ny, &( Aux4_1->h_Aux_z ), &( Aux4_1->Aux_z ), &( Aux4_1->t_Aux_z ), &( Aux4_1->m_Aux_z ) );
                                                                              
	if ( border.isx2 ) allocAuxPML( nPML, ny, nz, &( Aux4_2->h_Aux_x ), &( Aux4_2->Aux_x ), &( Aux4_2->t_Aux_x ), &( Aux4_2->m_Aux_x ) );
	if ( border.isy2 ) allocAuxPML( nPML, nx, nz, &( Aux4_2->h_Aux_y ), &( Aux4_2->Aux_y ), &( Aux4_2->t_Aux_y ), &( Aux4_2->m_Aux_y ) );

#ifndef FREE_SURFACE
	if ( border.isz2 ) allocAuxPML( nPML, nx, ny, &( Aux4_2->h_Aux_z ), &( Aux4_2->Aux_z ), &( Aux4_2->t_Aux_z ), &( Aux4_2->m_Aux_z ) );
#endif

}

void freePML( const MPI_BORDER border,  AUX4 Aux4_1, AUX4 Aux4_2 )
{

	if ( border.isx1 )  checkCudaErrors( cudaFree( Aux4_1.h_Aux_x.Vx ) );
	if ( border.isy1 )  checkCudaErrors( cudaFree( Aux4_1.h_Aux_y.Vx ) );
	if ( border.isz1 )  checkCudaErrors( cudaFree( Aux4_1.h_Aux_z.Vx ) );
	                  
	if ( border.isx2 )  checkCudaErrors( cudaFree( Aux4_2.h_Aux_x.Vx ) );
	if ( border.isy2 )  checkCudaErrors( cudaFree( Aux4_2.h_Aux_y.Vx ) );
#ifndef FREE_SURFACE
	if ( border.isz2 )  checkCudaErrors( cudaFree( Aux4_2.h_Aux_z.Vx ) );
#endif

}


__global__ void pml_deriv_x( const WAVE h_W, const WAVE W, const AUXILIARY h_Aux_x, const AUXILIARY Aux_x,
							 const FLOAT * XI_X, const FLOAT * XI_Y, const FLOAT * XI_Z,
							 const MEDIUM_FLOAT medium, const FLOAT * pml_alpha_x, const FLOAT * pml_beta_x,
							 const FLOAT * pml_d_x, const int nPML, const int _nx_, const int _ny_, const int _nz_,
							 const int FLAG, const float rDH, const int FB1, const float DT )
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	FLOAT mu;
	FLOAT lambda;
	FLOAT buoyancy;

	FLOAT beta_x;
	FLOAT d_x;
	FLOAT alpha_d_x;

	FLOAT xi_x;		FLOAT xi_y; 	FLOAT xi_z;

	FLOAT Txx_xi;	FLOAT Tyy_xi;	FLOAT Txy_xi;
	FLOAT Txz_xi;	FLOAT Tyz_xi;	FLOAT Tzz_xi;
	FLOAT Vx_xi ;	FLOAT Vy_xi ;	FLOAT Vz_xi ;

	FLOAT Vx1 ;
	FLOAT Vy1 ;
	FLOAT Vz1 ;
	FLOAT Txx1;
	FLOAT Tyy1;
	FLOAT Tzz1;
	FLOAT Txy1;
	FLOAT Txz1;
	FLOAT Tyz1;

	int stride = FLAG * ( nx - nPML );
	if ( i0 >= 0 && i0 < nPML && j0 >= 0 && j0 < ny && k0 >= 0 && k0 < nz )
	{
		i = i0 + HALO + stride;
		j = j0 + HALO;
		k = k0 + HALO;
		index = INDEX(i, j, k);
		pos	= Index3D( i0, j0, k0, nPML, ny, nz );	//i0 + j0 * nPML + k0 * nPML * ny;

		mu       = medium.mu[index];
		lambda   = medium.lambda[index];
		buoyancy = medium.buoyancy[index];

		beta_x = pml_beta_x[i];
		d_x = pml_d_x[i];
		alpha_d_x = d_x + pml_alpha_x[i];

		// if ( i0 == nPML - 10 && j0 == ny - 10 && k0 == nz - 10 )
		// printf( "beta_x = %e, d_x = %e, alpha_d_x = %e\n",  (float)beta_x, (float)d_x, (float)alpha_d_x );

		xi_x = XI_X[index];		xi_y = XI_Y[index];		xi_z = XI_Z[index];

		 Vx_xi = MacCormack( W.Vx , FB1, xi ) * d_x;
		 Vy_xi = MacCormack( W.Vy , FB1, xi ) * d_x;
		 Vz_xi = MacCormack( W.Vz , FB1, xi ) * d_x;
		Txx_xi = MacCormack( W.Txx, FB1, xi ) * d_x;
		Tyy_xi = MacCormack( W.Tyy, FB1, xi ) * d_x;
		Tzz_xi = MacCormack( W.Tzz, FB1, xi ) * d_x;
		Txy_xi = MacCormack( W.Txy, FB1, xi ) * d_x;
		Txz_xi = MacCormack( W.Txz, FB1, xi ) * d_x;
		Tyz_xi = MacCormack( W.Tyz, FB1, xi ) * d_x;

		Vx1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txx_xi, Txy_xi, Txz_xi ) * buoyancy;
		Vy1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txy_xi, Tyy_xi, Tyz_xi ) * buoyancy;
		Vz1  = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Txz_xi, Tyz_xi, Tzz_xi ) * buoyancy;

		Txx1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_x * Vx_xi );
		Tyy1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_y * Vy_xi );
		Tzz1 = DOT_PRODUCT3D( xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi ) * lambda + 2.0f * mu * ( xi_z * Vz_xi );

		Txy1 = DOT_PRODUCT2D( xi_y, xi_x, Vx_xi, Vy_xi ) * mu;
		Txz1 = DOT_PRODUCT2D( xi_z, xi_x, Vx_xi, Vz_xi ) * mu;
		Tyz1 = DOT_PRODUCT2D( xi_z, xi_y, Vy_xi, Vz_xi ) * mu;
															
		h_Aux_x.Vx [pos] = Vx1  - ( alpha_d_x * Aux_x.Vx [pos] ) * DT;
		h_Aux_x.Vy [pos] = Vy1  - ( alpha_d_x * Aux_x.Vy [pos] ) * DT;
		h_Aux_x.Vz [pos] = Vz1  - ( alpha_d_x * Aux_x.Vz [pos] ) * DT;
		h_Aux_x.Txx[pos] = Txx1 - ( alpha_d_x * Aux_x.Txx[pos] ) * DT;
		h_Aux_x.Tyy[pos] = Tyy1 - ( alpha_d_x * Aux_x.Tyy[pos] ) * DT;
		h_Aux_x.Tzz[pos] = Tzz1 - ( alpha_d_x * Aux_x.Tzz[pos] ) * DT;
		h_Aux_x.Txy[pos] = Txy1 - ( alpha_d_x * Aux_x.Txy[pos] ) * DT;
		h_Aux_x.Txz[pos] = Txz1 - ( alpha_d_x * Aux_x.Txz[pos] ) * DT;
		h_Aux_x.Tyz[pos] = Tyz1 - ( alpha_d_x * Aux_x.Tyz[pos] ) * DT;
															
		h_W.Vx [index] = h_W.Vx [index] - beta_x * Aux_x.Vx [pos] * DT;		
		h_W.Vy [index] = h_W.Vy [index] - beta_x * Aux_x.Vy [pos] * DT;		
		h_W.Vz [index] = h_W.Vz [index] - beta_x * Aux_x.Vz [pos] * DT;		
		h_W.Txx[index] = h_W.Txx[index] - beta_x * Aux_x.Txx[pos] * DT;		
		h_W.Tyy[index] = h_W.Tyy[index] - beta_x * Aux_x.Tyy[pos] * DT;		
		h_W.Tzz[index] = h_W.Tzz[index] - beta_x * Aux_x.Tzz[pos] * DT;		
		h_W.Txy[index] = h_W.Txy[index] - beta_x * Aux_x.Txy[pos] * DT;		
		h_W.Txz[index] = h_W.Txz[index] - beta_x * Aux_x.Txz[pos] * DT;		
		h_W.Tyz[index] = h_W.Tyz[index] - beta_x * Aux_x.Tyz[pos] * DT;
										
	}

}


__global__ void pml_deriv_y( const WAVE h_W, const WAVE W, const AUXILIARY h_Aux_y, const AUXILIARY Aux_y,
							 const FLOAT * ET_X, const FLOAT * ET_Y, const FLOAT * ET_Z,
							 const MEDIUM_FLOAT medium, const FLOAT * pml_alpha_y, const FLOAT * pml_beta_y,
							 const FLOAT * pml_d_y, const int nPML, const int _nx_, const int _ny_, const int _nz_,
							 const int FLAG, const float rDH, const int FB2, const float DT )
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	FLOAT mu;
	FLOAT lambda;
	FLOAT buoyancy;

	FLOAT beta_y;
	FLOAT d_y;
	FLOAT alpha_d_y;

	FLOAT et_x;		FLOAT et_y;		FLOAT et_z;

	FLOAT Txx_et;	FLOAT Tyy_et;	FLOAT Txy_et;
	FLOAT Txz_et;	FLOAT Tyz_et;	FLOAT Tzz_et;
	FLOAT Vx_et ;	FLOAT Vy_et ;	FLOAT Vz_et ;

	FLOAT Vx2 ;
	FLOAT Vy2 ;
	FLOAT Vz2 ;
	FLOAT Txx2;
	FLOAT Tyy2;
	FLOAT Tzz2;
	FLOAT Txy2;
	FLOAT Txz2;
	FLOAT Tyz2;

	int stride = FLAG * ( ny - nPML );
	if ( i0 >= 0 && i0 < nx && j0 >= 0 && j0 < nPML && k0 >= 0 && k0 < nz )
	{
		i = i0 + HALO;
		j = j0 + HALO + stride;
		k = k0 + HALO;	
		index = INDEX(i, j, k);
		pos	= Index3D( i0, j0, k0, nx, nPML, nz );	//i0 + j0 * nx + k0 * nx * nPML;

		mu       = medium.mu[index];
		lambda   = medium.lambda[index];
		buoyancy = medium.buoyancy[index];

		beta_y = pml_beta_y[j];
		d_y = pml_d_y[j];
		alpha_d_y = d_y + pml_alpha_y[j];
		
		et_x = ET_X[index];		et_y = ET_Y[index];		et_z = ET_Z[index];

		 Vx_et = MacCormack( W.Vx , FB2, et ) * d_y;
		 Vy_et = MacCormack( W.Vy , FB2, et ) * d_y;
		 Vz_et = MacCormack( W.Vz , FB2, et ) * d_y;
		Txx_et = MacCormack( W.Txx, FB2, et ) * d_y;
		Tyy_et = MacCormack( W.Tyy, FB2, et ) * d_y;
		Tzz_et = MacCormack( W.Tzz, FB2, et ) * d_y;
		Txy_et = MacCormack( W.Txy, FB2, et ) * d_y;
		Txz_et = MacCormack( W.Txz, FB2, et ) * d_y;
		Tyz_et = MacCormack( W.Tyz, FB2, et ) * d_y;

		Vx2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txx_et, Txy_et, Txz_et ) * buoyancy;
		Vy2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txy_et, Tyy_et, Tyz_et ) * buoyancy;
		Vz2  = DOT_PRODUCT3D( et_x, et_y, et_z, Txz_et, Tyz_et, Tzz_et ) * buoyancy;

		Txx2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_x * Vx_et );
		Tyy2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_y * Vy_et );
		Tzz2 = DOT_PRODUCT3D( et_x, et_y, et_z, Vx_et, Vy_et, Vz_et ) * lambda + 2.0f * mu * ( et_z * Vz_et );

		Txy2 = DOT_PRODUCT2D( et_y, et_x, Vx_et, Vy_et ) * mu;
		Txz2 = DOT_PRODUCT2D( et_z, et_x, Vx_et, Vz_et ) * mu;
		Tyz2 = DOT_PRODUCT2D( et_z, et_y, Vy_et, Vz_et ) * mu;

		h_Aux_y.Vx [pos] = Vx2  - ( alpha_d_y * Aux_y.Vx [pos] ) * DT;
		h_Aux_y.Vy [pos] = Vy2  - ( alpha_d_y * Aux_y.Vy [pos] ) * DT;
		h_Aux_y.Vz [pos] = Vz2  - ( alpha_d_y * Aux_y.Vz [pos] ) * DT;
		h_Aux_y.Txx[pos] = Txx2 - ( alpha_d_y * Aux_y.Txx[pos] ) * DT;
		h_Aux_y.Tyy[pos] = Tyy2 - ( alpha_d_y * Aux_y.Tyy[pos] ) * DT;
		h_Aux_y.Tzz[pos] = Tzz2 - ( alpha_d_y * Aux_y.Tzz[pos] ) * DT;
		h_Aux_y.Txy[pos] = Txy2 - ( alpha_d_y * Aux_y.Txy[pos] ) * DT;
		h_Aux_y.Txz[pos] = Txz2 - ( alpha_d_y * Aux_y.Txz[pos] ) * DT;
		h_Aux_y.Tyz[pos] = Tyz2 - ( alpha_d_y * Aux_y.Tyz[pos] ) * DT;


		h_W.Vx [index] = h_W.Vx [index] - beta_y * Aux_y.Vx [pos] * DT;
		h_W.Vy [index] = h_W.Vy [index] - beta_y * Aux_y.Vy [pos] * DT;
		h_W.Vz [index] = h_W.Vz [index] - beta_y * Aux_y.Vz [pos] * DT;
		h_W.Txx[index] = h_W.Txx[index] - beta_y * Aux_y.Txx[pos] * DT;
		h_W.Tyy[index] = h_W.Tyy[index] - beta_y * Aux_y.Tyy[pos] * DT;
		h_W.Tzz[index] = h_W.Tzz[index] - beta_y * Aux_y.Tzz[pos] * DT;
		h_W.Txy[index] = h_W.Txy[index] - beta_y * Aux_y.Txy[pos] * DT;
		h_W.Txz[index] = h_W.Txz[index] - beta_y * Aux_y.Txz[pos] * DT;
		h_W.Tyz[index] = h_W.Tyz[index] - beta_y * Aux_y.Tyz[pos] * DT;

	}

}

														
__global__ void pml_deriv_z( const WAVE h_W, const WAVE W, const AUXILIARY h_Aux_z, const AUXILIARY Aux_z,
							 const FLOAT * ZT_X, const FLOAT * ZT_Y, const FLOAT * ZT_Z,
							 const MEDIUM_FLOAT medium, const FLOAT * pml_alpha_z, const FLOAT * pml_beta_z,
							 const FLOAT * pml_d_z, const int nPML, const int _nx_, const int _ny_, const int _nz_,
							 const int FLAG, const float rDH, const int FB3, const float DT )			
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	FLOAT mu;
	FLOAT lambda;
	FLOAT buoyancy;

	FLOAT beta_z;
	FLOAT d_z;
	FLOAT alpha_d_z;

	FLOAT zt_x;		FLOAT zt_y;		FLOAT zt_z;	

	FLOAT Txx_zt;	FLOAT Tyy_zt;	FLOAT Txy_zt;
	FLOAT Txz_zt;	FLOAT Tyz_zt;	FLOAT Tzz_zt;
	FLOAT Vx_zt ;	FLOAT Vy_zt ;	FLOAT Vz_zt ;

	FLOAT Vx3 ;
	FLOAT Vy3 ;
	FLOAT Vz3 ;
	FLOAT Txx3;
	FLOAT Tyy3;
	FLOAT Tzz3;
	FLOAT Txy3;
	FLOAT Txz3;
	FLOAT Tyz3;


	int stride = FLAG * ( nz - nPML );
	if ( i0 >= 0 && i0 < nx && j0 >= 0 && j0 < ny && k0 >= 0 && k0 < nPML )
	{
		i = i0 + HALO;
		j = j0 + HALO;
		k = k0 + HALO + stride;
		index = INDEX(i, j, k);
		pos	= Index3D( i0, j0, k0, nx, ny, nPML );	//i0 + j0 * nx + k0 * nx * ny;

		mu       = medium.mu[index];			
		lambda   = medium.lambda[index];
		buoyancy = medium.buoyancy[index];

		beta_z = pml_beta_z[k];
		d_z = pml_d_z[k];
		alpha_d_z = d_z + pml_alpha_z[k];

		zt_x = ZT_X[index];		zt_y = ZT_Y[index];		zt_z = ZT_Z[index];

		 Vx_zt = MacCormack( W.Vx , FB3, zt ) * d_z;
		 Vy_zt = MacCormack( W.Vy , FB3, zt ) * d_z;
		 Vz_zt = MacCormack( W.Vz , FB3, zt ) * d_z;
		Txx_zt = MacCormack( W.Txx, FB3, zt ) * d_z;
		Tyy_zt = MacCormack( W.Tyy, FB3, zt ) * d_z;
		Tzz_zt = MacCormack( W.Tzz, FB3, zt ) * d_z;
		Txy_zt = MacCormack( W.Txy, FB3, zt ) * d_z;
		Txz_zt = MacCormack( W.Txz, FB3, zt ) * d_z;
		Tyz_zt = MacCormack( W.Tyz, FB3, zt ) * d_z;

		Vx3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txx_zt, Txy_zt, Txz_zt ) * buoyancy;
		Vy3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txy_zt, Tyy_zt, Tyz_zt ) * buoyancy;
		Vz3  = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Txz_zt, Tyz_zt, Tzz_zt ) * buoyancy;

		Txx3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_x * Vx_zt );
		Tyy3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_y * Vy_zt );
		Tzz3 = DOT_PRODUCT3D( zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt ) * lambda + 2.0f * mu * ( zt_z * Vz_zt );

		Txy3 = DOT_PRODUCT2D( zt_y, zt_x, Vx_zt, Vy_zt ) * mu;
		Txz3 = DOT_PRODUCT2D( zt_z, zt_x, Vx_zt, Vz_zt ) * mu;
		Tyz3 = DOT_PRODUCT2D( zt_z, zt_y, Vy_zt, Vz_zt ) * mu;

		h_Aux_z.Vx [pos] = Vx3  - ( alpha_d_z * Aux_z.Vx [pos] ) * DT;	
		h_Aux_z.Vy [pos] = Vy3  - ( alpha_d_z * Aux_z.Vy [pos] ) * DT;	
		h_Aux_z.Vz [pos] = Vz3  - ( alpha_d_z * Aux_z.Vz [pos] ) * DT;	
		h_Aux_z.Txx[pos] = Txx3 - ( alpha_d_z * Aux_z.Txx[pos] ) * DT;	
		h_Aux_z.Tyy[pos] = Tyy3 - ( alpha_d_z * Aux_z.Tyy[pos] ) * DT;	
		h_Aux_z.Tzz[pos] = Tzz3 - ( alpha_d_z * Aux_z.Tzz[pos] ) * DT;	
		h_Aux_z.Txy[pos] = Txy3 - ( alpha_d_z * Aux_z.Txy[pos] ) * DT;	
		h_Aux_z.Txz[pos] = Txz3 - ( alpha_d_z * Aux_z.Txz[pos] ) * DT;	
		h_Aux_z.Tyz[pos] = Tyz3 - ( alpha_d_z * Aux_z.Tyz[pos] ) * DT;	

		h_W.Vx [index] = h_W.Vx [index] - beta_z * Aux_z.Vx [pos] * DT;
		h_W.Vy [index] = h_W.Vy [index] - beta_z * Aux_z.Vy [pos] * DT;
		h_W.Vz [index] = h_W.Vz [index] - beta_z * Aux_z.Vz [pos] * DT;
		h_W.Txx[index] = h_W.Txx[index] - beta_z * Aux_z.Txx[pos] * DT;
		h_W.Tyy[index] = h_W.Tyy[index] - beta_z * Aux_z.Tyy[pos] * DT;
		h_W.Tzz[index] = h_W.Tzz[index] - beta_z * Aux_z.Tzz[pos] * DT;
		h_W.Txy[index] = h_W.Txy[index] - beta_z * Aux_z.Txy[pos] * DT;
		h_W.Txz[index] = h_W.Txz[index] - beta_z * Aux_z.Txz[pos] * DT;
		h_W.Tyz[index] = h_W.Tyz[index] - beta_z * Aux_z.Tyz[pos] * DT;

	}

}

// PML derive
void pmlDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
			   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
			   const AUX4 Aux4_1, const AUX4 Aux4_2, const PML_ALPHA_FLOAT pml_alpha,
			   const PML_BETA_FLOAT pml_beta, const PML_D_FLOAT pml_d, const MPI_BORDER border,
			   const int FB1, const int FB2, const int FB3,
			   const float DT )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	float rDH = grid->rDH;
	int nPML = grid->nPML;

	FLOAT * XI_X = con.xi_x;	FLOAT * XI_Y = con.xi_y;	FLOAT * XI_Z = con.xi_z;
	FLOAT * ET_X = con.et_x;	FLOAT * ET_Y = con.et_y;	FLOAT * ET_Z = con.et_z;
	FLOAT * ZT_X = con.zt_x;	FLOAT * ZT_Y = con.zt_y;	FLOAT * ZT_Z = con.zt_z;

	FLOAT * pml_alpha_x = pml_alpha.x;    FLOAT * pml_beta_x = pml_beta.x;    FLOAT * pml_d_x = pml_d.x;
	FLOAT * pml_alpha_y = pml_alpha.y;    FLOAT * pml_beta_y = pml_beta.y;    FLOAT * pml_d_y = pml_d.y;
	FLOAT * pml_alpha_z = pml_alpha.z;    FLOAT * pml_beta_z = pml_beta.z;    FLOAT * pml_d_z = pml_d.z;

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	dim3 thread( 8, 8, 8);
	dim3 blockX;
	blockX.x = ( nPML + thread.x - 1 ) / thread.x;
	blockX.y = ( ny   + thread.y - 1 ) / thread.y;
	blockX.z = ( nz   + thread.z - 1 ) / thread.z;

	dim3 blockY;
	blockY.x = ( nx   + thread.x - 1 ) / thread.x;
	blockY.y = ( nPML + thread.y - 1 ) / thread.y;
	blockY.z = ( nz   + thread.z - 1 ) / thread.z;

	dim3 blockZ;
	blockZ.x = ( nx   + thread.x - 1 ) / thread.x;
	blockZ.y = ( ny   + thread.y - 1 ) / thread.y;
	blockZ.z = ( nPML + thread.z - 1 ) / thread.z;


	if ( border.isx1 )
	{
		pml_deriv_x <<< blockX, thread >>>
		( h_W, W, Aux4_1.h_Aux_x, Aux4_1.Aux_x, XI_X, XI_Y, XI_Z,
		  medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML,
		  _nx_, _ny_, _nz_, 0, rDH, FB1, DT );
	}
	if ( border.isy1 )
	{
		pml_deriv_y <<< blockY, thread >>>
		( h_W, W, Aux4_1.h_Aux_y, Aux4_1.Aux_y, ET_X, ET_Y, ET_Z,
		  medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML,
		  _nx_, _ny_, _nz_, 0, rDH, FB2, DT );
	}
	if ( border.isz1 )
	{
		pml_deriv_z <<< blockZ, thread >>>
		( h_W, W, Aux4_1.h_Aux_z, Aux4_1.Aux_z, ZT_X, ZT_Y, ZT_Z,
		  medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML,
		  _nx_, _ny_, _nz_, 0, rDH, FB3, DT );
	}


	if ( border.isx2 )
	{
		pml_deriv_x <<< blockX, thread >>>
		( h_W, W, Aux4_2.h_Aux_x, Aux4_2.Aux_x, XI_X, XI_Y, XI_Z,
		  medium, pml_alpha_x, pml_beta_x, pml_d_x, nPML,
		  _nx_, _ny_, _nz_, 1, rDH, FB1, DT );
	}
	if ( border.isy2 )
	{
		pml_deriv_y <<< blockY, thread >>>
		( h_W, W, Aux4_2.h_Aux_y, Aux4_2.Aux_y, ET_X, ET_Y, ET_Z,
		  medium, pml_alpha_y, pml_beta_y, pml_d_y, nPML,
		  _nx_, _ny_, _nz_, 1, rDH, FB2, DT );
	}
#ifndef FREE_SURFACE                                                                                                                                                                                   
	if ( border.isz2 )
	{
		pml_deriv_z <<< blockZ, thread >>>
		( h_W, W, Aux4_2.h_Aux_z, Aux4_2.Aux_z, ZT_X, ZT_Y, ZT_Z,
		  medium, pml_alpha_z, pml_beta_z, pml_d_z, nPML,
		  _nx_, _ny_, _nz_, 1, rDH, FB3, DT );
	}
#endif
	checkCudaErrors( cudaDeviceSynchronize() );

}


