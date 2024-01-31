#include "header.h"

// Allocate device float memory for contravariant and Jacobi matrix
void allocContravariantJac( const GRID * const grid, float ** Jac, CONTRAVARIANT * con )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	long long num = _nx_ * _ny_ * _nz_;
	long long sizeJac = sizeof( float ) * num;
	long long sizeCon = sizeof( float ) * num * CONTRASIZE;

	float * pJac = NULL;
	float * pContravariant = NULL;

	checkCudaErrors( cudaMalloc((void **)&pJac, sizeJac) );
	checkCudaErrors( cudaMalloc((void **)&pContravariant, sizeCon) );
	
	checkCudaErrors( cudaMemset(pJac, 0, sizeJac) );
	checkCudaErrors( cudaMemset(pContravariant, 0, sizeCon) );
	
	*Jac = pJac;

	con->xi_x = pContravariant + num * 0;
	con->xi_y = pContravariant + num * 1;
	con->xi_z = pContravariant + num * 2;
                                                  
	con->et_x = pContravariant + num * 3;
	con->et_y = pContravariant + num * 4;
	con->et_z = pContravariant + num * 5;
                                                  
	con->zt_x = pContravariant + num * 6;
	con->zt_y = pContravariant + num * 7;
	con->zt_z = pContravariant + num * 8;

}

void freeContravariant( CONTRAVARIANT con )
{	
	checkCudaErrors( cudaFree(con.xi_x) );
}

void freeJac( float * Jac )
{	
	checkCudaErrors( cudaFree(Jac) );
}

// Allocate device FLOAT(half) memory for contravariant and Jacobi matrix
void allocContravariantJac_FP16( const GRID * const grid, FLOAT ** Jac, CONTRAVARIANT_FLOAT * con )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	long long num = _nx_ * _ny_ * _nz_;
	long long sizeJac = sizeof( FLOAT ) * num;
	long long sizeCon = sizeof( FLOAT ) * num * CONTRASIZE;

	FLOAT * pJac = NULL;
	FLOAT * pContravariant = NULL;

	checkCudaErrors( cudaMalloc((void **)&pJac, sizeJac) );
	checkCudaErrors( cudaMalloc((void **)&pContravariant, sizeCon) );
	
	checkCudaErrors( cudaMemset(pJac, 0, sizeJac) );
	checkCudaErrors( cudaMemset(pContravariant, 0, sizeCon) );
	
	*Jac = pJac;

	con->xi_x = pContravariant + num * 0;
	con->xi_y = pContravariant + num * 1;
	con->xi_z = pContravariant + num * 2;
                                                  
	con->et_x = pContravariant + num * 3;
	con->et_y = pContravariant + num * 4;
	con->et_z = pContravariant + num * 5;
                                                  
	con->zt_x = pContravariant + num * 6;
	con->zt_y = pContravariant + num * 7;
	con->zt_z = pContravariant + num * 8;

}

void freeContravariantJac_FP16( FLOAT * Jac, CONTRAVARIANT_FLOAT con )
{	
	checkCudaErrors( cudaFree(Jac) );
	checkCudaErrors( cudaFree(con.xi_x) );
}


//When change the fast axises:
/*
 * =============================================
 *             BE careful!!!!!!!!!!!!
 * =============================================
*/

// Compute coordinate conversion coefficients contravariant and Jacobi
// using the same difference format as for solving equations
__global__ void solve_con_jac( const CONTRAVARIANT con, const COORD coord, float * const Jac, \
							   const int _nx_, const int _ny_, const int _nz_, const float rDH )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO; 
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + HALO;

	long long index = 0;
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	float Jacobi = 0.0f;
	float JacobiInv = 0.0f;

	float x_xi = 0.0f, x_et = 0.0f, x_zt = 0.0f;
	float y_xi = 0.0f, y_et = 0.0f, y_zt = 0.0f;
	float z_xi = 0.0f, z_et = 0.0f, z_zt = 0.0f;

	float xi_x = 0.0f, et_x = 0.0f, zt_x = 0.0f;
	float xi_y = 0.0f, et_y = 0.0f, zt_y = 0.0f;
	float xi_z = 0.0f, et_z = 0.0f, zt_z = 0.0f;

	if ( i >= HALO && i < _nx && j >= HALO && j < _ny && k >= HALO && k < _nz )
	{
		index = INDEX( i, j, k );
		x_xi = 0.5f * ( MacCormack( coord.x, 1, xi ) + MacCormack( coord.x, -1, xi ) );
		x_et = 0.5f * ( MacCormack( coord.x, 1, et ) + MacCormack( coord.x, -1, et ) );
		x_zt = 0.5f * ( MacCormack( coord.x, 1, zt ) + MacCormack( coord.x, -1, zt ) );

		y_xi = 0.5f * ( MacCormack( coord.y, 1, xi ) + MacCormack( coord.y, -1, xi ) );
		y_et = 0.5f * ( MacCormack( coord.y, 1, et ) + MacCormack( coord.y, -1, et ) );
		y_zt = 0.5f * ( MacCormack( coord.y, 1, zt ) + MacCormack( coord.y, -1, zt ) );

		z_xi = 0.5f * ( MacCormack( coord.z, 1, xi ) + MacCormack( coord.z, -1, xi ) );
		z_et = 0.5f * ( MacCormack( coord.z, 1, et ) + MacCormack( coord.z, -1, et ) );
		z_zt = 0.5f * ( MacCormack( coord.z, 1, zt ) + MacCormack( coord.z, -1, zt ) );

		Jacobi = x_xi * y_et * z_zt + x_et * y_zt * z_xi + x_zt * y_xi * z_et \
				   - x_zt * y_et * z_xi - x_et * y_xi * z_zt - x_xi * z_et * y_zt;
	  	JacobiInv = 1.0f / Jacobi;

	  	xi_x = ( y_et * z_zt - y_zt * z_et ) * JacobiInv;
		xi_y = ( x_zt * z_et - x_et * z_zt ) * JacobiInv;
		xi_z = ( x_et * y_zt - x_zt * y_et ) * JacobiInv;

	  	et_x = ( y_zt * z_xi - y_xi * z_zt ) * JacobiInv;
		et_y = ( x_xi * z_zt - x_zt * z_xi ) * JacobiInv;
		et_z = ( x_zt * y_xi - x_xi * y_zt ) * JacobiInv;

	  	zt_x = ( y_xi * z_et - y_et * z_xi ) * JacobiInv;
		zt_y = ( x_et * z_xi - x_xi * z_et ) * JacobiInv;
		zt_z = ( x_xi * y_et - x_et * y_xi ) * JacobiInv;

		Jac[index] = Jacobi;

		con.xi_x[index] = xi_x;
		con.xi_y[index] = xi_y;
		con.xi_z[index] = xi_z;
		con.et_x[index] = et_x;
		con.et_y[index] = et_y;
		con.et_z[index] = et_z;
		con.zt_x[index] = zt_x;
		con.zt_y[index] = zt_y;
		con.zt_z[index] = zt_z;
	}

}

// When change the fast axises:
/*
 * =============================================
 *             BE careful!!!!!!!!!!!!
 * =============================================
 */
// Allocate device FLOAT memory for free surface -(DZ)^(-1)*DX, -(DZ)^(-1)*DY
void allocMat3x3( const GRID * const grid, Mat3x3 * _rDZ_DX, Mat3x3 * _rDZ_DY )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	
	long long num = _nx_ * _ny_; 
		
	FLOAT * pSurf = NULL;
	long long size = sizeof( FLOAT ) * num * 2 * 9;

	checkCudaErrors( cudaMalloc((void **)&pSurf, size) );
	checkCudaErrors( cudaMemset(pSurf, 0, size) );
	
	_rDZ_DX->M11 = pSurf + 0 * num; 
	_rDZ_DX->M12 = pSurf + 1 * num; 
	_rDZ_DX->M13 = pSurf + 2 * num;
	_rDZ_DX->M21 = pSurf + 3 * num; 
	_rDZ_DX->M22 = pSurf + 4 * num; 
	_rDZ_DX->M23 = pSurf + 5 * num;
	_rDZ_DX->M31 = pSurf + 6 * num; 
	_rDZ_DX->M32 = pSurf + 7 * num; 
	_rDZ_DX->M33 = pSurf + 8 * num;

	pSurf		 = pSurf + 9 * num; 

	_rDZ_DY->M11 = pSurf + 0 * num; 
	_rDZ_DY->M12 = pSurf + 1 * num; 
	_rDZ_DY->M13 = pSurf + 2 * num;
	_rDZ_DY->M21 = pSurf + 3 * num; 
	_rDZ_DY->M22 = pSurf + 4 * num; 
	_rDZ_DY->M23 = pSurf + 5 * num;
	_rDZ_DY->M31 = pSurf + 6 * num; 
	_rDZ_DY->M32 = pSurf + 7 * num; 
	_rDZ_DY->M33 = pSurf + 8 * num;

}

void freeMat3x3( Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY )
{
	checkCudaErrors( cudaFree(_rDZ_DX.M11) );
}

// Free surface conditions for velocity
// calculate matrix -(DZ)^(-1)*DX, -(DZ)^(-1)*DY
__global__ void solve_coordinate_on_free_surface( const CONTRAVARIANT con, const COORD coord, float * const Jac, \
												  const MEDIUM medium, const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY, \
												  const int _nx_, const int _ny_, const int _nz_ ) 
{
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;
	
	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO; 
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + _nz - 1;

	long long index = 0;
	long long pos = 0;

	float DZ[9];
	float DX[9];
	float DY[9];
	float rDZ[9]; //the inverse matrix of DZ
	float DZ_det;
	float chi = 0.0f;
	float lambda = 0.0f;
	float mu = 0.0f;
	float lam_mu = 0.0f;
	
	int indexOnSurf = 0;

	float xi_x = 0.0f, et_x = 0.0f, zt_x = 0.0f;
	float xi_y = 0.0f, et_y = 0.0f, zt_y = 0.0f;
	float xi_z = 0.0f, et_z = 0.0f, zt_z = 0.0f;	


	if ( i >= HALO && i < _nx && j >= HALO && j < _ny && k >= (_nz - 1) && k < _nz_ )
	{
		index = INDEX( i, j, k );

		if ( k == _nz - 1 )
		{	
			 xi_x = con.xi_x[index];  xi_y = con.xi_y[index];  xi_z = con.xi_z[index];
			 et_x = con.et_x[index];  et_y = con.et_y[index];  et_z = con.et_z[index];
			 zt_x = con.zt_x[index];  zt_y = con.zt_y[index];  zt_z = con.zt_z[index];

			indexOnSurf = INDEX( i, j, 0 );
			lambda = medium.lambda[index];
			mu = medium.mu[index];
			chi = 2.0f * lambda + mu;//希腊字母
			lam_mu = lambda + mu;
			/****************
			---				  ---
			| DZ[0] DZ[1] DZ[2] |
			| DZ[3] DZ[4] DZ[5] |
			| DZ[6] DZ[7] DZ[8] |
			---				  ---
			*****************/
			DZ[0] = chi * zt_x * zt_x + mu * ( zt_y * zt_y + zt_z * zt_z );
			DZ[1] = lam_mu * zt_x * zt_y;
			DZ[2] = lam_mu * zt_x * zt_z;
			DZ[3] = DZ[1];
			DZ[4] = chi * zt_y * zt_y + mu * ( zt_x * zt_x + zt_z * zt_z );
			DZ[5] = lam_mu * zt_y * zt_z;
			DZ[6] = DZ[2];
			DZ[7] = DZ[5];
			DZ[8] = chi * zt_z * zt_z + mu * ( zt_x * zt_x + zt_y * zt_y );

			DZ_det = DZ[0] * DZ[4] * DZ[8]
				   + DZ[1] * DZ[5] * DZ[6]
				   + DZ[2] * DZ[7] * DZ[3]
				   - DZ[2] * DZ[4] * DZ[6]
				   - DZ[1] * DZ[3] * DZ[8]
				   - DZ[0] * DZ[7] * DZ[5];

			DX[0] = chi * zt_x * xi_x + mu * ( zt_y * xi_y + zt_z * xi_z );
			DX[1] = lambda * zt_x * xi_y + mu * zt_y * xi_x;
			DX[2] = lambda * zt_x * xi_z + mu * zt_z * xi_x;
			DX[3] = lambda * zt_y * xi_x + mu * zt_x * xi_y;
			DX[4] = chi * zt_y * xi_y + mu * ( zt_x * xi_x + zt_z * xi_z );
			DX[5] = lambda * zt_y * xi_z + mu * zt_z * xi_y;
			DX[6] = lambda * zt_z * xi_x + mu * zt_x * xi_z;
			DX[7] = lambda * zt_z * xi_y + mu * zt_y * xi_z;
			DX[8] = chi * zt_z * xi_z + mu * ( zt_x * xi_x + zt_y * xi_y );

			DY[0] = chi * zt_x * et_x + mu * ( zt_y * et_y + zt_z * et_z );
			DY[1] = lambda * zt_x * et_y + mu * zt_y * et_x;
			DY[2] = lambda * zt_x * et_z + mu * zt_z * et_x;
			DY[3] = lambda * zt_y * et_x + mu * zt_x * et_y;
			DY[4] = chi * zt_y * et_y + mu * ( zt_x * et_x + zt_z * et_z );
			DY[5] = lambda * zt_y * et_z + mu * zt_z * et_y;
			DY[6] = lambda * zt_z * et_x + mu * zt_x * et_z;
			DY[7] = lambda * zt_z * et_y + mu * zt_y * et_z;
			DY[8] = chi * zt_z * et_z + mu * ( zt_x * et_x + zt_y * et_y );

			rDZ[0] = (   DZ[4] * DZ[8] - DZ[5] * DZ[7] ) / DZ_det; 
			rDZ[1] = ( - DZ[3] * DZ[8] + DZ[5] * DZ[6] ) / DZ_det; 
			rDZ[2] = (   DZ[3] * DZ[7] - DZ[4] * DZ[6] ) / DZ_det; 
			rDZ[3] = ( - DZ[1] * DZ[8] + DZ[2] * DZ[7] ) / DZ_det; 
			rDZ[4] = (   DZ[0] * DZ[8] - DZ[2] * DZ[6] ) / DZ_det; 
			rDZ[5] = ( - DZ[0] * DZ[7] + DZ[1] * DZ[6] ) / DZ_det; 
			rDZ[6] = (   DZ[1] * DZ[5] - DZ[2] * DZ[4] ) / DZ_det; 
			rDZ[7] = ( - DZ[0] * DZ[5] + DZ[2] * DZ[3] ) / DZ_det; 
			rDZ[8] = (   DZ[0] * DZ[4] - DZ[1] * DZ[3] ) / DZ_det;
			
			// If define FP16, Implicit conversion (__float2half())
			_rDZ_DX.M11[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DX[0], DX[3], DX[6]);
			_rDZ_DX.M12[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DX[1], DX[4], DX[7]);
			_rDZ_DX.M13[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DX[2], DX[5], DX[8]);
			_rDZ_DX.M21[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DX[0], DX[3], DX[6]);
			_rDZ_DX.M22[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DX[1], DX[4], DX[7]);
			_rDZ_DX.M23[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DX[2], DX[5], DX[8]);
			_rDZ_DX.M31[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DX[0], DX[3], DX[6]);
			_rDZ_DX.M32[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DX[1], DX[4], DX[7]);
			_rDZ_DX.M33[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DX[2], DX[5], DX[8]);
			
			_rDZ_DY.M11[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DY[0], DY[3], DY[6]);
			_rDZ_DY.M12[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DY[1], DY[4], DY[7]);
			_rDZ_DY.M13[indexOnSurf] = -DOT_PRODUCT3D(rDZ[0], rDZ[1], rDZ[2], DY[2], DY[5], DY[8]);
			_rDZ_DY.M21[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DY[0], DY[3], DY[6]);
			_rDZ_DY.M22[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DY[1], DY[4], DY[7]);
			_rDZ_DY.M23[indexOnSurf] = -DOT_PRODUCT3D(rDZ[3], rDZ[4], rDZ[5], DY[2], DY[5], DY[8]);
			_rDZ_DY.M31[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DY[0], DY[3], DY[6]);
			_rDZ_DY.M32[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DY[1], DY[4], DY[7]);
			_rDZ_DY.M33[indexOnSurf] = -DOT_PRODUCT3D(rDZ[6], rDZ[7], rDZ[8], DY[2], DY[5], DY[8]);

			//if ( index == INDEX( _nx/2, _ny / 2, _nz - 1 ) )
			//	printf("_rDZ_DX = %3.10e\n", _rDZ_DX.M33[indexOnSurf]);
		}
		else
		{
			pos = INDEX( i, j, 2 * ( _nz - 1 ) - k );
			Jac[index] = Jac[pos];
			con.xi_x[index] = con.xi_x[pos]; con.xi_y[index] = con.xi_y[pos]; con.xi_z[index] = con.xi_z[pos];
			con.et_x[index] = con.et_x[pos]; con.et_y[index] = con.et_y[pos]; con.et_z[index] = con.et_z[pos];
			con.zt_x[index] = con.zt_x[pos]; con.zt_y[index] = con.zt_y[pos]; con.zt_z[index] = con.zt_z[pos];
		}

	}

}

// Solve contravariant and Jacobi matrix, and free surface matrix for velocity
#ifdef FREE_SURFACE
void solveContravariantJac( MPI_Comm comm_cart, const GRID * const grid, const CONTRAVARIANT con, \
							const COORD coord, float * const Jac, const MEDIUM medium, \
							const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY )
#else
void solveContravariantJac( MPI_Comm comm_cart, const GRID * const grid, const CONTRAVARIANT con, \
							const COORD coord, float * const Jac )
#endif
{
	float DH = grid->DH;
	float rDH = 1.0 / DH;

	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;

	dim3 threads( 32, 4, 4);
	dim3 blocks;
	blocks.x = ( nx + threads.x - 1 ) / threads.x;
	blocks.y = ( ny + threads.y - 1 ) / threads.y;
	blocks.z = ( nz + threads.z - 1 ) / threads.z;

	// Compute coordinate conversion coefficients contravariant and Jacobi
	// using the same difference format as for solving equations
	solve_con_jac <<< blocks, threads >>> ( con, coord, Jac, _nx_, _ny_, _nz_, rDH );

#ifdef FREE_SURFACE
	dim3 threadSurf( 32, 16, 1);
	dim3 blockSurf;
	blockSurf.x = ( nx + threadSurf.x - 1 ) / threadSurf.x;
	blockSurf.y = ( ny + threadSurf.y - 1 ) / threadSurf.y;
	blockSurf.z = HALO + 1;

	// Free surface conditions for velocity, calculate matrix -(DZ)^(-1)*DX, -(DZ)^(-1)*DY
	solve_coordinate_on_free_surface <<< blockSurf, threadSurf >>>
	( con, coord, Jac, medium, _rDZ_DX, _rDZ_DY, _nx_, _ny_, _nz_);
#endif // FREE_SURFACE

}


__global__ void matrix_float2FLOAT( FLOAT * const Jac_fp, float * const Jac, \
									const CONTRAVARIANT_FLOAT con_fp, const CONTRAVARIANT con, \
									const MEDIUM_FLOAT medium_fp, const MEDIUM medium, const int num )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= 0 && i < num)
	{
		Jac_fp[i] = __float2half(Jac[i]);
		
		con_fp.xi_x[i] = __float2half(con.xi_x[i]); 
		con_fp.xi_y[i] = __float2half(con.xi_y[i]); 
		con_fp.xi_z[i] = __float2half(con.xi_z[i]); 
		con_fp.et_x[i] = __float2half(con.et_x[i]);
		con_fp.et_y[i] = __float2half(con.et_y[i]);
		con_fp.et_z[i] = __float2half(con.et_z[i]);
		con_fp.zt_x[i] = __float2half(con.zt_x[i]);
		con_fp.zt_y[i] = __float2half(con.zt_y[i]);
		con_fp.zt_z[i] = __float2half(con.zt_z[i]);

		medium_fp.mu[i]       = __float2half(medium.mu[i]);
		medium_fp.lambda[i]   = __float2half(medium.lambda[i]);
		medium_fp.buoyancy[i] = __float2half(medium.buoyancy[i]);
	}

}

// Convert float(jac, con, medium) to half(jac_fp, con_fp, medium_fp)
void matrixfloat2FLOAT( const GRID * const grid, FLOAT * const Jac_fp, float * const Jac, \
						const CONTRAVARIANT_FLOAT con_fp, CONTRAVARIANT con, \
						const MEDIUM_FLOAT medium_fp, const MEDIUM medium )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	long long num = _nx_ * _ny_ * _nz_;

	dim3 threadX(512, 1, 1);
	dim3 blockX;
	blockX.x = ( num + threadX.x - 1 ) / threadX.x;
	blockX.y = 1;
	blockX.z = 1;

	matrix_float2FLOAT <<< blockX, threadX >>>
	( Jac_fp, Jac, con_fp, con, medium_fp, medium, num );

}


void mpiSendRecvCon( const GRID * const grid, MPI_Comm comm_cart, const MPI_NEIGHBOR * const mpiNeighbor, \
					 const CONTRAVARIANT_FLOAT con, const SEND_RECV_DATA sr );
void mpiSendRecvJac( const GRID * const grid, MPI_Comm comm_cart, const MPI_NEIGHBOR * const mpiNeighbor, \
					 FLOAT * const jac, const SEND_RECV_DATA sr );

// MPI send and receive contravariant and Jacobi HALO data
void MPI_SendRecv_con_jac(MPI_Comm comm_cart, const GRID * const grid, const SEND_RECV_DATA sr, \
						  const MPI_NEIGHBOR * const mpiNeighbor, const CONTRAVARIANT_FLOAT con, FLOAT * const Jac )
{
	MPI_Barrier( comm_cart );
	// send and receive contravariant HALO data
	mpiSendRecvCon( grid, comm_cart, mpiNeighbor, con, sr );
	// send and receive Jacobi HALO data
	mpiSendRecvJac( grid, comm_cart, mpiNeighbor, Jac, sr );

}
