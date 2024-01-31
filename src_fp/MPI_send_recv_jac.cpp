#include "header.h"

typedef void (*PACK_UNPACK_FUNC_JAC)( FLOAT * const jac, FLOAT * const thisSend, \
									  const int xStartHalo, const int _nx_, const int _ny_, const int _nz_ );


__global__ void packJacX( FLOAT * const jac, FLOAT * const thisSend, \
						  const int xStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("packJac_MPI_x\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < _nz_ )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		thisSend[pos] = jac[index];
	}

}

__global__ void unpackJacX( FLOAT * const jac, FLOAT * const thisRecv, \
						 const int xStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < _nz_ )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		jac[index] = thisRecv[pos];
	}

}

void PackUnpackJacX( FLOAT * const jac, FLOAT * const thisSendRecv, \
					 const int xStartHalo, const int _nx_, const int _ny_, const int _nz_, \
					 PACK_UNPACK_FUNC_JAC packJac_unpackJac_func )
{
	dim3 threads( 4, 8, 16);
	dim3 blocks;
	blocks.x = ( HALO + threads.x - 1 ) / threads.x;
	blocks.y = ( _ny_ + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ + threads.z - 1 ) / threads.z;
	packJac_unpackJac_func <<< blocks, threads >>>
	( jac, thisSendRecv, xStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );

}


__global__ void packJacY( FLOAT * const jac, FLOAT * const thisSend, \
						  const int yStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("packJac_MPI_y\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < _nz_ )
	{
		i = i0;
		j = j0 + yStartHalo;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		thisSend[pos] = jac[index];
	}

}

__global__ void unpackJacY( FLOAT * const jac, FLOAT * const thisRecv, \
						 const int yStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("unpackJac_MPI_y\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < _nz_ )
	{
	    i = i0;
	    j = j0 + yStartHalo;
	    k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		jac[index] = thisRecv[pos];
	}

}

void PackUnpackJacY( FLOAT * const jac, FLOAT * const thisSendRecv, \
					 const int yStartHalo, const int _nx_, const int _ny_, const int _nz_, \
					 PACK_UNPACK_FUNC_JAC packJac_unpackJac_func )
{
	dim3 threads( 8, 4, 16);
	dim3 blocks;
	blocks.x = ( _nx_ + threads.x - 1 ) / threads.x;
	blocks.y = ( HALO + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ + threads.z - 1 ) / threads.z;
	packJac_unpackJac_func <<< blocks, threads >>>
	( jac, thisSendRecv, yStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );

}

void packJacZ( FLOAT * const jac, FLOAT * const thisSend, \
			   const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen; 
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_; 

	checkCudaErrors( cudaMemcpy( thisSend + 0 * blockLen, jac + zStartStride, size, cudaMemcpyDeviceToDevice ) );

}

void unpackJacZ( FLOAT * const jac, FLOAT * const thisRecv, \
				 const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen; 
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_; 

	checkCudaErrors( cudaMemcpy( jac + zStartStride, thisRecv + 0 * blockLen, size, cudaMemcpyDeviceToDevice ) );

}

void PackUnpackJacZ( FLOAT * const jac, FLOAT * const thisSendRecv, \
					 const int zStartHalo, const int _nx_, const int _ny_, const int _nz_, \
					 PACK_UNPACK_FUNC_JAC packJac_unpackJac_func )
{
	packJac_unpackJac_func( jac, thisSendRecv, zStartHalo, _nx_, _ny_, _nz_ );

}


void mpiSendRecvJac( const GRID * const grid, MPI_Comm comm_cart, const MPI_NEIGHBOR * const mpiNeighbor, \
					 FLOAT * const jac, const SEND_RECV_DATA sr )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;
	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	long long num = 0;

	FLOAT * thisXSend1 = sr.thisXSend1;
	FLOAT * thisXRecv1 = sr.thisXRecv1;
	FLOAT * thisYSend1 = sr.thisYSend1;
	FLOAT * thisYRecv1 = sr.thisYRecv1;
	FLOAT * thisZSend1 = sr.thisZSend1;
	FLOAT * thisZRecv1 = sr.thisZRecv1;
                                 
	FLOAT * thisXSend2 = sr.thisXSend2;
	FLOAT * thisXRecv2 = sr.thisXRecv2;
	FLOAT * thisYSend2 = sr.thisYSend2;
	FLOAT * thisYRecv2 = sr.thisYRecv2;
	FLOAT * thisZSend2 = sr.thisZSend2;
	FLOAT * thisZRecv2 = sr.thisZRecv2;

	int xStartHalo, yStartHalo, zStartHalo;


	MPI_Status stat;

//x direction data exchange

	xStartHalo = nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackJacX( jac, thisXSend2, xStartHalo, _nx_, _ny_, _nz_, packJacX );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	//printf( "======================================================\n"  );

	num = HALO * _ny_ * _nz_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend2, num, MPI_CHAR, mpiNeighbor->X2, 101,
				  sr.thisXRecv1, num, MPI_CHAR, mpiNeighbor->X1, 101,
				  comm_cart, &stat );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	xStartHalo = 0;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackJacX( jac, thisXRecv1, xStartHalo, _nx_, _ny_, _nz_, unpackJacX );

	
	xStartHalo = HALO;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackJacX( jac, thisXSend1, xStartHalo, _nx_, _ny_, _nz_, packJacX );

	num = HALO * _ny_ * _nz_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend1, num, MPI_CHAR, mpiNeighbor->X1, 102,
				  sr.thisXRecv2, num, MPI_CHAR, mpiNeighbor->X2, 102,
				  comm_cart, &stat );

	xStartHalo = _nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackJacX( jac, thisXRecv2, xStartHalo, _nx_, _ny_, _nz_, unpackJacX );

//y direction data exchange
	yStartHalo = ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackJacY( jac, thisYSend2, yStartHalo, _nx_, _ny_, _nz_, packJacY );

	num = HALO * _nx_ * _nz_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend2, num, MPI_CHAR, mpiNeighbor->Y2, 103,
				  sr.thisYRecv1, num, MPI_CHAR, mpiNeighbor->Y1, 103,
				  comm_cart, &stat );

	yStartHalo = 0;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackJacY( jac, thisYRecv1, yStartHalo, _nx_, _ny_, _nz_, unpackJacY );

	
	yStartHalo = HALO;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackJacY( jac, thisYSend1, yStartHalo, _nx_, _ny_, _nz_, packJacY );

	num = HALO * _nx_ * _nz_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend1, num, MPI_CHAR, mpiNeighbor->Y1, 104,
				  sr.thisYRecv2, num, MPI_CHAR, mpiNeighbor->Y2, 104,
				  comm_cart, &stat );

	yStartHalo = _ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackJacY( jac, thisYRecv2, yStartHalo, _nx_, _ny_, _nz_, unpackJacY );

//z direction data exchange
	zStartHalo = nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackJacZ( jac, thisZSend2, zStartHalo, _nx_, _ny_, _nz_, packJacZ );
	
	num = HALO * _nx_ * _ny_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend2, num, MPI_CHAR, mpiNeighbor->Z2, 105,
				  sr.thisZRecv1, num, MPI_CHAR, mpiNeighbor->Z1, 105,
				  comm_cart, &stat );

	zStartHalo = 0;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackJacZ( jac, thisZRecv1, zStartHalo, _nx_, _ny_, _nz_, unpackJacZ );

	
	zStartHalo = HALO;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackJacZ( jac, thisZSend1, zStartHalo, _nx_, _ny_, _nz_, packJacZ );

	num = HALO * _nx_ * _ny_ * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend1, num, MPI_CHAR, mpiNeighbor->Z1, 106,
				  sr.thisZRecv2, num, MPI_CHAR, mpiNeighbor->Z2, 106,
				  comm_cart, &stat );

	zStartHalo = _nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackJacZ( jac, thisZRecv2, zStartHalo, _nx_, _ny_, _nz_, unpackJacZ );

}




