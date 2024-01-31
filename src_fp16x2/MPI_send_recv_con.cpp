#include "header.h"

typedef void (*PACK_UNPACK_FUNC_CON)( FLOAT * const conData, FLOAT * const thisSend, const int xStartHalo, \
										 const int _nx_, const int _ny_, const int _nz_ );


__global__ void packXCon( FLOAT * const conData, FLOAT * const thisSend, \
					   const int xStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("pack_MPI_x\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < (_nz_ * CONTRASIZE) )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ * CONTRASIZE );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		thisSend[pos] = conData[index];
	}
}

__global__ void unpackXCon( FLOAT * const conData, FLOAT * const thisRecv, \
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
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < (_nz_ * CONTRASIZE) )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ * CONTRASIZE );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		conData[index] = thisRecv[pos];
	}
}

void PackUnpackXCon( FLOAT * const conData, FLOAT * const thisSendRecv, \
					 const int xStartHalo, const int _nx_, const int _ny_, const int _nz_, \
					 PACK_UNPACK_FUNC_CON pack_unpack_func )
{
	dim3 threads( 4, 8, 16);
	dim3 blocks;
	blocks.x = ( HALO + threads.x - 1 ) / threads.x;
	blocks.y = ( _ny_ + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ * CONTRASIZE + threads.z - 1 ) / threads.z;
	pack_unpack_func <<< blocks, threads >>>
	( conData, thisSendRecv, xStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );

}


__global__ void packYCon( FLOAT * const conData, FLOAT * const thisSend, \
					   const int yStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("pack_MPI_y\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < (_nz_ * CONTRASIZE) )
	{
		i = i0;
		j = j0 + yStartHalo;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ * CONTRASIZE );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		thisSend[pos] = conData[index];
	}

}

__global__ void unpackYCon( FLOAT * const conData, FLOAT * const thisRecv, \
						 const int yStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	//printf("unpack_MPI_y\n");
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < (_nz_ * CONTRASIZE) )
	{
	    i = i0;
	    j = j0 + yStartHalo;
	    k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ * CONTRASIZE );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		conData[index] = thisRecv[pos];
	}
}

void PackUnpackYCon( FLOAT * const conData, FLOAT * const thisSendRecv, \
					 const int yStartHalo, const int _nx_, const int _ny_, const int _nz_, \
					 PACK_UNPACK_FUNC_CON pack_unpack_func )
{
	dim3 threads( 8, 4, 16);
	dim3 blocks;
	blocks.x = ( _nx_ + threads.x - 1 ) / threads.x;
	blocks.y = ( HALO + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ * WAVESIZE + threads.z - 1 ) / threads.z;
	pack_unpack_func <<< blocks, threads >>>
	( conData, thisSendRecv, yStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );
}


void packZCon( FLOAT * const conData, FLOAT * const thisSend, \
			const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen;
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_; 

	checkCudaErrors( cudaMemcpy( thisSend + 0 * blockLen, conData + 0 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 1 * blockLen, conData + 1 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 2 * blockLen, conData + 2 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 3 * blockLen, conData + 3 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 4 * blockLen, conData + 4 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 5 * blockLen, conData + 5 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 6 * blockLen, conData + 6 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 7 * blockLen, conData + 7 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 8 * blockLen, conData + 8 * num + zStartStride, size, cudaMemcpyDeviceToDevice ) );

}

void unpackZCon( FLOAT * const conData, FLOAT * const thisRecv, \
			  const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen;
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_;

	checkCudaErrors( cudaMemcpy( conData + 0 * num + zStartStride, thisRecv + 0 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 1 * num + zStartStride, thisRecv + 1 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 2 * num + zStartStride, thisRecv + 2 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 3 * num + zStartStride, thisRecv + 3 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 4 * num + zStartStride, thisRecv + 4 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 5 * num + zStartStride, thisRecv + 5 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 6 * num + zStartStride, thisRecv + 6 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 7 * num + zStartStride, thisRecv + 7 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( conData + 8 * num + zStartStride, thisRecv + 8 * blockLen, size, cudaMemcpyDeviceToDevice ) );

}

void PackUnpackZCon( FLOAT * const conData, FLOAT * const thisSendRecv, \
					 const int zStartHalo, const int _nx_, const int _ny_, const int _nz_,  \
					 PACK_UNPACK_FUNC_CON pack_unpack_func )
{
	pack_unpack_func( conData, thisSendRecv, zStartHalo, _nx_, _ny_, _nz_ );
}


void mpiSendRecvCon( const GRID * const grid, MPI_Comm comm_cart, const MPI_NEIGHBOR * const mpiNeighbor, \
					 const CONTRAVARIANT_FLOAT con, const SEND_RECV_DATA sr )
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
	FLOAT * conData = con.xi_x;
	MPI_Status stat;

//x direction data exchange

	xStartHalo = nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackXCon( conData, thisXSend2, xStartHalo, _nx_, _ny_, _nz_, packXCon );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	//printf( "======================================================\n"  );

	num = HALO * _ny_ * _nz_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X2, 101,
				  sr.thisXRecv1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X1, 101,
				  comm_cart, &stat );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	xStartHalo = 0;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackXCon( conData, thisXRecv1, xStartHalo, _nx_, _ny_, _nz_, unpackXCon );

	
	xStartHalo = HALO;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackXCon( conData, thisXSend1, xStartHalo, _nx_, _ny_, _nz_, packXCon );

	num = HALO * _ny_ * _nz_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X1, 102,
				  sr.thisXRecv2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X2, 102,
				  comm_cart, &stat );

	xStartHalo = _nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackXCon( conData, thisXRecv2, xStartHalo, _nx_, _ny_, _nz_, unpackXCon );

//y direction data exchange
	yStartHalo = ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackYCon( conData, thisYSend2, yStartHalo, _nx_, _ny_, _nz_, packYCon );

	num = HALO * _nx_ * _nz_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend2, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Y2, 103,
				  sr.thisYRecv1, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Y1, 103,
				  comm_cart, &stat );

	yStartHalo = 0;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackYCon( conData, thisYRecv1, yStartHalo, _nx_, _ny_, _nz_, unpackYCon );

	
	yStartHalo = HALO;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackYCon( conData, thisYSend1, yStartHalo, _nx_, _ny_, _nz_, packYCon );

	num = HALO * _nx_ * _nz_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend1, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Y1, 104,
				  sr.thisYRecv2, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Y2, 104,
				  comm_cart, &stat );

	yStartHalo = _ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackYCon( conData, thisYRecv2, yStartHalo, _nx_, _ny_, _nz_, unpackYCon );

//z direction data exchange
	zStartHalo = nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackZCon( conData, thisZSend2, zStartHalo, _nx_, _ny_, _nz_, packZCon );
	
	num = HALO * _nx_ * _ny_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend2, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Z2, 105,
				  sr.thisZRecv1, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Z1, 105,
				  comm_cart, &stat );

	zStartHalo = 0;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackZCon( conData, thisZRecv1, zStartHalo, _nx_, _ny_, _nz_, unpackZCon );

	
	zStartHalo = HALO;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackZCon( conData, thisZSend1, zStartHalo, _nx_, _ny_, _nz_, packZCon );

	num = HALO * _nx_ * _ny_ * CONTRASIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend1, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Z1, 106,
				  sr.thisZRecv2, num, MPI_CHAR/*MPI_float*/, mpiNeighbor->Z2, 106,
				  comm_cart, &stat );

	zStartHalo = _nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackZCon( conData, thisZRecv2, zStartHalo, _nx_, _ny_, _nz_, unpackZCon );

}




