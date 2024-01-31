#include "header.h"

typedef void (*PACK_UNPACK_FUNC)( FLOAT * const waveData, FLOAT * const thisSend, const int xStartHalo, \
								  const int _nx_, const int _ny_, const int _nz_ );


__global__ void packX( FLOAT * const waveData, FLOAT * const thisSend, \
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
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < (_nz_ * WAVESIZE) )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ * WAVESIZE );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		thisSend[pos] = waveData[index];
	}
}

__global__ void unpackX( FLOAT * const waveData, FLOAT * const thisRecv, \
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
	
	if ( i0 >= 0 && i0 < HALO && j0 >= 0 && j0 < _ny_ && k0 >= 0 && k0 < (_nz_ * WAVESIZE) )
	{
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, HALO, _ny_, _nz_ * WAVESIZE );//i - xStartHalo + j * HALO + k * HALO * _ny_;
		waveData[index] = thisRecv[pos];
	}
}

void PackUnpackX( FLOAT * const waveData, FLOAT * const thisSendRecv, \
				  const int xStartHalo, const int _nx_, const int _ny_, const int _nz_, \
				  PACK_UNPACK_FUNC pack_unpack_func )
{
	dim3 threads( 4, 8, 16);
	dim3 blocks;
	blocks.x = ( HALO + threads.x - 1 ) / threads.x;
	blocks.y = ( _ny_ + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ * WAVESIZE + threads.z - 1 ) / threads.z;
	pack_unpack_func <<< blocks, threads >>>
	( waveData, thisSendRecv, xStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );

}


__global__ void packY( FLOAT * const waveData, FLOAT * const thisSend, \
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

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < (_nz_ * WAVESIZE) )
	{
		i = i0;
		j = j0 + yStartHalo;
		k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ * WAVESIZE );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		thisSend[pos] = waveData[index];
	}

}

__global__ void unpackY( FLOAT * const waveData, FLOAT * const thisRecv, \
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

	if ( i0 >= 0 && i0 < _nx_ && j0 >= 0 && j0 < HALO && k0 >= 0 && k0 < (_nz_ * WAVESIZE) )
	{
	    i = i0;
	    j = j0 + yStartHalo;
	    k = k0;
		index = INDEX( i, j, k );
		pos = Index3D( i0, j0, k0, _nx_, HALO, _nz_ * WAVESIZE );//i + ( j - yStartHalo ) * _nx_ + k * HALO * _nx_;
		waveData[index] = thisRecv[pos];
	}

}

void PackUnpackY( FLOAT * const waveData, FLOAT * const thisSendRecv, \
				  const int yStartHalo, const int _nx_, const int _ny_, const int _nz_, \
				  PACK_UNPACK_FUNC pack_unpack_func )
{
	dim3 threads( 8, 4, 16);
	dim3 blocks;
	blocks.x = ( _nx_ + threads.x - 1 ) / threads.x;
	blocks.y = ( HALO + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ * WAVESIZE + threads.z - 1 ) / threads.z;
	pack_unpack_func <<< blocks, threads >>>
	( waveData, thisSendRecv, yStartHalo, _nx_, _ny_, _nz_ );
	checkCudaErrors( cudaDeviceSynchronize() );

}


void packZ( FLOAT * const waveData, FLOAT * const thisSend, \
			const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen; 
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_; 

	WAVE W = { 0 };

	W.Vx  = waveData + 0 * num;
	W.Vy  = waveData + 1 * num;
	W.Vz  = waveData + 2 * num;
	W.Txx = waveData + 3 * num;
	W.Tyy = waveData + 4 * num;
	W.Tzz = waveData + 5 * num;
	W.Txy = waveData + 6 * num;
	W.Txz = waveData + 7 * num;
	W.Tyz = waveData + 8 * num;

	checkCudaErrors( cudaMemcpy( thisSend + 0 * blockLen, W.Vx  + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 1 * blockLen, W.Vy  + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 2 * blockLen, W.Vz  + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 3 * blockLen, W.Txx + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 4 * blockLen, W.Tyy + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 5 * blockLen, W.Tzz + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 6 * blockLen, W.Txy + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 7 * blockLen, W.Txz + zStartStride, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( thisSend + 8 * blockLen, W.Tyz + zStartStride, size, cudaMemcpyDeviceToDevice ) );

}

void unpackZ( FLOAT * const waveData, FLOAT * const thisRecv, \
			  const int zStartHalo, const int _nx_, const int _ny_, const int _nz_ )
{
	long long blockLen = _nx_ * _ny_ * HALO;
	long long size = sizeof( FLOAT ) * blockLen; 
	long long zStartStride = zStartHalo * _nx_ * _ny_;
	long long num = _nx_ * _ny_ * _nz_;

	WAVE W = { 0 };

	W.Vx  = waveData + 0 * num;
	W.Vy  = waveData + 1 * num;
	W.Vz  = waveData + 2 * num;
	W.Txx = waveData + 3 * num;
	W.Tyy = waveData + 4 * num;
	W.Tzz = waveData + 5 * num;
	W.Txy = waveData + 6 * num;
	W.Txz = waveData + 7 * num;
	W.Tyz = waveData + 8 * num;

	checkCudaErrors( cudaMemcpy( W.Vx  + zStartStride, thisRecv + 0 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Vy  + zStartStride, thisRecv + 1 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Vz  + zStartStride, thisRecv + 2 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Txx + zStartStride, thisRecv + 3 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Tyy + zStartStride, thisRecv + 4 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Tzz + zStartStride, thisRecv + 5 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Txy + zStartStride, thisRecv + 6 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Txz + zStartStride, thisRecv + 7 * blockLen, size, cudaMemcpyDeviceToDevice ) );
	checkCudaErrors( cudaMemcpy( W.Tyz + zStartStride, thisRecv + 8 * blockLen, size, cudaMemcpyDeviceToDevice ) );

}

void PackUnpackZ( FLOAT * const waveData, FLOAT * const thisSendRecv, \
				  const int zStartHalo, const int _nx_, const int _ny_, const int _nz_,  \
				  PACK_UNPACK_FUNC pack_unpack_func )
{
	pack_unpack_func( waveData, thisSendRecv, zStartHalo, _nx_, _ny_, _nz_ );
}


void alloc_send_recv( const GRID * const grid, FLOAT **send, FLOAT **recv, const AXIS axis )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	long long num = 0;
	
	switch( axis )
	{
		case XAXIS:
			num = _ny_ * _nz_* HALO * WAVESIZE;
			break;       
		case YAXIS:         
			num = _nx_ * _nz_* HALO * WAVESIZE;
			break;       
		case ZAXIS:         
			num = _nx_ * _ny_* HALO * WAVESIZE;
			break;
	}
		
	long long size = sizeof( FLOAT ) * num;

	checkCudaErrors( cudaMalloc( (void **)send, size ) );
	checkCudaErrors( cudaMemset( *send, 0, size ) );

	checkCudaErrors( cudaMalloc( (void **)recv, size ) );
	checkCudaErrors( cudaMemset( *recv, 0, size ) );

}

// Allcoate MPI send and receive devicememory
void allocSendRecv( const GRID * const grid, SEND_RECV_DATA *sr )
{
	memset( sr, 0, sizeof( SEND_RECV_DATA ) );
	
	alloc_send_recv( grid, &( sr->thisXSend1 ), &( sr->thisXRecv1 ), XAXIS );
	alloc_send_recv( grid, &( sr->thisYSend1 ), &( sr->thisYRecv1 ), YAXIS );
	alloc_send_recv( grid, &( sr->thisZSend1 ), &( sr->thisZRecv1 ), ZAXIS ); 

	alloc_send_recv( grid, &( sr->thisXSend2 ), &( sr->thisXRecv2 ), XAXIS );
	alloc_send_recv( grid, &( sr->thisYSend2 ), &( sr->thisYRecv2 ), YAXIS );
	alloc_send_recv( grid, &( sr->thisZSend2 ), &( sr->thisZRecv2 ), ZAXIS );
	
}

void freeSendRecv( const MPI_NEIGHBOR * const mpiNeighbor, SEND_RECV_DATA sr )
{
	cudaFree( sr.thisXSend1);	cudaFree( sr.thisXRecv1 );
	cudaFree( sr.thisYSend1);	cudaFree( sr.thisYRecv1 );
	cudaFree( sr.thisZSend1);	cudaFree( sr.thisZRecv1 );                             
	cudaFree( sr.thisXSend2);	cudaFree( sr.thisXRecv2 );
	cudaFree( sr.thisYSend2);	cudaFree( sr.thisYRecv2 );
	cudaFree( sr.thisZSend2);	cudaFree( sr.thisZRecv2 );
	
}

// MPI send and recv wave field data(FLOAT)
void mpiSendRecv( const GRID * const grid, MPI_Comm comm_cart, \
				  const MPI_NEIGHBOR * const mpiNeighbor, const WAVE W, const SEND_RECV_DATA sr )
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
	FLOAT * waveData = W.Vx;
	MPI_Status stat;

//x direction data exchange

	xStartHalo = nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackX( waveData, thisXSend2, xStartHalo, _nx_, _ny_, _nz_, packX );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	//printf( "======================================================\n"  );

	num = HALO * _ny_ * _nz_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X2, 101,
				  sr.thisXRecv1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X1, 101,
				  comm_cart, &stat );

	//printf( "X1 = %d, X2 = %d\n", mpiNeighbor.X1, mpiNeighbor.X2  );
	xStartHalo = 0;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackX( waveData, thisXRecv1, xStartHalo, _nx_, _ny_, _nz_, unpackX );

	
	xStartHalo = HALO;
	if ( mpiNeighbor->X1 >= 0 ) PackUnpackX( waveData, thisXSend1, xStartHalo, _nx_, _ny_, _nz_, packX );

	num = HALO * _ny_ * _nz_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisXSend1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X1, 102,
				  sr.thisXRecv2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->X2, 102,
				  comm_cart, &stat );

	xStartHalo = _nx;
	if ( mpiNeighbor->X2 >= 0 ) PackUnpackX( waveData, thisXRecv2, xStartHalo, _nx_, _ny_, _nz_, unpackX );

//y direction data exchange
	yStartHalo = ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackY( waveData, thisYSend2, yStartHalo, _nx_, _ny_, _nz_, packY );

	num = HALO * _nx_ * _nz_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Y2, 103,
				  sr.thisYRecv1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Y1, 103,
				  comm_cart, &stat );

	yStartHalo = 0;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackY( waveData, thisYRecv1, yStartHalo, _nx_, _ny_, _nz_, unpackY );


	yStartHalo = HALO;
	if ( mpiNeighbor->Y1 >= 0 ) PackUnpackY( waveData, thisYSend1, yStartHalo, _nx_, _ny_, _nz_, packY );

	num = HALO * _nx_ * _nz_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisYSend1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Y1, 104,
				  sr.thisYRecv2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Y2, 104,
				  comm_cart, &stat );

	yStartHalo = _ny;
	if ( mpiNeighbor->Y2 >= 0 ) PackUnpackY( waveData, thisYRecv2, yStartHalo, _nx_, _ny_, _nz_, unpackY );

//z direction data exchange
	zStartHalo = nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackZ( waveData, thisZSend2, zStartHalo, _nx_, _ny_, _nz_, packZ );
	
	num = HALO * _nx_ * _ny_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Z2, 105,
				  sr.thisZRecv1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Z1, 105,
				  comm_cart, &stat );

	zStartHalo = 0;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackZ( waveData, thisZRecv1, zStartHalo, _nx_, _ny_, _nz_, unpackZ );

	
	zStartHalo = HALO;
	if ( mpiNeighbor->Z1 >= 0 ) PackUnpackZ( waveData, thisZSend1, zStartHalo, _nx_, _ny_, _nz_, packZ );

	num = HALO * _nx_ * _ny_ * WAVESIZE * sizeof( FLOAT );
	MPI_Sendrecv( sr.thisZSend1, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Z1, 106,
				  sr.thisZRecv2, num, MPI_CHAR/*MPI_FLOAT*/, mpiNeighbor->Z2, 106,
				  comm_cart, &stat );

	zStartHalo = _nz;
	if ( mpiNeighbor->Z2 >= 0 ) PackUnpackZ( waveData, thisZRecv2, zStartHalo, _nx_, _ny_, _nz_, unpackZ );
	
}




