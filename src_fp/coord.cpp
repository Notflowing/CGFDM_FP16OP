#include "header.h"

// Allocate host/device memory for x, y, z coordinate
void allocCoord( const GRID * const grid, COORD * coord, const HeterArch arch )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	long long num = _nx_ * _ny_ * _nz_; 
	long long size = sizeof( float ) * num * COORDSIZE;
	float *pCoord = NULL;

	switch(arch)
	{
		case HOST:
			pCoord = (float *)malloc(size);
			memset( pCoord, 0, size );
			break;
		case DEVICE:
			checkCudaErrors( cudaMalloc((void **)&pCoord, size) );
			checkCudaErrors( cudaMemset(pCoord, 0, size) );
			break;
	}

	coord->x = pCoord;
	coord->y = pCoord + num;
	coord->z = pCoord + num * 2;
}

void freeCoord( COORD coord_host, COORD coord_dev )
{	
	free( coord_host.x );
	checkCudaErrors( cudaFree(coord_dev.x) );
}


// construct flat coordinate grid
__global__ void construct_flat_coord( const COORD coord, const int _nx_, const int _ny_, \
									  const int _nz_, const int frontNX, const int frontNY, \
									  const int frontNZ, const int originalX, const int originalY, \
									  const int NZ, const float DH )
{
	long long index = 0;
	int I = 0, J = 0, K = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
// printf("construct_flat_coord\n");
	if ( i >= 0 && i < _nx_ && j >= 0 && j < _ny_ && k >= 0 && k < _nz_ )
	{
        index = INDEX( i, j, k );
		I = frontNX + i;
		J = frontNY + j;
		K = frontNZ + k;
		coord.x[index] = ( I - HALO ) * DH - originalX * DH;
		coord.y[index] = ( J - HALO ) * DH - originalY * DH;
		coord.z[index] = ( K - HALO + 1 ) * DH - NZ * DH;
    }
}

// construct gauss hill surface
__global__ void construct_gauss_hill_surface( float *DZ, const int _nx_, const int _ny_, \
											  const int frontNX, const int frontNY, \
											  const int originalX, const int originalY, \
											  const int NZ, const float DH, const float cal_depth )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	float x, y;
	float height = 0.0f;
	float h = 0.2f * cal_depth;
	float a = 0.1f * cal_depth, b = 0.1f * cal_depth;
	int I = 0, J = 0;
	long long index;

	if ( i >= 0 && i < _nx_ && j >= 0 && j < _ny_ )
	{        
		index = Index2D( i, j, _nx_, _ny_ );
		I = frontNX + i;
		J = frontNY + j;
		x = ( I - HALO ) * DH - originalX * DH;
		y = ( J - HALO ) * DH - originalY * DH;

		height = h * exp( -0.5f * ( x * x / ( a * a ) + y * y / ( b * b ) ) );
		DZ[index] = double( height + abs(cal_depth) ) / double( NZ - 1 );
	
	}
}


// construct terrain grid coordinate
__global__ void construct_terrain_coord( COORD coord, const float *DZ, \
										 const int _nx_, const int _ny_, const int _nz_, \
										 const int frontNZ, const int NZ, const float cal_depth )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	long long index = 0, pos = 0;
	int K;

    if ( i >= 0 && i < _nx_ && j >= 0 && j < _ny_ && k >= 0 && k < _nz_)
	{
		index = INDEX( i, j, k );
		pos = Index2D( i, j, _nx_, _ny_ );
		K = frontNZ + k - HALO; 
		//coord.z[index] = DZ[pos] * ( K + 3 * HALO  );
		coord.z[index] = -abs(cal_depth) + DZ[pos] * K;
    }

}

//void calculate_range( float * data, float range[2], long long num );

// construct host and device coordinate
void constructCoord(MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord, \
					const GRID * const grid, const PARAMS * const params, \
					const COORD coord_dev, const COORD coord_host )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;

	int originalX = grid->originalX;
	int originalY = grid->originalY;
	
	int NZ = grid->NZ;
	float DH = grid->DH;
	float cal_depth = params->Depth * 1000;
	int useTerrain = params->useTerrain;
	int gauss_hill = params->gauss_hill;

	dim3 threads(32, 4, 4);
	dim3 blocks;
	blocks.x = ( _nx_ + threads.x - 1 ) / threads.x;
	blocks.y = ( _ny_ + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ + threads.z - 1 ) / threads.z;;
	construct_flat_coord <<< blocks, threads >>>
	( coord_dev, _nx_, _ny_, _nz_, frontNX, frontNY, frontNZ, originalX, originalY, NZ, DH );

	// gauss hill topography
	if (gauss_hill)
	{
		float *DZ_dev;
		long long size = sizeof(float) * _nx_ * _ny_;
		checkCudaErrors( cudaMalloc( (void **)&DZ_dev, size ) );
		checkCudaErrors( cudaMemset( DZ_dev, 0, size ) );
// printf("%s: %d\n", __FILE__, __LINE__);
		dim3 blockXY( 32, 16, 1 );
		dim3 gridXY;
		gridXY.x = ( _nx_ + blockXY.x - 1 ) / blockXY.x;
		gridXY.y = ( _ny_ + blockXY.y - 1 ) / blockXY.y;
		gridXY.z = 1;
		construct_gauss_hill_surface <<< gridXY, blockXY >>>
		( DZ_dev, _nx_, _ny_, frontNX, frontNY, originalX, originalY, NZ, DH, cal_depth );
		//verifyDZ <<< blocks, threads >>> ( DZ, _nx_, _ny_, _nz_ );
		construct_terrain_coord <<< blocks, threads >>>
		( coord_dev, DZ_dev, _nx_, _ny_, _nz_, frontNZ, NZ, cal_depth );

		checkCudaErrors( cudaDeviceSynchronize() );
		checkCudaErrors( cudaFree(DZ_dev) );
	}

	long long size = _nx_ * _ny_ * _nz_ * COORDSIZE * sizeof(float);
	checkCudaErrors( cudaMemcpy( coord_host.x, coord_dev.x, size, cudaMemcpyDeviceToHost ) );

	if (useTerrain)
	{
		preprocessTerrain( *params, comm_cart, *thisMPICoord, *grid, coord_host );
		checkCudaErrors( cudaMemcpy( coord_dev.x, coord_host.x, size, cudaMemcpyHostToDevice ) );
	}

}
