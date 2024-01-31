#include "header.h"

// Locate data output slice
void locateSlice( const PARAMS * const params, const GRID * const grid, SLICE *slice )
{
	int sliceX = params->sliceX;
	int sliceY = params->sliceY;
	int sliceZ = params->sliceZ;
	
	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;
	
 	slice->X = sliceX - frontNX + HALO;
 	slice->Y = sliceY - frontNY + HALO;
 	slice->Z = sliceZ - frontNZ + HALO;
	
	printf( "slice.X = %d, slice.Y = %d, slice.Z = %d\n", slice->X, slice->Y, slice->Z );
}

// Locate free surface 2D slice data
void locateFreeSurfSlice( const GRID * const grid, SLICE * slice )
{
	int sliceX = -1;
	int sliceY = -1;
	int sliceZ = grid->NZ - 1;
	
	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;
	
 	slice->X = sliceX - frontNX + HALO;
 	slice->Y = sliceY - frontNY + HALO;
 	slice->Z = sliceZ - frontNZ + HALO;
	//printf( "slice.X = %d, slice.Y = %d, slice.Z = %d\n", slice->X, slice->Y, slice->Z );

}



__global__ void pack_iodata_x( FLOAT * const datain, float * const dataout, const int nx,
							   const int ny, const int nz, const int I, const float factor )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int j0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;

	int i = I;
	int j = j0 + HALO;
	int k = k0 + HALO;

	long long index, pos;
	if ( j0 >= 0 && j0 < ny && k0 >= 0 && k0 < nz )
	{
		index = INDEX( i, j, k );	
		pos = Index2D( j0, k0, ny, nz );
		dataout[pos] = (float)datain[index] * factor;
		//printf( "1:datain = %e\n", datain[pos]  );
	}

}



__global__ void pack_iodata_y( FLOAT * const datain, float * const dataout, const int nx,
							   const int ny, const int nz, const int J, const float factor )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;
	
	int i = i0 + HALO;
	int j = J;
	int k = k0 + HALO;

	long long index, pos;
	if ( i0 >= 0 && i0 < nx && k0 >= 0 && k0 < nz )
	{
		index = INDEX( i, j, k );	
		pos = Index2D( i0, k0, nx, nz );
		dataout[pos] = (float)datain[index] * factor;
		//printf( "2:datain = %e\n", datain[pos]  );
	}

}



__global__ void pack_iodata_z( FLOAT * const datain, float * const dataout, const int nx,
							   const int ny, const int nz, const int K, const float factor )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	
	int i = i0 + HALO;
	int j = j0 + HALO;
	int k = K;

	long long index, pos;
	if ( i0 >= 0 && i0 < nx && j0 >= 0 && j0 < ny )
	{
		index = INDEX( i, j, k );	
		pos = Index2D( i0, j0, nx, ny );
		dataout[pos] = (float)datain[index] * factor;
		//printf( "3:datain = %e\n", datain[index]  );
	}

}


void allocDataout( const GRID * const grid, const AXIS axis, float **dataout, const HeterArch arch )
{
	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;

	long long num = 0;
	
	switch(axis)
	{
		case XAXIS:
			num = ny * nz;
			break;
		case YAXIS:
			num = nx * nz;
			break;
		case ZAXIS:
			num = nx * ny;
			break;
	}
		
	// float * pData = NULL;
	long long size = sizeof( float ) * num;

	switch(arch)
	{
		case HOST:
			*dataout = (float *)malloc(size);
			memset( *dataout, 0, size );
			break;
		case DEVICE:
			checkCudaErrors( cudaMalloc( (void **)dataout, size ) );
			checkCudaErrors( cudaMemset( *dataout, 0, size ) );
			break;
	}
}

void freeDataout( float * dataout  )
{
	cudaFree(dataout);
}

// Allocate host/device memory for data output slice
void allocSliceData( const GRID * const grid, const SLICE slice, SLICE_DATA * sliceData, const HeterArch arch )
{
	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	if ( slice.X >= HALO && slice.X < _nx )
		allocDataout( grid, XAXIS, &( sliceData->x ), arch );
	if ( slice.Y >= HALO && slice.Y < _ny )
		allocDataout( grid, YAXIS, &( sliceData->y ), arch );
	if ( slice.Z >= HALO && slice.Z < _nz )
		allocDataout( grid, ZAXIS, &( sliceData->z ), arch );
}



void freeSliceData( const GRID * const grid, SLICE slice, SLICE_DATA sliceData_host, SLICE_DATA sliceData_dev)
{
	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	if ( slice.X >= HALO && slice.X < _nx )
		{ free(sliceData_host.x);	checkCudaErrors( cudaFree(sliceData_dev.x) ); }
	if ( slice.Y >= HALO && slice.Y < _ny )
		{ free(sliceData_host.y);	checkCudaErrors( cudaFree(sliceData_dev.y) ); }
	if ( slice.Z >= HALO && slice.Z < _nz )
		{ free(sliceData_host.z);	checkCudaErrors( cudaFree(sliceData_dev.z) ); }
}


void data2D_output_bin( const GRID * const grid, const SLICE slice, const MPI_COORD * const thisMPICoord, 
						FLOAT * const datain, const SLICE_DATA sliceData_dev,
						const SLICE_DATA sliceData_host, const char * name, const float factor )
{
	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;

	if ( slice.X >= HALO && slice.X < _nx )
	{
		dim3 threads( 32, 16, 1 );
		dim3 blocks;
		blocks.x = ( ny + threads.x - 1 ) / threads.x;
		blocks.y = ( nz + threads.y - 1 ) / threads.y;
		blocks.z = 1;
		pack_iodata_x <<< blocks, threads >>>
		( datain, sliceData_dev.x, nx, ny, nz, slice.X, factor );
		long long size = sizeof( float ) * ny * nz;
		checkCudaErrors( cudaMemcpy(sliceData_host.x, sliceData_dev.x, size, cudaMemcpyDeviceToHost) );

		FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_X_mpi_%d_%d_%d.bin", name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" ); 
		fwrite( sliceData_host.x, sizeof( float ), ny * nz, fp );
		fclose( fp );
	}

	if ( slice.Y >= HALO && slice.Y < _ny )
	{
		dim3 threads( 32, 16, 1 );
		dim3 blocks;
		blocks.x = ( nx + threads.x - 1 ) / threads.x;
		blocks.y = ( nz + threads.y - 1 ) / threads.y;
		blocks.z = 1;
		pack_iodata_y <<< blocks, threads >>>
		( datain, sliceData_dev.y, nx, ny, nz, slice.Y, factor );
		long long size = sizeof( float ) * nx * nz;
		checkCudaErrors( cudaMemcpy(sliceData_host.y, sliceData_dev.y, size, cudaMemcpyDeviceToHost) );

		FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_Y_mpi_%d_%d_%d.bin", name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" ); 
		fwrite( sliceData_host.y, sizeof( float ), nx * nz, fp );
		fclose( fp );
	}

	if ( slice.Z >= HALO && slice.Z < _nz )
	{
		dim3 threads( 32, 16, 1 );
		dim3 blocks;
		blocks.x = ( nx + threads.x - 1 ) / threads.x;
		blocks.y = ( ny + threads.y - 1 ) / threads.y;
		blocks.z = 1;
		pack_iodata_z <<< blocks, threads >>>
		( datain, sliceData_dev.z, nx, ny, nz, slice.Z, factor );
		long long size = sizeof( float ) * nx * ny;
		checkCudaErrors( cudaMemcpy(sliceData_host.z, sliceData_dev.z, size, cudaMemcpyDeviceToHost) );

		FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_Z_mpi_%d_%d_%d.bin", name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" ); 
		fwrite( sliceData_host.z, sizeof( float ), nx * ny, fp );
		fclose( fp );
	}

}


// Output wave field data
void data2D_XYZ_out( const MPI_COORD * const thisMPICoord, const PARAMS * const params,
					 const GRID * const grid, const WAVE W, const SLICE slice,
					 const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host,
					 const VTF vtf, const int it, const float ev, const float es )
{
	float cv = pow(2.0f, ev - es);
	float cs = pow(2.0f, -es);
	
	switch ( vtf )
	{
		case velocity:
			{
				char VxFileName[128], VyFileName[128], VzFileName[128];

				sprintf( VxFileName, "%s/Vx_%d", params->OUT, it );
				sprintf( VyFileName, "%s/Vy_%d", params->OUT, it );
				sprintf( VzFileName, "%s/Vz_%d", params->OUT, it );

				data2D_output_bin( grid, slice, thisMPICoord, W.Vx, sliceData_dev, sliceData_host, VxFileName, cv );
				data2D_output_bin( grid, slice, thisMPICoord, W.Vy, sliceData_dev, sliceData_host, VyFileName, cv );
				data2D_output_bin( grid, slice, thisMPICoord, W.Vz, sliceData_dev, sliceData_host, VzFileName, cv );
			}	
			break;	
		case stress:
			{
				char TxxFileName[128], TyyFileName[128], TzzFileName[128];
				char TxyFileName[128], TxzFileName[128], TyzFileName[128];

				sprintf( TxxFileName, "%s/Txx_%d", params->OUT, it );
				sprintf( TyyFileName, "%s/Tyy_%d", params->OUT, it );
				sprintf( TzzFileName, "%s/Tzz_%d", params->OUT, it );
				sprintf( TxyFileName, "%s/Txy_%d", params->OUT, it );
				sprintf( TxzFileName, "%s/Txz_%d", params->OUT, it );
				sprintf( TyzFileName, "%s/Tyz_%d", params->OUT, it );

				data2D_output_bin( grid, slice, thisMPICoord, W.Txx, sliceData_dev, sliceData_host, TxxFileName, cs );
				data2D_output_bin( grid, slice, thisMPICoord, W.Tyy, sliceData_dev, sliceData_host, TyyFileName, cs );
				data2D_output_bin( grid, slice, thisMPICoord, W.Tzz, sliceData_dev, sliceData_host, TzzFileName, cs );
				data2D_output_bin( grid, slice, thisMPICoord, W.Txy, sliceData_dev, sliceData_host, TxyFileName, cs );
				data2D_output_bin( grid, slice, thisMPICoord, W.Txz, sliceData_dev, sliceData_host, TxzFileName, cs );
				data2D_output_bin( grid, slice, thisMPICoord, W.Tyz, sliceData_dev, sliceData_host, TyzFileName, cs );
			}	
			break;	

		case freesurf:
			{
				char VxFileName[128], VyFileName[128], VzFileName[128];
				sprintf( VxFileName, "%s/FreeSurfVx_%d", params->OUT, it );
				sprintf( VyFileName, "%s/FreeSurfVy_%d", params->OUT, it );
				sprintf( VzFileName, "%s/FreeSurfVz_%d", params->OUT, it );

				data2D_output_bin( grid, slice, thisMPICoord, W.Vx, sliceData_dev, sliceData_host, VxFileName, cv );
				data2D_output_bin( grid, slice, thisMPICoord, W.Vy, sliceData_dev, sliceData_host, VyFileName, cv );
				data2D_output_bin( grid, slice, thisMPICoord, W.Vz, sliceData_dev, sliceData_host, VzFileName, cv );
			}
	}

}


