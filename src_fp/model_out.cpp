#include "header.h"

__global__ void pack_iomodel_x( const float * const datain, float * const dataout, \
                                const int nx, const int ny, const int nz, const int Iidx )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int j0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;
	int i = Iidx;
    int j, k;

	long long index, pos;
	
    if ( j0 >= 0 && j0 < ny && k0 >= 0 && k0 < nz )
    {
		j = j0 + HALO;
		k = k0 + HALO;
		index = INDEX( i, j, k );	
		pos = Index2D( j0, k0, ny, nz );
		dataout[pos] = datain[index];
		//printf( "1:datain = %e\n", datain[pos]  );
	}

}

__global__ void pack_iomodel_y( const float * const datain, float * const dataout, \
                                const int nx, const int ny, const int nz, const int Jidx )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;
	int j = Jidx;
    int i, k;

	long long index, pos;
	
    if ( i0 >= 0 && i0 < nx && k0 >= 0 && k0 < nz )
    {
		i = i0 + HALO;
		k = k0 + HALO;
		index = INDEX( i, j, k );	
		pos = Index2D( i0, k0, nx, nz );
		dataout[pos] = datain[index];
		//printf( "1:datain = %e\n", datain[pos]  );
	}

}

__global__ void pack_iomodel_z( const float * const datain, float * const dataout, \
                                const int nx, const int ny, const int nz, const int Kidx )
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k = Kidx;
    int i, j;

	long long index, pos;
	
    if ( i0 >= 0 && i0 < nx && j0 >= 0 && j0 < ny )
    {
		i = i0 + HALO;
		j = j0 + HALO;
		index = INDEX( i, j, k );	
		pos = Index2D( i0, j0, nx, ny );
		dataout[pos] = datain[index];
		//printf( "1:datain = %e\n", datain[pos]  );
	}

}

// output 2D sliceX, sliceY, sliceZ data
void model2D_XYZ_out ( const MPI_COORD * const thisMPICoord, const GRID * const grid, \
                       const float * const dataout, const SLICE slice, \
					   const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host, \
                       const char * Name )
{
    int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
    
    int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	int nx = grid->nx;
	int ny = grid->ny;
	int nz = grid->nz;

    // long long NUM = _nx_ * _ny_ * _nz_;
    dim3 threads( 32, 16, 1 );
	dim3 blocks;

    if ( slice.X >= HALO && slice.X < _nx )
    {
        blocks.x = ( ny + threads.x - 1 ) / threads.x;
		blocks.y = ( nz + threads.y - 1 ) / threads.y;
		blocks.z = 1;
        pack_iomodel_x <<< blocks, threads >>> ( dataout, sliceData_dev.x, nx, ny, nz, slice.X );
		long long size = sizeof(float) * ny * nz;
        FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_X_mpi_%d_%d_%d.bin", Name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" );

		checkCudaErrors( cudaMemcpy( sliceData_host.x, sliceData_dev.x, size, cudaMemcpyDeviceToHost ) );
		fwrite( sliceData_host.x, sizeof(float), ny * nz, fp );
		fclose( fp );
    }

    if ( slice.Y >= HALO && slice.Y < _ny )
    {
        blocks.x = ( nx + threads.x - 1 ) / threads.x;
		blocks.y = ( nz + threads.y - 1 ) / threads.y;
		blocks.z = 1;
        pack_iomodel_y <<< blocks, threads >>> ( dataout, sliceData_dev.y, nx, ny, nz, slice.Y );
		long long size = sizeof(float) * nx * nz;
        FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_Y_mpi_%d_%d_%d.bin", Name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" );

		checkCudaErrors( cudaMemcpy( sliceData_host.y, sliceData_dev.y, size, cudaMemcpyDeviceToHost ) );
		fwrite( sliceData_host.y, sizeof(float), nx * nz, fp );
		fclose( fp );
    }

    if ( slice.Z >= HALO && slice.Z < _nz )
    {
        blocks.x = ( nx + threads.x - 1 ) / threads.x;
		blocks.y = ( ny + threads.y - 1 ) / threads.y;
		blocks.z = 1;
        pack_iomodel_z <<< blocks, threads >>> ( dataout, sliceData_dev.z, nx, ny, nz, slice.Z );
		long long size = sizeof(float) * nx * ny;
		checkCudaErrors( cudaMemcpy( sliceData_host.z, sliceData_dev.z, size, cudaMemcpyDeviceToHost ) );
    
        FILE * fp;
		char fileName[256];
		sprintf( fileName, "%s_Z_mpi_%d_%d_%d.bin", Name, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
		fp = fopen( fileName, "wb" ); 
		fwrite( sliceData_host.z, sizeof(float), nx * ny, fp );
		fclose( fp );
    }

}

// output 2D slcie data. coordinate data x, y, z and medium data Vs, Vp, rho.
void data2D_Model_out( const MPI_COORD * const thisMPICoord, const PARAMS * const params, \
					   const GRID * const grid, const COORD coord, \
					   const STRUCTURE structure, const SLICE slice, \
					   const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host )
{

	char XName[256], YName[256], ZName[256];
    sprintf( XName, "%s/coordX", params->OUT );
    sprintf( YName, "%s/coordY", params->OUT );
    sprintf( ZName, "%s/coordZ", params->OUT );

    model2D_XYZ_out( thisMPICoord, grid, coord.x, slice, sliceData_dev, sliceData_host, XName );
    model2D_XYZ_out( thisMPICoord, grid, coord.y, slice, sliceData_dev, sliceData_host, YName );
    model2D_XYZ_out( thisMPICoord, grid, coord.z, slice, sliceData_dev, sliceData_host, ZName );

    memset( XName, 0, 256 );
    memset( YName, 0, 256 );
    memset( ZName, 0, 256 );

    sprintf( XName, "%s/Vs", params->OUT );
    sprintf( YName, "%s/Vp", params->OUT );
    sprintf( ZName, "%s/rho", params->OUT );

    model2D_XYZ_out( thisMPICoord, grid, structure.Vs , slice, sliceData_dev, sliceData_host, XName );
    model2D_XYZ_out( thisMPICoord, grid, structure.Vp , slice, sliceData_dev, sliceData_host, YName );
    model2D_XYZ_out( thisMPICoord, grid, structure.rho, slice, sliceData_dev, sliceData_host, ZName );

}