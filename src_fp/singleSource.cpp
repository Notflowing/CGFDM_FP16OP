#include "header.h"

// Locate single source
void locateSource( const PARAMS * const params, const GRID * const grid, SOURCE * source )
{
	int sourceX = params->sourceX;
	int sourceY = params->sourceY;
	int sourceZ = params->sourceZ;
	
	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;
	
 	source->X = sourceX - frontNX + HALO;
 	source->Y = sourceY - frontNY + HALO;
 	source->Z = sourceZ - frontNZ + HALO;
	printf( "source.X = %d, source.Y = %d, source.Z = %d\n", source->X, source->Y, source->Z );
	
}

__device__ float sourceFunction( const float rickerfc, const int it, const float DT )
{
    float t = it * DT;
    float tdelay = 1.2f / rickerfc;

    float f0 = 2;
    float r = PI * rickerfc * ( t - tdelay);
    float rr = r * r;
    float s = r * ( 2.0f * rr - 3.0 ) * exp( -rr ) * f0 * PI * rickerfc;
    
    float M0 = 1e16;
    s *= M0;

    return s;

}

__device__ float sourceFunction_finitefault_max( const float rickerfc, const int it, const float DT )
{
    float t = it * DT;
    float tdelay = 1.2f / rickerfc;

    float f0 = 2;
    float r = PI * rickerfc * ( t - tdelay);
    float rr = r * r;
    float s = (100 * 100 + 30 * 30) * r * ( 2.0f * rr - 3.0 ) * exp( -rr ) * f0 * PI * rickerfc;
    
    float M0 = 1e10;
    s *= M0;

    return s;

}

__device__ float sourceFunction_finitefault( const float rickerfc, const int it, const float DT, const int ddx, const int ddz )
{
    float t = it * DT;

	float dd = ddx * ddx + ddz * ddz;
    // float tdelay = 1.2f * (1.0f + (exp(dd) - exp(-dd)) / (exp(dd) + exp(-dd))) / rickerfc;

	// float tdelay = 1.2f / rickerfc;
	// float tdelay = 1.2f * (1.0f + (exp(-dd))) / rickerfc;
	float tdelay = 1.2f * (1.0f + 1.0 / (1 + dd)) / rickerfc;

    float f0 = 2;
    float r = PI * rickerfc * ( t - tdelay);
    float rr = r * r;
    float s = dd * r * ( 2.0f * rr - 3.0 ) * exp( -rr ) * f0 * PI * rickerfc;
    
    float M0 = 1e10;
    s *= M0;

    return s;

}


__global__ void load_smooth_source( const SOURCE S, const WAVE h_W, const int _nx_, const int _ny_, \
									const int _nz_, const FLOAT * Jac, const int it, const float DT, \
									const float DH, const float rickerfc, const float cs )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + S.X - nGauss;
	int j = threadIdx.y + blockIdx.y * blockDim.y + S.Y - nGauss;
	int k = threadIdx.z + blockIdx.z * blockDim.z + S.Z - nGauss;

	long long index = 0;
	float s = 0.0f;
	float value = 0.0f;
	if ( i >= (S.X - nGauss) && i < (S.X + nGauss + 1) && \
		 j >= (S.Y - nGauss) && j < (S.Y + nGauss + 1) && \
		 k >= (S.Z - nGauss) && k < (S.Z + nGauss + 1) )
	{  
		index = INDEX( i, j, k );
		s = sourceFunction( rickerfc, it, DT );
		float ra = nGauss * 0.5;
		float D1 = GAUSS_FUN(i - S.X, ra, 0.0);
		float D2 = GAUSS_FUN(j - S.Y, ra, 0.0);
		float D3 = GAUSS_FUN(k - S.Z, ra, 0.0);
		float amp = D1 * D2 * D3;

		amp /= 0.998125703461425; // # 3
		//amp /= 0.9951563131100551; // # 5
		float jacb = Jac[INDEX( S.X, S.Y, S.Z )];
    	value = -1.0f * s * amp / ( jacb * ( DH * DH * DH ) );
		
		value = value * cs * DT;

    	h_W.Txx[index] = h_W.Txx[index] + value;
		h_W.Tyy[index] = h_W.Tyy[index] + value;
		h_W.Tzz[index] = h_W.Tzz[index] + value;
	}

}

__global__ void load_finitefault_source( const SOURCE S, const WAVE h_W, const int _nx_, const int _ny_, \
										 const int _nz_, const FLOAT * Jac, const int it, const float DT, \
										 const float DH, const float rickerfc, const float cs )
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x + S.X - 100;
	int iz = threadIdx.y + blockIdx.y * blockDim.y + S.Z -  30;
	int iy = threadIdx.z + blockIdx.z * blockDim.z + S.Y;

	long long index = 0;
	float s = 0.0f;
	float value = 0.0f;
	if ( ix >= (S.X - 100) && ix < (S.X + 100 + 1) && \
		 iz >= (S.Z -  30) && iz < (S.Z +  30 + 1))
	{  
		index = INDEX( ix, iy, iz );
		s = sourceFunction_finitefault( rickerfc, it, DT, ix - S.X, iz - S.Z );
		// printf( "value = %10.10lf\n", s );
		float jacb = Jac[index];
    	value = -1.0f * s / ( jacb * ( DH * DH * DH ) );

		value = value * cs * DT;

    	h_W.Txx[index] = h_W.Txx[index] + value;
		h_W.Tyy[index] = h_W.Tyy[index] + value;
		h_W.Tzz[index] = h_W.Tzz[index] + value;
	}

}

// Load gauss smooth point source
void loadPointSource( const GRID * const grid, const SOURCE S, const WAVE h_W, \
					  const FLOAT * Jac, const int it, const float DT, \
					  const float DH, const float rickerfc, const float es )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	float cs = pow(2.0f, es);
// printf("cs = %f\n", cs);

#ifndef FINITEFAULT_TEST
	dim3 threads( 4, 4, 4);
	dim3 blocks;
	blocks.x = ( 2 * nGauss + 1 + threads.x - 1 ) / threads.x;
	blocks.y = ( 2 * nGauss + 1 + threads.y - 1 ) / threads.y;
	blocks.z = ( 2 * nGauss + 1 + threads.z - 1 ) / threads.z;
	
	if ( S.X >= HALO && S.X < _nx && S.Y >= HALO && S.Y < _ny && S.Z >= HALO && S.Z < _nz )
	{
		load_smooth_source <<< blocks, threads >>>
		( S, h_W, _nx_, _ny_, _nz_, Jac, it, DT, DH, rickerfc, cs );
	}
#else
	dim3 threads(32, 16, 1);
	dim3 blocks;

	blocks.x = ( 201 + threads.x - 1 ) / threads.x;
	blocks.y = (  61 + threads.y - 1 ) / threads.y;
	blocks.z = (   1 + threads.z - 1 ) / threads.z;

	if ( S.X >= HALO && S.X < _nx && S.Y >= HALO && S.Y < _ny && S.Z >= HALO && S.Z < _nz )
	{
		load_finitefault_source <<< blocks, threads >>>
		( S, h_W, _nx_, _ny_, _nz_, Jac, it, DT, DH, rickerfc, cs );
	}
#endif

	checkCudaErrors( cudaDeviceSynchronize() );

}


__global__ void load_smooth_source_max( const SOURCE S, float * src, const int _nx_, \
										const int _ny_, const int _nz_, const float * Jac, \
										const float DT, const float DH, const int NT, const float rickerfc)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tstep = threadIdx.y + blockIdx.y * blockDim.y;
	int gaussPoints = (nGauss * 2 + 1) * (nGauss * 2 + 1) * (nGauss * 2 + 1);

	float sval = 0.0f;
	float value = 0.0f;

	if ( index >= 0 && index < gaussPoints && tstep >= 0 and tstep < NT)
	{
		int kk = index / ( (nGauss * 2 + 1) * (nGauss * 2 + 1) );
		int indexXY = index % ( (nGauss * 2 + 1) * (nGauss * 2 + 1) );
		int jj = indexXY / (nGauss * 2 + 1);
		int ii = indexXY % (nGauss * 2 + 1);

		int i = ii + S.X - nGauss;
		int j = jj + S.Y - nGauss;
		int k = kk + S.Z - nGauss;

		long long src_idx = tstep * gaussPoints + index;
		sval = sourceFunction( rickerfc, tstep, DT );
		float ra = nGauss * 0.5;
		float D1 = GAUSS_FUN(i - S.X, ra, 0.0);
		float D2 = GAUSS_FUN(j - S.Y, ra, 0.0);
		float D3 = GAUSS_FUN(k - S.Z, ra, 0.0);
		float amp = D1 * D2 * D3;

		amp /= 0.998125703461425; // # 3
		//amp /= 0.9951563131100551; // # 5
		float jacb = Jac[INDEX( S.X, S.Y, S.Z )];
    	value = -1.0f * sval * amp / ( jacb * ( DH * DH * DH ) );

        src[src_idx] = abs(value);
	}

}

__global__ void load_finitefault_source_max( const SOURCE S, float * src, const int _nx_, \
											 const int _ny_, const int _nz_, const float * Jac, \
											 const float DT, const float DH, const int NT, const float rickerfc)
{
	int tstep = threadIdx.x + blockIdx.x * blockDim.x;

	float sval = 0.0f;
	float value = 0.0f;

	if (tstep >= 0 and tstep < NT)
	{
		sval = sourceFunction_finitefault_max( rickerfc, tstep, DT );

		float jacb = Jac[INDEX( S.X, S.Y, S.Z )];
    	value = -1.0f * sval / ( jacb * ( DH * DH * DH ) );
		// printf("value = %f\n", value);
        src[tstep] = abs(value);
	}

}

// Calculate gauss smooth source term
void loadSourceMax( const SOURCE S, float * src, const GRID * const grid, \
					const float * Jac, const float DT, const int NT, const float rickerfc )
{
	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	float DH = grid->DH;

#ifndef FINITEFAULT_TEST
	int gaussPoints = (nGauss * 2 + 1) * (nGauss * 2 + 1) * (nGauss * 2 + 1);
	dim3 threads( 32, 16, 1);
	dim3 blocks;
	blocks.x = ( gaussPoints + threads.x - 1 ) / threads.x;
	blocks.y = ( NT + threads.y - 1 ) / threads.y;
	blocks.z = 1;
	
	if ( S.X >= HALO && S.X < _nx && S.Y >= HALO && S.Y < _ny && S.Z >= HALO && S.Z < _nz )
	{
		load_smooth_source_max <<< blocks, threads >>>
		( S, src, _nx_, _ny_, _nz_, Jac, DT, DH, NT, rickerfc );
	}
#else
	dim3 threads(512, 1, 1);
	dim3 blocks;
	blocks.x = ( NT + threads.x - 1 ) / threads.x;
	if ( S.X >= HALO && S.X < _nx && S.Y >= HALO && S.Y < _ny && S.Z >= HALO && S.Z < _nz )
	{
		load_finitefault_source_max <<< blocks, threads >>>
		( S, src, _nx_, _ny_, _nz_, Jac, DT, DH, NT, rickerfc );
	}
#endif
	checkCudaErrors( cudaDeviceSynchronize() );

}


// if define FP16, rescale source term to half precision range
void getSourceMax( const SOURCE S, const PARAMS * const params, const GRID * const grid, \
				   const float * Jac, float * source_max, float * es, const float sf )
{
	int indexMax;
	float max_MPI;
	float *src;
	float DT = params->DT;
	int NT = params->TMAX / DT;

#ifndef FINITEFAULT_TEST
	int count = (nGauss * 2 + 1) * (nGauss * 2 + 1) * (nGauss * 2 + 1) * NT;
#else
	int count = NT;
#endif
	long long size = sizeof(float) * count;
	checkCudaErrors( cudaMalloc((void **)&src, size) );
	checkCudaErrors( cudaMemset(src, 0, size) );

	float rickerfc = params->rickerfc;
	loadSourceMax( S, src, grid, Jac, DT, NT, rickerfc );

	cublasHandle_t handle;
	checkCudaErrors( cublasCreate(&handle) );
	checkCudaErrors( cublasIsamax(handle, count, src, 1, &indexMax) );
	indexMax -= 1;
	checkCudaErrors( cudaMemcpy(&max_MPI, &src[indexMax], sizeof(float), cudaMemcpyDeviceToHost) );
	// printf("value = %f\n", max_MPI);

	MPI_Barrier( MPI_COMM_WORLD );
	MPI_Allreduce( &max_MPI, source_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD );
	// *es = -log2( 0.1f * params->DT * (*source_max) );
	*es = -log2( sf * params->DT * (*source_max) );
	checkCudaErrors( cudaFree(src) );
	checkCudaErrors( cublasDestroy(handle) );

	printf("source_max = %f, es = %f\n", *source_max, *es);
	printf("new source = %f\n", pow(2.0f, *es) * params->DT * (*source_max));

}

