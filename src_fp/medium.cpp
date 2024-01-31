#include "header.h"

// Allocate host/device memory for Vs, Vp, rho
void allocStructure( const GRID * const grid, STRUCTURE * structure, const HeterArch arch )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	long long num = _nx_ * _ny_ * _nz_;
	long long size = sizeof(float) * num * MEDIUMSIZE;
	float * pStructure = NULL;

	switch(arch)
	{
		case HOST:
			pStructure = (float *)malloc(size);
			memset( pStructure, 0, size );
			break;
		case DEVICE:
			checkCudaErrors( cudaMalloc((void **)&pStructure, size) );
			checkCudaErrors( cudaMemset(pStructure, 0, size) );
			break;
	}

	structure->Vs  = pStructure;
	structure->Vp  = pStructure + num;
	structure->rho = pStructure + num * 2;

}

void freeStructure( STRUCTURE structure_host, STRUCTURE structure_dev )
{
	free(structure_host.Vs);
	checkCudaErrors( cudaFree(structure_dev.Vs) );
}

// Allocate device float memory for mu, lambda, buoyancy
void allocMedium( const GRID * const grid, MEDIUM * medium )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 
	long long size = sizeof( float ) * num * MEDIUMSIZE;

	float * pMedium = NULL;
	checkCudaErrors( cudaMalloc((void **)&pMedium, size) );
	checkCudaErrors( cudaMemset(pMedium, 0, size) );

	medium->mu       = pMedium;
	medium->lambda   = pMedium + num;
	medium->buoyancy = pMedium + num * 2;
}

void freeMedium( MEDIUM medium )
{	
	checkCudaErrors( cudaFree(medium.mu) );
}


// Allocate device FLOAT(half) memory for mu, lambda, buoyancy
void allocMedium_FP16( const GRID * const grid, MEDIUM_FLOAT * medium )
{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;
	
	long long num = _nx_ * _ny_ * _nz_; 
	long long size = sizeof( FLOAT ) * num * MEDIUMSIZE;

	FLOAT * pMedium = NULL;
	checkCudaErrors( cudaMalloc((void **)&pMedium, size) );
	checkCudaErrors( cudaMemset(pMedium, 0, size) );

	medium->mu       = pMedium;
	medium->lambda   = pMedium + num;
	medium->buoyancy = pMedium + num * 2;
}

void freeMedium_FP16( MEDIUM_FLOAT medium )
{	
	checkCudaErrors( cudaFree(medium.mu) );
}



// homogeneous medium
__global__ void construct_homo_medium( const STRUCTURE structure, const int _nx_, const int _ny_, const int _nz_)
{
	long long index = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if ( i >= 0 && i < _nx_ && j >= 0 && j < _ny_ && k >= 0 && k < _nz_)
	{
        index = INDEX( i, j, k );
		structure.Vs [index] = 3464.0f;
		structure.Vp [index] = 6000.0f;
		structure.rho[index] = 2670.0f;
    }

}

// layer medium
__global__ void construct_layer_medium( const STRUCTURE structure, const int _nx_, const int _ny_, \
										const int _nz_, const int NZ, const int frontNZ, const float DH, const float Depth )
{
	long long index = 0;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	float coordz;
	int K = 0;

	if ( i >= 0 && i < _nx_ && j >= 0 && j < _ny_ && k >= 0 && k < _nz_)
	{
        index = INDEX( i, j, k );
		K = frontNZ + k;
		coordz = ( K + 1 - HALO - NZ ) * DH;
		if ( coordz > - 0.1f * Depth )
		{
			structure.rho[index] = 2000.0f;
			structure.Vs [index] = 1800.0f;
			structure.Vp [index] = 2500.0f;
		}
		if ( coordz <= - 0.1f * Depth &&  coordz > - 0.4f * Depth)
		{
			structure.rho[index] = 2400.0f;
			structure.Vs [index] = 2500.0f;
			structure.Vp [index] = 4000.0f;
		}
		if ( coordz <= - 0.4f * Depth )
		{
			structure.rho[index] = 2800.0f;
			structure.Vs [index] = 3500.0f;
			structure.Vp [index] = 6000.0f;
		}
    }

}



// construct Structure pameters for Vs, Vp, rho
void constructStructure( const MPI_COORD * const thisMPICoord, const PARAMS * const params, \
						 const GRID * const grid, const COORD coord_host, \
						 const STRUCTURE structure_dev, const STRUCTURE structure_host )

{
	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;


	// STRUCTURE structure = { 0 };

	// structure.Vs  = medium.mu;
	// structure.Vp  = medium.lambda;
	// structure.rho = medium.buoyancy;

	//printf( "Vs: %p, Vp: %p, rho: %p\n", structure.Vs, structure.Vp, structure.rho  );

	dim3 threads( 32, 4, 4);
	dim3 blocks;
	blocks.x = ( _nx_ + threads.x - 1 ) / threads.x;
	blocks.y = ( _ny_ + threads.y - 1 ) / threads.y;
	blocks.z = ( _nz_ + threads.z - 1 ) / threads.z;

	construct_homo_medium <<< blocks, threads >>> ( structure_dev, _nx_, _ny_, _nz_ );

#ifdef LayerMedium
	construct_layer_medium <<< blocks, threads >>> ( structure_dev, _nx_, _ny_, _nz_, grid->NZ, grid->frontNZ, grid->DH, params->Depth );
#endif


	long long size = _nx_ * _ny_ * _nz_ * MEDIUMSIZE * sizeof( float );
	MPI_Barrier( MPI_COMM_WORLD );
	if ( params->useMedium )
	{
		
		if ( params->Crust_1Medel )
			readCrustal_1( *params, *grid, *thisMPICoord, coord_host, structure_host );
		if ( params->ShenModel )
			readWeisenShenModel( *params, *grid, *thisMPICoord, coord_host, structure_host );
		checkCudaErrors( cudaMemcpy( structure_dev.Vs, structure_host.Vs, size, cudaMemcpyHostToDevice ));
	}


}


__global__ void rescale_medium( const MEDIUM medium, const STRUCTURE structure, \
								const long long num, const float ev, const float dt )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float lambda, mu, buoyancy;

	if ( i >= 0 && i < num )
	{	
		mu  = structure.Vs[i];
		lambda  = structure.Vp[i];
		buoyancy = structure.rho[i];

		medium.mu[i] = pow(2.0f, ev) * dt * mu;
		medium.lambda[i] = pow(2.0f, ev) * dt * lambda;
		medium.buoyancy[i] = pow(2.0f, -ev) * dt * buoyancy;
		if( i == 100 * 100 * 100 ) {
			printf( "      mu = %f\n", medium.mu[i] );
			printf( "  lambda = %f\n", medium.lambda[i] );
			printf( "buoyancy = %f\n", medium.buoyancy[i] );
		}
	}

}


__global__ void calculate_medium(const STRUCTURE structure, const long long num )
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float Vp, Vs, rho;

	if ( i >= 0 && i < num )
	{	
		Vs  = structure.Vs[i];
		Vp  = structure.Vp[i];
		rho = structure.rho[i];

		structure.Vs[i] = rho * ( Vs * Vs );
		structure.Vp[i] = rho * ( Vp * Vp - 2.0f * Vs * Vs );
		structure.rho[i] = 1.0f / rho;
	}

}

// translate Vs Vp rho to lambda mu and bouyancy
// if define FP16, rescale medium parameters to half precision range
void vs_vp_rho2lam_mu_bou( const MEDIUM medium, const STRUCTURE structure, \
						   const GRID * const grid, const PARAMS * const params, \
						   float *ev, float *sf, const float rho_max )
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

	// translate Vs Vp rho to lambda mu and bouyancy
	calculate_medium <<< blockX, threadX >>> ( structure, num );

	*ev = 0.0f;
	float lambda_max = 0.0f;
#ifdef FP16
	long long max_index;
	int tmp_index;
	float maxValue = -1.0e20;
	float tmpMaxVl = -1.0e20;
	int blockSize = 1024 * 1024 * 1024;
	int	numBlocks = num / blockSize + 1;
	int retSize   = num % blockSize;
	int i, cnt;

	cublasHandle_t handle;
	checkCudaErrors( cublasCreate(&handle) );

	for (i = 0; i < numBlocks; i++)
	{
		long long stride = i * blockSize;
		cnt = blockSize;
		if ( i == numBlocks - 1 )
			cnt = retSize;
		checkCudaErrors( cublasIsamax(handle, cnt, structure.Vp + stride, 1, &tmp_index) );
		max_index = stride + tmp_index - 1;
		checkCudaErrors( cudaMemcpy(&tmpMaxVl, &structure.Vp[max_index], sizeof(float), cudaMemcpyDeviceToHost) );
		maxValue = MAX(maxValue, tmpMaxVl);
	}

	// get the maximum lambda value from all MPI process
	MPI_Allreduce(&maxValue, &lambda_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Barrier( MPI_COMM_WORLD );

	*sf = 1.0f / (params->DT * sqrt(lambda_max / rho_max));
	// *ev = -log2( 0.1f * params->DT * (*lambda_max) );
	*ev = -log2( (*sf) * params->DT * (lambda_max) );
#endif
	rescale_medium <<< blockX, threadX >>> ( medium, structure, num, *ev, params->DT );
	printf("lambda_max = %f, ev = %f\n", lambda_max, *ev);
	printf("new lambda = %f\n", pow(2.0f, *ev) * params->DT * (lambda_max));
	// checkCudaErrors( cudaMemcpy(medium.mu, structure.Vs, MEDIUMSIZE * num * sizeof(float), cudaMemcpyDeviceToDevice) );

}

