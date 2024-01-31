#include "header.h"

// Allocate host/device float memory for PGV
void allocatePGV( const GRID * const grid, PGV * pgv, const HeterArch arch )
{
	int nx = grid->nx;
	int ny = grid->ny;

	int size = sizeof( float ) * nx * ny * 2;
	int num = nx * ny;

	float * pPgv = NULL;

	switch(arch)
	{
		case HOST:
			pPgv = (float *)malloc(size);
			memset( pPgv, 0, size );
			break;
		case DEVICE:
			checkCudaErrors( cudaMalloc( (void **)&pPgv, size ) );
			checkCudaErrors( cudaMemset(pPgv, 0, size) );
			break;
	}

	pgv->pgvh = pPgv;
	pgv->pgv  = pgv->pgvh + num;
	//printf( "pgvh = %p, pgv = %p\n",  pgv->pgvh, pgv->pgv );

}

void freePGV( PGV pgv_host, PGV pgv_dev )
{
	free(pgv_host.pgvh);
	checkCudaErrors( cudaFree( pgv_dev.pgvh ) );
}


