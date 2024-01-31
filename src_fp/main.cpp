#include "header.h"


int main(int argc, char **argv)
{
	PARAMS *params;
	GRID *grid;
	MPI_COORD *thisMPICoord;
	MPI_NEIGHBOR *mpiNeighbor;
	MPI_Comm comm_cart;

	params = (PARAMS *)malloc(sizeof(PARAMS));
	grid = (GRID *)malloc(sizeof(GRID));
	thisMPICoord = (MPI_COORD *)malloc(sizeof(MPI_COORD));
	mpiNeighbor = (MPI_NEIGHBOR *)malloc(sizeof(MPI_NEIGHBOR));

	// Read parameter variables from parameters json file and get some basic parameters
	getParams( params );

	// Initialize MPI and Cartesian topology construction
	init_MPI( &argc, &argv, params, &comm_cart, thisMPICoord, mpiNeighbor );

	// Initialize grid and divided into sub-grids by MPI
	init_grid( params, grid, thisMPICoord );
	
	// create data output dicretory
	createDir( params );

	// Initialize NVIDIA GPU for each MPI process
	init_gpu( grid );

	MPI_Barrier( comm_cart );

	run( comm_cart, thisMPICoord, mpiNeighbor, grid, params );
	
	free(params);
	free(grid);
	free(thisMPICoord);
	free(mpiNeighbor);
	
	MPI_Barrier( comm_cart );
	MPI_Comm_free( &comm_cart );
	MPI_Finalize( );

	return 0;
}

