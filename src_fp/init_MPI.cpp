#include "header.h"

void finalize_MPI( MPI_Comm * comm_cart )
{
	MPI_Comm_free( comm_cart );
	MPI_Finalize( );
}

// Initialize MPI and Cartesian topology construction
// Get MPI cooridinate and neighbor information
void init_MPI( int *argc, char *** argv, const PARAMS * const params, MPI_Comm * comm_cart,
			   MPI_COORD * thisMPICoord, MPI_NEIGHBOR * mpiNeighbor )
{
	int PX = params->PX;
	int PY = params->PY;
	int PZ = params->PZ;

	int thisRank, thisMPICoordXYZ[3];
	
	int nDim = 3;
	int mpiDims[3] = { PX, PY, PZ };
	int periods[3] = { 0, 0, 0 };
	int reorder = 0;
	
	MPI_Init( argc, argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );
	//MPI_Comm_size( MPI_COMM_WORLD, &nProcs );
	
	MPI_Cart_create( MPI_COMM_WORLD, nDim, mpiDims, periods, reorder, comm_cart );

	MPI_Cart_shift( *comm_cart, 0, 1, &( mpiNeighbor->X1 ), & ( mpiNeighbor->X2 ) );
	MPI_Cart_shift( *comm_cart, 1, 1, &( mpiNeighbor->Y1 ), & ( mpiNeighbor->Y2 ) );
	MPI_Cart_shift( *comm_cart, 2, 1, &( mpiNeighbor->Z1 ), & ( mpiNeighbor->Z2 ) );	

	MPI_Cart_coords( *comm_cart, thisRank, 3, thisMPICoordXYZ );

	thisMPICoord->X = thisMPICoordXYZ[0]; 
	thisMPICoord->Y = thisMPICoordXYZ[1]; 
	thisMPICoord->Z = thisMPICoordXYZ[2]; 

}

// Determine whether it is an MPI border
void isMPIBorder( const GRID * const grid, const MPI_COORD * const thisMPICoord, MPI_BORDER * border )
{
	if ( 0 == thisMPICoord->X ) border->isx1 = 1;   if ( ( grid->PX - 1 ) == thisMPICoord->X ) border->isx2 = 1;
	if ( 0 == thisMPICoord->Y ) border->isy1 = 1;   if ( ( grid->PY - 1 ) == thisMPICoord->Y ) border->isy2 = 1;
	if ( 0 == thisMPICoord->Z ) border->isz1 = 1;   if ( ( grid->PZ - 1 ) == thisMPICoord->Z ) border->isz2 = 1;

}