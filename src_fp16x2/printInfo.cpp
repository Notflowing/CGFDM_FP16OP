#include "header.h"

// Print grid information
void printInfo( const GRID * const grid )
{
	printf( 
	"=============================================\n"
	"MPI:  PX = %5d, PY = %5d, PZ = %5d\n"
	"GRID: NX = %5d, NY = %5d, NZ = %5d\n"
	"DH = %5.2e\n"
	"=============================================\n",
	grid->PX, grid->PY, grid->PZ,
	grid->NX, grid->NY, grid->NZ,
	grid->DH
	);
	
}

