#include "header.h"

void createDir( const PARAMS * const params )
{
	int thisRank;
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );
	
	if ( 0 == thisRank )
	{
#if __GNUC__
 		mkdir( params->OUT, 0777 );
#elif _MSC_VER
		_mkdir( params->OUT );
#endif 
	}

	MPI_Barrier( MPI_COMM_WORLD );
	
}

