#include "header.h"

// Check model configuration information from parameters json file
void modelChecking( const PARAMS * const params )
{
	int thisRank;
	MPI_Barrier( MPI_COMM_WORLD );
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );

	if ( params->useMultiSource && params->useSingleSource )
	{
		if ( 0 == thisRank )
			printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"	
					"You set \"useMultiSource\" and \"useSingleSource\" at the same time. Please check the json file of parameters configuration!\n"
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" );	
        MPI_Barrier( MPI_COMM_WORLD );
		MPI_Abort( MPI_COMM_WORLD, 180 );
	}
	if ( !( params->useMultiSource || params->useSingleSource ) )
	{
		if ( 0 == thisRank )
			printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"	
				    "You did not set any source. The program will abort!\n" 
				    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" );	
		MPI_Barrier( MPI_COMM_WORLD );
		MPI_Abort( MPI_COMM_WORLD, 180 );
	}
	if ( params->ShenModel && params->Crust_1Medel )
	{
		if ( 0 == thisRank )
			printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"	
				    "You set ShenModel and Crust_1Medel both to be 1. Please check the json file of parameters configuration!\n" 
				    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" );	
		MPI_Barrier( MPI_COMM_WORLD );
        MPI_Abort( MPI_COMM_WORLD, 180 );
	}
	if ( params->useTerrain && params->gauss_hill )
	{
		if ( 0 == thisRank )
            printf( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
		            "You set \"useTerrain\" and \"gauss_hill\" both to be 1 at the same time. Please check the json file of parameters configuration!\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" );
		MPI_Barrier( MPI_COMM_WORLD );
        MPI_Abort( MPI_COMM_WORLD, 180 );
	}

	if ( params->useTerrain == 0 )
	{
		if ( 0 == thisRank )
			printf( "No SRTM90 Terrain model is used!\n" );
	}
	if ( params->useTerrain == 1 )
	{
		if ( 0 == thisRank )
			printf( "SRTM90 Terrain model is used!\n" );
	}
	if ( params->useMultiSource == 0 )
	{
		if ( 0 == thisRank )
			printf( "No Multi-source model is used!\n" );
	}
	if ( params->useMultiSource == 1 )
	{
		if ( 0 == thisRank )
			printf( "Multi-source model is used!\n" );
	}
	if ( params->ShenModel == 0 )
	{
		if ( 0 == thisRank )
			printf( "No ShenModel is used!\n" );
	}
	if ( params->ShenModel == 1 )
	{
		if ( 0 == thisRank )
			printf( "ShenModel is used!\n" );
	}
	if ( params->Crust_1Medel == 0 )
	{
		if ( 0 == thisRank )
			printf( "No Crust_1Medel is used!\n" );
	}
	if ( params->Crust_1Medel == 1 )
	{
		if ( 0 == thisRank )
			printf( "Crust_1Medel is used!\n" );
	}
	MPI_Barrier( MPI_COMM_WORLD );

}