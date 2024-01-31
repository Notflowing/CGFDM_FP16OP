#include "header.h"

// Allocate GPU for each MPI process
void init_gpu( const GRID * const grid )
{
	int PX = grid->PX;
	int PY = grid->PY;
	int PZ = grid->PZ;

	char jsonFile[1024] = { 0 };
#ifdef FP16
	strcpy( jsonFile, "./paramsDir/paramsCGFDM3D_fp16.json" );
#else
	strcpy( jsonFile, "./paramsDir/paramsCGFDM3D_fp32.json" );
#endif
	FILE * fp;
	fp = fopen( jsonFile, "r" );
	if ( NULL == fp )
	{
		printf( "There is not %s file!\n", jsonFile );
		MPI_Abort( MPI_COMM_WORLD, 100 );
	}
	
	fseek( fp, 0, SEEK_END );
	int len = ftell( fp );
	
	fseek( fp, 0, SEEK_SET );
	char * jsonStr = ( char * ) malloc( len * sizeof( char ) );
	if ( NULL == jsonStr )
	{
		printf( "Can't allocate json string memory\n" );
	}

	fread( jsonStr, sizeof( char ), len, fp );
	cJSON * object;
	cJSON * objArray;
	object = cJSON_Parse( jsonStr );
	if ( NULL == object )
	{
		printf( "Can't parse json file!\n");
		return;
	}
	fclose( fp );

	// numbers of nodes
	int nodeCnt = 0;
	if (objArray = cJSON_GetObjectItem(object, "gpu_nodes"))
	{
		nodeCnt = cJSON_GetArraySize( objArray );
		// printf( "nodeCnt = %d\n", nodeCnt );
	}
	
	int i, j;
	cJSON *nodeObj, *nodeItem;

	int nameLens;
	char thisMPINodeName[256];
	
	MPI_Get_processor_name( thisMPINodeName, &nameLens );
	// printf( "this mpi node name is %s\n", thisMPINodeName );

	int nodeGPUCnt = 0;
	int frontGPNCnt = 0;
	int thisRank;
	int thisNodeRankID;
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );

	//printf( "==================================" );
	for ( i = 0; i < nodeCnt; i ++  )
	{
		nodeObj    = cJSON_GetArrayItem( objArray, i );

		nodeGPUCnt = cJSON_GetArraySize( nodeObj );
		if ( 0 == strcmp( nodeObj->string, thisMPINodeName ) )
		{
			thisNodeRankID = thisRank - frontGPNCnt;
			for ( j = 0; j < nodeGPUCnt; j ++ )
			{	
				nodeItem = cJSON_GetArrayItem( nodeObj, j );
				if ( thisNodeRankID == j )
				{
					printf( "  rank %d: %s[%d] is available!\n", thisRank, nodeObj->string, nodeItem->valueint );
					checkCudaErrors( cudaSetDevice( nodeItem->valueint ) );
#ifdef FP16
					if (thisRank == 0) printf("Compute Capabilities should be 5.3 or higher!\n");
					fflush(stdout);
					checkCudaCapabilities(5, 3);
#endif
				}
			}
		}
		frontGPNCnt += nodeGPUCnt;
	}
	
	MPI_Barrier( MPI_COMM_WORLD );

	if ( frontGPNCnt != PX * PY * PZ )
	{
		printf( "The GPU numbers can't match the MPI numbers\n" );
		exit( 1 );
		//MPI_Finalize( );
	}
	
}

