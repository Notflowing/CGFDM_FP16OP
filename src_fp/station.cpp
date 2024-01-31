#include "header.h"

// Allocate host/device float memory for stations data
void allocStation( STATION * station, const int stationNum, const int NT, const HeterArch arch )
{
	int sizeIdx = sizeof( int ) * stationNum * 3;
	int * pIndex = NULL;

	long long sizeWave = sizeof( float ) * NT * stationNum * WAVESIZE;
	float * pWave = NULL;

	switch(arch)
	{
		case HOST:
			pIndex = (int *)malloc(sizeIdx);
			pWave = (float *)malloc(sizeWave);
			memset( pIndex, 0, sizeIdx );
			memset( pWave, 0, sizeWave );
			break;
		case DEVICE:
			checkCudaErrors( cudaMalloc( (void **)&pIndex, sizeIdx ) );
			checkCudaErrors( cudaMalloc( (void **)&pWave, sizeWave ) );
			checkCudaErrors( cudaMemset( pIndex, 0, sizeIdx ) );
			checkCudaErrors( cudaMemset( pWave, 0, sizeWave ) );
			break;
	}

	station->X = pIndex;
	station->Y = pIndex + stationNum;
	station->Z = pIndex + stationNum * 2;

	station->wave.Vx  = pWave + NT * stationNum * 0; 
	station->wave.Vy  = pWave + NT * stationNum * 1; 
	station->wave.Vz  = pWave + NT * stationNum * 2; 
	station->wave.Txx = pWave + NT * stationNum * 3;
	station->wave.Tyy = pWave + NT * stationNum * 4;
	station->wave.Tzz = pWave + NT * stationNum * 5;
	station->wave.Txy = pWave + NT * stationNum * 6;
	station->wave.Txz = pWave + NT * stationNum * 7;
	station->wave.Tyz = pWave + NT * stationNum * 8;

}


void freeStation( STATION station_host, STATION station_dev )
{
	free(station_host.X);
	free(station_host.wave.Vx);
	checkCudaErrors( cudaFree(station_dev.X) );
	checkCudaErrors( cudaFree(station_dev.wave.Vx) );
}


int readStationIndex( const GRID * const grid )
{
	char jsonFile[1024] = { 0 };
	strcpy( jsonFile, "./stationsDir/station.json" );
	FILE * fp;
	fp = fopen( jsonFile, "r" );
	if ( NULL == fp )
	{
		printf( "There is not %s file!\n", jsonFile );
		return 0;
	}
	
	fseek( fp, 0, SEEK_END );
	int len = ftell( fp );
	
	fseek( fp, 0, SEEK_SET );
	char * jsonStr = ( char * ) malloc( len * sizeof( char ) );

	if ( NULL == jsonStr )
	{
		printf( "Can't allocate json string memory\n" );
		return 0;
	}

	fread( jsonStr, sizeof( char ), len, fp );
	//printf( "%s\n", jsonStr );
	cJSON * object;
	cJSON * objArray;

	object = cJSON_Parse( jsonStr );
	if ( NULL == object )
	{
		printf( "Can't parse json file!\n");
		//exit( 1 );
		return 0;
	}
	fclose( fp );

	int stationCnt= 0;
	if (objArray = cJSON_GetObjectItem(object, "station(point)"))
	{
		stationCnt = cJSON_GetArraySize( objArray );
	}

	cJSON *stationObj, *stationItem;
	int i, j, stationNum;
	int X, Y, Z, thisX, thisY, thisZ;
	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;

	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;

	stationNum = 0;
	for ( i = 0; i < stationCnt; i ++  )
	{
		stationObj = cJSON_GetArrayItem( objArray, i );

		int a = cJSON_GetArraySize( stationObj );
		if ( a != 3 )
		{
			printf( "In file %s, the coodinate index don't equal to 3. However, it equals to %d\n", jsonFile, a );
			return 0;
		}
	
		stationItem = cJSON_GetArrayItem( stationObj, 0 );
		X = stationItem->valueint;
		thisX = X - frontNX + HALO;

		stationItem = cJSON_GetArrayItem( stationObj, 1 );
		Y = stationItem->valueint;
		thisY = Y - frontNY + HALO;

		stationItem = cJSON_GetArrayItem( stationObj, 2 );
		Z = stationItem->valueint;
		thisZ = Z - frontNZ + HALO;
			
		if ( thisX >= HALO && thisX < _nx &&
			 thisY >= HALO && thisY < _ny &&
			 thisZ >= HALO && thisZ < _nz )
		{
			stationNum ++;
		}

	}
	//printf( "stationNum = %d\n", stationNum );
	return stationNum;

}


void initStationIndex( const GRID * const grid, const STATION station ) 
{
	char jsonFile[1024] = { 0 };
	strcpy( jsonFile, "./stationsDir/station.json" );
	FILE * fp;
	fp = fopen( jsonFile, "r" );
	if ( NULL == fp )
	{
		printf( "There is not %s file!\n", jsonFile );
		return;
	}
	
	fseek( fp, 0, SEEK_END );
	int len = ftell( fp );
	
	fseek( fp, 0, SEEK_SET );
	char * jsonStr = ( char * ) malloc( len * sizeof( char ) );

	if ( NULL == jsonStr )
	{
		printf( "Can't allocate json string memory\n" );
		return;
	}

	fread( jsonStr, sizeof( char ), len, fp );
	//printf( "%s\n", jsonStr );
	cJSON * object;
	cJSON * objArray;

	object = cJSON_Parse( jsonStr );
	if ( NULL == object )
	{
		printf( "Can't parse json file!\n");
		//exit( 1 );	
		return;
	}
	fclose( fp );

	int stationCnt= 0;

	if (objArray = cJSON_GetObjectItem(object, "station(point)"))
	{
		stationCnt = cJSON_GetArraySize( objArray );
		//printf( "stationCnt = %d\n", stationCnt );
	}

	cJSON *stationObj, *stationItem;
	int i, j;
	int thisX, thisY, thisZ;
	
	int frontNX = grid->frontNX;
	int frontNY = grid->frontNY;
	int frontNZ = grid->frontNZ;

	int _nx = grid->_nx;
	int _ny = grid->_ny;
	int _nz = grid->_nz;
	
	int X, Y, Z;
	//printf( "X = %p\n", station.X  );
	//printf( "Y = %p\n", station.Y  );
	//printf( "Z = %p\n", station.Z  );

	int stationIdx = 0;
	for ( i = 0; i < stationCnt; i ++  )
	{
		stationObj = cJSON_GetArrayItem( objArray, i );

		int a = cJSON_GetArraySize( stationObj );
		if ( a != 3 )
		{
			printf( "In file %s, the coodinate index don't equal to 3. However, it equals to %d\n", jsonFile, a );
			return;
		}
	
		stationItem = cJSON_GetArrayItem( stationObj, 0 );
		X = stationItem->valueint;
		thisX = X - frontNX + HALO;

		stationItem = cJSON_GetArrayItem( stationObj, 1 );
		Y = stationItem->valueint;
		thisY = Y - frontNY + HALO;

		stationItem = cJSON_GetArrayItem( stationObj, 2 );
		Z = stationItem->valueint;
		thisZ = Z - frontNZ + HALO;
			
		if ( thisX >= HALO && thisX < _nx &&
			 thisY >= HALO && thisY < _ny &&
			 thisZ >= HALO && thisZ < _nz )
		{
			//printf( "X = %p\n", station.X  );
			//printf( "Y = %p\n", station.Y  );
			//printf( "Z = %p\n", station.Z  );
			station.X[stationIdx] = thisX;
			station.Y[stationIdx] = thisY;
			station.Z[stationIdx] = thisZ;
		
			stationIdx ++;
		}
	}

}

void stationCPU2GPU( const STATION station_dev, const STATION station_host, const int stationNum )
{
	int size = sizeof( int ) * stationNum * 3;
	//int i =0;
	//for ( i = 0; i < stationNum; i ++ )
	//{
	//	printf( "X = %d, Y = %d, Z = %d\n", station_cpu.X[i], station_cpu.Y[i], station_cpu.Z[i] );
	//}
	//printf( "=============size = %d =========\n", size );
	checkCudaErrors( cudaMemcpy( station_dev.X, station_host.X, size, cudaMemcpyHostToDevice ) );
	
}



__global__ void storage_station( const int stationNum, const STATION station, const WAVE W,
								 const int _nx_, const int _ny_, const int _nz_,
								 const int NT, const int it, const float cv, const float cs )
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	long long index = 0, pos = 0;
	int X, Y, Z;

	if ( i >= 0 && i < stationNum )
	{
		X = station.X[i];
		Y = station.Y[i];
		Z = station.Z[i];

		index = INDEX( X, Y, Z );
		pos = it + i * NT;

		station.wave.Vx [pos] = (float)W.Vx [index] * cv;
		station.wave.Vy [pos] = (float)W.Vy [index] * cv;
		station.wave.Vz [pos] = (float)W.Vz [index] * cv;
		station.wave.Txx[pos] = (float)W.Txx[index] * cs;
		station.wave.Tyy[pos] = (float)W.Tyy[index] * cs;
		station.wave.Tzz[pos] = (float)W.Tzz[index] * cs;
		station.wave.Txy[pos] = (float)W.Txy[index] * cs;
		station.wave.Txz[pos] = (float)W.Txz[index] * cs;
		station.wave.Tyz[pos] = (float)W.Tyz[index] * cs;
	}

}

// Storage station wave field data(Vx, Vy, Vz, Txx, Tyy, Tzz, Txy, Txz, Tyz)
void storageStation( const GRID * const grid, const int NT, const int stationNum,
					 const STATION station, const WAVE W, const int it,
					 const float ev, const float es )
{
	long long num = stationNum;

	int _nx_ = grid->_nx_;
	int _ny_ = grid->_ny_;
	int _nz_ = grid->_nz_;

	float cv = pow(2.0f, ev - es);
	float cs = pow(2.0f, -es);

	dim3 threads( 8, 8, 8 );
	dim3 blocks;
	blocks.x = ( num + threads.x - 1 ) / threads.x;
	blocks.y = 1;
	blocks.z = 1;
	storage_station <<< blocks, threads >>>
	( stationNum, station, W, _nx_, _ny_, _nz_, NT, it, cv, cs );
	checkCudaErrors( cudaDeviceSynchronize() );

}

void stationGPU2CPU( const STATION station_dev, const STATION station_host, const int stationNum, const int NT )
{
	long long sizeWave = sizeof( float ) * NT * stationNum * WAVESIZE;
	checkCudaErrors( cudaMemcpy(station_host.wave.Vx, station_dev.wave.Vx, sizeWave, cudaMemcpyDeviceToHost) );
}

// write station wave data into binary file
void writeStation( const PARAMS * const params, const GRID * const grid,
				   const MPI_COORD * const thisMPICoord, const STATION station,
				   const int NT, const int stationNum )
{	
	FILE * fp;
	char fileName[256];
	sprintf( fileName, "%s/station_mpi_%d_%d_%d.bin", params->OUT, thisMPICoord->X, thisMPICoord->Y, thisMPICoord->Z );
	int i = 0;
	for ( i = 0; i < stationNum; i ++ )
	{
		station.X[i] = grid->frontNX + station.X[i] - HALO;
		station.Y[i] = grid->frontNY + station.Y[i] - HALO;
		station.Z[i] = grid->frontNZ + station.Z[i] - HALO;
	}

	fp = fopen( fileName, "wb" );
	fwrite( station.X, sizeof( int ), stationNum * 3, fp );
	fwrite( station.wave.Vx, sizeof( float ), NT * stationNum * WAVESIZE, fp );
	fclose( fp );

}



