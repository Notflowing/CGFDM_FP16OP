#include "header.h"


void run( MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord, const MPI_NEIGHBOR * const mpiNeighbor, \
		  const GRID * const grid, const PARAMS * const params )
{
	int thisRank;
	// MPI_Barrier( MPI_COMM_WORLD );
	MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );

	// print grid information and check parameters configuration
	if ( thisRank == 0 )
		printInfo( grid );
	MPI_Barrier( MPI_COMM_WORLD );
	modelChecking( params );

	// set up MPI message communication
	SEND_RECV_DATA sr;	// FLOAT
	allocSendRecv( grid, &sr );

	// set up data output slices
	SLICE_DATA sliceData_host, sliceData_dev;	// float
	SLICE slice = {0};
	locateSlice( params, grid, &slice );
	allocSliceData( grid, slice, &sliceData_host, HOST );
	allocSliceData( grid, slice, &sliceData_dev, DEVICE );

	// construct coordinate
	fflush(stdout);
	if ( thisRank == 0 )
		printf( "Construct coordinate including precessing terrian data...\n" );
	MPI_Barrier( MPI_COMM_WORLD );
	COORD coord_host, coord_dev;	// float
	allocCoord( grid, &coord_host, HOST );
	allocCoord( grid, &coord_dev, DEVICE );
	// printf("%p, %p\n", coord_host->x, coord_dev->x);
	constructCoord( comm_cart, thisMPICoord, grid, params, coord_dev, coord_host );

	// construct medium
	fflush(stdout);
	if ( thisRank == 0 )
		printf( "Construct medium including precessing Vs Vp Rho...\n"  );
	MPI_Barrier( MPI_COMM_WORLD );
	STRUCTURE structure_host, structure_dev;	// float
	allocStructure( grid, &structure_host, HOST );
	allocStructure( grid, &structure_dev, DEVICE );
	constructStructure( thisMPICoord, params, grid, coord_host, structure_dev, structure_host );


	//read multisource model
	long long * srcIndex;
	MOMENT_RATE momentRate, momentRateSlice;
	SOURCE_FILE_INPUT src_in;

	if ( params->useMultiSource )
	{
		init_MultiSource( *params, *grid, *thisMPICoord, coord_host, &srcIndex, &momentRate, &momentRateSlice, &src_in );
	}

	// calculate CFL condition
	float rho_max;
	calc_CFL( grid, coord_dev, structure_dev, params, &rho_max );

	fflush(stdout);
	if ( thisRank == 0 )
		printf( "Slice Position Coordinate(x, y, z) and Medium(Vp, Vs, Rho) data output...\n"  );
	MPI_Barrier( MPI_COMM_WORLD );
	data2D_Model_out( thisMPICoord, params, grid, coord_dev, structure_dev, slice, sliceData_dev, sliceData_host );


	// translate Vs Vp rho to lambda mu and bouyancy
	// if define FP16, rescale medium
	MEDIUM medium;	// float
	float ev;	// rescale factor
	float scaleFactor;	// maximum value of lambda/rho
	allocMedium( grid, &medium );
	vs_vp_rho2lam_mu_bou( medium, structure_dev, grid, params, &ev, &scaleFactor, rho_max );

	// solve contravariant and Jacobi matrix and release coordinate memory
	float * Jac;	// float
	CONTRAVARIANT con;	// float
	allocContravariantJac( grid, &Jac, &con );

	Mat3x3 _rDZ_DX, _rDZ_DY;	// FLOAT
#ifdef FREE_SURFACE
	allocMat3x3( grid, &_rDZ_DX, &_rDZ_DY );
	solveContravariantJac( comm_cart, grid, con, coord_dev, Jac, medium, _rDZ_DX, _rDZ_DY );
#else
	solveContravariantJac( comm_cart, grid, con, coord_dev, Jac );
#endif
	freeCoord( coord_host, coord_dev );

	// ifdef FP16, convert float(jac, con, medium) to half(jac_fp, con_fp, medium_fp)
	MEDIUM_FLOAT medium_fp;
	FLOAT * Jac_fp;
	CONTRAVARIANT_FLOAT con_fp;
#ifdef FP16
	allocMedium_FP16(grid, &medium_fp);
	allocContravariantJac_FP16( grid, &Jac_fp, &con_fp );
	matrixfloat2FLOAT( grid, Jac_fp, Jac, con_fp, con, medium_fp, medium );
#else
	memcpy( &medium_fp, &medium, sizeof(MEDIUM_FLOAT) );
	Jac_fp = Jac;
	memcpy( &con_fp, &con, sizeof(CONTRAVARIANT_FLOAT) );
#endif

	// MPI send and receive contravariant and Jacobi HALO data
	MPI_SendRecv_con_jac( comm_cart, grid, sr, mpiNeighbor, con_fp, Jac_fp);
	MPI_Barrier( MPI_COMM_WORLD );

	// Allocate device FLOAT memory for wave filed wave
	WAVE h_W, W, t_W, m_W;	// FLOAT
	allocWave( grid, &h_W, &W, &t_W, &m_W );

	float DT = params->DT;
	int NT = params->TMAX / DT;
	float DH = grid->DH;
	int IT_SKIP = params->IT_SKIP;

	// Free surface and PGV
	int sliceFreeSurf = params->sliceFreeSurf;
	SLICE freeSurfSlice;
	locateFreeSurfSlice( grid, &freeSurfSlice );
	SLICE_DATA freeSurfData_dev, freeSurfData_host;	// float
	PGV pgv_dev, pgv_host;	// float

	int IsFreeSurface = 0;
#ifdef FREE_SURFACE
	if ( thisMPICoord->Z == grid->PZ - 1 )
		IsFreeSurface = 1;
#endif
	if ( IsFreeSurface )
	{
		allocatePGV( grid, &pgv_host, HOST );
		allocatePGV( grid, &pgv_dev, DEVICE );
		if ( sliceFreeSurf )
			allocSliceData( grid, freeSurfSlice, &freeSurfData_host, HOST );
			allocSliceData( grid, freeSurfSlice, &freeSurfData_dev, DEVICE );
	}

	// Initialize station information
	int stationNum;
	STATION station_dev, station_host;	// float
	stationNum = readStationIndex( grid );
	if ( stationNum > 0 )
	{
		allocStation( &station_host, stationNum, NT, HOST );
		allocStation( &station_dev , stationNum, NT, DEVICE );
		initStationIndex( grid, station_host ); 
		stationCPU2GPU( station_dev, station_host, stationNum );
	}

	// Initialize PML information
	MPI_BORDER border = { 0 };
	isMPIBorder( grid, thisMPICoord, &border );
	AUX4 Aux4_1, Aux4_2;	// FLOAT
	PML_ALPHA pml_alpha;	// float
	PML_BETA pml_beta;
	PML_D pml_d;
	PML_ALPHA_FLOAT pml_alpha_fp;	// FLOAT
	PML_BETA_FLOAT pml_beta_fp;
	PML_D_FLOAT pml_d_fp;

#ifdef PML
	allocPML( grid, &Aux4_1, &Aux4_2, border );
	allocPMLParameter( grid, &pml_alpha, &pml_beta, &pml_d );
	init_pml_parameter( params, grid, border, pml_alpha, pml_beta, pml_d );

#ifdef FP16
	allocPMLParameter_FP16( grid, &pml_alpha_fp, &pml_beta_fp, &pml_d_fp );
	PMLParameterfloat2FLOAT( grid, pml_alpha_fp, pml_alpha, pml_beta_fp, pml_beta, pml_d_fp, pml_d );
	freePMLParameter( pml_alpha, pml_beta, pml_d );
#else
	memcpy( &pml_alpha_fp, &pml_alpha, sizeof(PML_ALPHA_FLOAT) );
	memcpy( &pml_beta_fp, &pml_beta, sizeof(PML_BETA_FLOAT) );
	memcpy( &pml_d_fp, &pml_d, sizeof(PML_D_FLOAT) );
#endif

#endif

	// Initialize source information, if define FP16, rescale source term
	SOURCE S = { 0 } ;	// { _nx_ / 2, _ny_ / 2, _nz_ / 2 };
	locateSource( params, grid, &S );
	int useMultiSource = params->useMultiSource;
	int useSingleSource = params->useSingleSource;
	float source_max = 0.0f;
	float es = 0.0f;	// maximum value of source term and rescale factor

	if (useMultiSource)
	{
		calculateMomentRate( src_in, structure_dev, Jac, momentRate, srcIndex, DH );
	}

#ifdef FP16
	if (useSingleSource)
	{
		getSourceMax( S, params, grid, Jac, &source_max, &es, scaleFactor );
	}
	if ( useMultiSource )
	{
		getMultiSourceMax( src_in, momentRate, DT, &source_max, &es, scaleFactor );
	}
#endif

	freeStructure( structure_host, structure_dev );
#ifdef FP16
	freeContravariant( con );
	freeJac( Jac );
	freeMedium( medium );
#endif

	int lenGauss = nGauss * 2 + 1;
	int gaussPoints =  lenGauss * lenGauss * lenGauss;
	float *gaussFactor = (float *)malloc(gaussPoints * sizeof(float));

	int gPos = 0;
	float sumGauss = 0.0;
	float factorGauss = 0.0;
	int gaussI = 0, gaussJ = 0, gaussK = 0;
	float ra = 0.5 * nGauss;
	for( gaussK = - nGauss; gaussK < nGauss + 1; gaussK ++ )
	{
		for( gaussJ = - nGauss; gaussJ < nGauss + 1; gaussJ ++ )
		{
			for( gaussI = - nGauss; gaussI < nGauss + 1; gaussI ++ )
			{
				gPos = ( gaussI + nGauss ) + ( gaussJ + nGauss ) * lenGauss + ( gaussK + nGauss ) * lenGauss * lenGauss;
				float D1 = GAUSS_FUN( gaussI, ra, 0.0);
				float D2 = GAUSS_FUN( gaussJ, ra, 0.0);
				float D3 = GAUSS_FUN( gaussK, ra, 0.0);
				float amp = D1*D2*D3 / 0.998125703461425;				
				gaussFactor[gPos] = amp;
				sumGauss += amp;
			}
		}
	}

	if ( thisRank == 0 )
		printf( "Start calculating Wave Field:\n"  );
	MPI_Barrier( MPI_COMM_WORLD );

	// Seismic wave prppagate
	propagate( comm_cart, thisMPICoord, mpiNeighbor, grid, params, \
			   sr, S, h_W, W, t_W, m_W, _rDZ_DX, _rDZ_DY, \
			   src_in, srcIndex, momentRate, momentRateSlice, gaussFactor, \
			   con_fp, Jac_fp, medium_fp, slice, sliceData_dev, sliceData_host, \
			   freeSurfSlice, freeSurfData_dev, freeSurfData_host, pgv_dev, \
			   pgv_host, station_dev, station_host, \
			   border, Aux4_1, Aux4_2, pml_alpha_fp, pml_beta_fp, pml_d_fp, \
			   IsFreeSurface, stationNum, ev, es);

	free( gaussFactor );
	freeWave( h_W, W, t_W, m_W );
	freeSendRecv( mpiNeighbor, sr );
	freeSliceData( grid, slice, sliceData_host, sliceData_dev );

	//release con, Jac, medium memory
#ifdef FREE_SURFACE
	freeMat3x3( _rDZ_DX, _rDZ_DY );
#endif
	freeMedium_FP16( medium_fp );
	freeContravariantJac_FP16( Jac_fp, con_fp );
#ifdef PML
	freePML( border, Aux4_1, Aux4_2 );
	freePMLParamter_FP16( pml_alpha_fp, pml_beta_fp, pml_d_fp );
#endif
	
	if ( stationNum > 0 )
	{
		freeStation( station_host, station_dev );
	}
	if ( IsFreeSurface )
	{
		// outputPGV( params, grid, thisMPICoord, pgv_dev, pgv_dev );
		freePGV( pgv_host, pgv_dev );
		if ( sliceFreeSurf ) 
			freeSliceData( grid, freeSurfSlice, freeSurfData_host, freeSurfData_dev );
	}

	MPI_Barrier( MPI_COMM_WORLD );
	if ( 0 == thisRank )
	{
		printf( "Finish Run function\n"  );
	}

}
