#ifndef __FUNCTIONS__
#define __FUNCTIONS__
#pragma once


// main.cpp
void getParams(PARAMS * params);
void init_MPI( int *argc, char *** argv, const PARAMS * const params, MPI_Comm * comm_cart,
			   MPI_COORD * thisMPICoord, MPI_NEIGHBOR * mpiNeighbor );
void init_grid( const PARAMS * const params, GRID * grid, const MPI_COORD * const thisMPICoord );
void createDir( const PARAMS * const params );
void init_gpu( const GRID * const grid );

void run( MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord, const MPI_NEIGHBOR * const mpiNeighbor, \
		  const GRID * const grid, const PARAMS * const params );


// run.cpp
void printInfo( const GRID * const grid );
void modelChecking( const PARAMS * const params );
void allocSendRecv( const GRID * const grid, SEND_RECV_DATA *sr );
void locateSlice( const PARAMS * const params, const GRID * const grid, SLICE *slice );
void allocSliceData( const GRID * const grid, const SLICE slice, SLICE_DATA * sliceData, const HeterArch arch );
void allocCoord( const GRID * const grid, COORD * coord, const HeterArch arch );
void constructCoord(MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord, \
					const GRID * const grid, const PARAMS * const params, \
					const COORD coord_dev, const COORD coord_host );
void allocStructure( const GRID * const grid, STRUCTURE * structure, const HeterArch arch );
void constructStructure( const MPI_COORD * const thisMPICoord, const PARAMS * const params, \
						 const GRID * const grid, const COORD coord_host, \
						 const STRUCTURE structure_dev, const STRUCTURE structure_host );
void calc_CFL( const GRID * const grid, const COORD coord, \
			   const STRUCTURE structure, const PARAMS * const params, float * rho_globalmax );
void data2D_Model_out( const MPI_COORD * const thisMPICoord, const PARAMS * const params, \
					   const GRID * const grid, const COORD coord, \
					   const STRUCTURE structure, const SLICE slice, \
					   const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host );
void allocMedium( const GRID * const grid, MEDIUM * medium );
void vs_vp_rho2lam_mu_bou( const MEDIUM medium, const STRUCTURE structure, \
						   const GRID * const grid, const PARAMS * const params, \
						   float *ev, float *sf, const float rho_max );
void freeStructure( STRUCTURE structure_host, STRUCTURE structure_dev );
void allocContravariantJac( const GRID * const grid, float ** Jac, CONTRAVARIANT * con );
void allocMat3x3( const GRID * const grid, Mat3x3 * _rDZ_DX, Mat3x3 * _rDZ_DY );

#ifdef FREE_SURFACE
void solveContravariantJac( MPI_Comm comm_cart, const GRID * const grid, const CONTRAVARIANT con,
							const COORD coord, float * const Jac, const MEDIUM medium,
							const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY );
#else
void solveContravariantJac( MPI_Comm comm_cart, const GRID * const grid, const CONTRAVARIANT con,
							const COORD coord, float * const Jac );
#endif

void allocMedium_FP16( const GRID * const grid, MEDIUM_FLOAT * medium );
void allocContravariantJac_FP16( const GRID * const grid, FLOAT ** Jac, CONTRAVARIANT_FLOAT * con );
void matrixfloat2FLOAT( const GRID * const grid, FLOAT * const Jac_fp, float * const Jac, \
						const CONTRAVARIANT_FLOAT con_fp, CONTRAVARIANT con, \
						const MEDIUM_FLOAT medium_fp, const MEDIUM medium );
void freeContravariant( CONTRAVARIANT con );
void freeMedium( MEDIUM medium );
void MPI_SendRecv_con_jac(MPI_Comm comm_cart, const GRID * const grid, const SEND_RECV_DATA sr, \
						  const MPI_NEIGHBOR * const mpiNeighbor, const CONTRAVARIANT_FLOAT con, FLOAT * const Jac );
void freeCoord( COORD coord_host, COORD coord_dev );

void allocWave( const GRID * const grid, WAVE * h_W, WAVE * W, WAVE * t_W, WAVE * m_W );
void locateFreeSurfSlice( const GRID * const grid, SLICE * slice );
int readStationIndex( const GRID * const grid );
void allocStation( STATION * station, const int stationNum, const int NT, const HeterArch arch );
void initStationIndex( const GRID * const grid, const STATION station );
void stationCPU2GPU( const STATION station_dev, const STATION station_host, const int stationNum );
void allocatePGV( const GRID * const grid, PGV * pgv, const HeterArch arch );
void isMPIBorder( const GRID * const grid, const MPI_COORD * const thisMPICoord, MPI_BORDER * border );
void allocPML( const GRID * const grid, AUX4 *Aux4_1, AUX4 *Aux4_2, const MPI_BORDER border );
void allocPMLParameter( const GRID * const grid, PML_ALPHA * pml_alpha, PML_BETA *pml_beta, PML_D * pml_d );
void init_pml_parameter( const PARAMS * const params, const GRID * const grid, const MPI_BORDER border, \
						 const PML_ALPHA pml_alpha, const PML_BETA pml_beta, const PML_D pml_d );
void allocPMLParameter_FP16( const GRID * const grid, PML_ALPHA_FLOAT * pml_alpha, \
							 PML_BETA_FLOAT *pml_beta, PML_D_FLOAT * pml_d );
void PMLParameterfloat2FLOAT( const GRID * const grid, const PML_ALPHA_FLOAT pml_alpha_fp, const PML_ALPHA pml_alpha, \
							  const PML_BETA_FLOAT pml_beta_fp, const PML_BETA pml_beta, \
							  const PML_D_FLOAT pml_d_fp, const PML_D pml_d );
void freePMLParameter( PML_ALPHA pml_alpha, PML_BETA pml_beta, PML_D pml_d );
void locateSource( const PARAMS * const params, const GRID * const grid, SOURCE * source );
void getSourceMax( const SOURCE S, const PARAMS * const params, const GRID * const grid, \
				   const float * Jac, float * source_max, float * es, const float sf );
void freeJac( float * Jac );

void propagate( MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord, \
                const MPI_NEIGHBOR * const mpiNeighbor, const GRID * const grid, \
                const PARAMS * const params, const SEND_RECV_DATA sr, const SOURCE S, \
                const WAVE h_W, const WAVE W, const WAVE t_W, const WAVE m_W, \
                const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY, \
                const CONTRAVARIANT_FLOAT con, const FLOAT * Jac, const MEDIUM_FLOAT medium, \
                const SLICE slice, const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host, \
                const SLICE freeSurfSlice, const SLICE_DATA freeSurfData_dev, \
                const SLICE_DATA freeSurfData_host, const PGV pgv_dev, const PGV pgv_host, \
                const STATION station_dev, const STATION station_host, \
                const MPI_BORDER border, const AUX4 Aux4_1, const AUX4 Aux4_2, \
                const PML_ALPHA_FLOAT pml_alpha, const PML_BETA_FLOAT pml_beta, const PML_D_FLOAT pml_d, \
                const int IsFreeSurface, const int stationNum, const float ev, const float es);

void propagate( MPI_Comm comm_cart, const MPI_COORD * const thisMPICoord,
                const MPI_NEIGHBOR * const mpiNeighbor, const GRID * const grid,
                const PARAMS * const params, const SEND_RECV_DATA sr, const SOURCE S,
                const WAVE h_W, const WAVE W, const WAVE t_W, const WAVE m_W,
                const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,
                const SOURCE_FILE_INPUT src_in, long long * srcIndex,
                const MOMENT_RATE momentRate, const MOMENT_RATE momentRateSlice, const float * gaussFactor,
                const CONTRAVARIANT_FLOAT con, const FLOAT * Jac, const MEDIUM_FLOAT medium,
                const SLICE slice, const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host,
                const SLICE freeSurfSlice, const SLICE_DATA freeSurfData_dev,
                const SLICE_DATA freeSurfData_host, const PGV pgv_dev, const PGV pgv_host,
                const STATION station_dev, const STATION station_host,
                const MPI_BORDER border, const AUX4 Aux4_1, const AUX4 Aux4_2,
                const PML_ALPHA_FLOAT pml_alpha, const PML_BETA_FLOAT pml_beta, const PML_D_FLOAT pml_d,
                const int IsFreeSurface, const int stationNum, const float ev, const float es);

void freeWave( WAVE h_W, WAVE W, WAVE t_W, WAVE m_W );
void freeSendRecv( const MPI_NEIGHBOR * const mpiNeighbor, SEND_RECV_DATA sr );
void freeSliceData( const GRID * const grid, SLICE slice, SLICE_DATA sliceData_host, SLICE_DATA sliceData_dev);
void freeMat3x3( Mat3x3 _rDZ_DX, Mat3x3 _rDZ_DY );
void freeMedium_FP16( MEDIUM_FLOAT medium );
void freeContravariantJac_FP16( FLOAT * Jac, CONTRAVARIANT_FLOAT con );
void freePML( const MPI_BORDER border,  AUX4 Aux4_1, AUX4 Aux4_2 );
void freePMLParamter_FP16( PML_ALPHA_FLOAT pml_alpha, PML_BETA_FLOAT pml_beta, PML_D_FLOAT pml_d );
void freeStation( STATION station_host, STATION station_dev );
void freePGV( PGV pgv_host, PGV pgv_dev );

void finish_MultiSource( long long * srcIndex, MOMENT_RATE momentRate, MOMENT_RATE momentRateSlice, long long npts );
void freeSrcIndex( long long * srcIndex, long long npts );
void freeMomentRate( MOMENT_RATE momentRate, long long pointNum );
void freeMomentRateSlice( MOMENT_RATE momentRateSlice, long long npts  );


// propagate.cpp
void loadPointSource( const GRID * const grid, const SOURCE S, const WAVE h_W, \
					  const FLOAT * Jac, const int it, const float DT, \
					  const float DH, const float rickerfc, const float es );
void mpiSendRecv( const GRID * const grid, MPI_Comm comm_cart, \
				  const MPI_NEIGHBOR * const mpiNeighbor, const WAVE W, const SEND_RECV_DATA sr );

void waveDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
				const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
#ifdef PML
				const PML_BETA_FLOAT pml_beta,
#endif
				const int irk, const int FB1, const int FB2, const int FB3,
				const float A, const float B, const float DT, const int IsFreeSurface );
void waveDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
				const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
#ifdef PML
				const PML_BETA_FLOAT pml_beta,
#endif
				const int irk, const int FB1, const int FB2, const int FB3,
				const float DT, const int IsFreeSurface );

void freeSurfaceDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
					   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
					   const FLOAT * Jac, const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,
#ifdef PML
					   PML_BETA_FLOAT pml_beta,
#endif
					   const int irk, const int FB1, const int FB2, const int FB3,
					   const float A, const float B, const float DT );
void freeSurfaceDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
					   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
					   const FLOAT * Jac, const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY,
#ifdef PML
					   PML_BETA_FLOAT pml_beta,
#endif
					   const int irk, const int FB1, const int FB2, const int FB3, const float DT );

void pmlDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
			   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
			   const AUX4 Aux4_1, const AUX4 Aux4_2, const PML_ALPHA_FLOAT pml_alpha,
			   const PML_BETA_FLOAT pml_beta, const PML_D_FLOAT pml_d, const MPI_BORDER border,
			   const int FB1, const int FB2, const int FB3,
			   const float A, const float B, const float DT );
void pmlDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
			   const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
			   const AUX4 Aux4_1, const AUX4 Aux4_2, const PML_ALPHA_FLOAT pml_alpha,
			   const PML_BETA_FLOAT pml_beta, const PML_D_FLOAT pml_d, const MPI_BORDER border,
			   const int FB1, const int FB2, const int FB3,
			   const float DT );
			   
void pmlFreeSurfaceDeriv( const GRID * const grid, const WAVE h_W, const WAVE W,
						  const CONTRAVARIANT_FLOAT con, const MEDIUM_FLOAT medium,
						  const AUX4 Aux4_1, const AUX4 Aux4_2,
						  const Mat3x3 _rDZ_DX, const Mat3x3 _rDZ_DY, const PML_D_FLOAT pml_d,
						  const MPI_BORDER border, const int FB1, const int FB2, const float DT );
void newRk( const GRID * const grid, FLOAT * const h_W, FLOAT * const W, const float B, const float DT );
void newPmlRk( const GRID * const grid, const MPI_BORDER border, const int irk,
			   const AUX4 Aux4_1, const AUX4 Aux4_2, const float B, const float DT );
void waveRk( const GRID * const grid, const int irk, FLOAT * const h_W, FLOAT * const W, FLOAT * const t_W, FLOAT * const m_W );
void pmlRk( const GRID * const grid, const MPI_BORDER border, const int irk, const AUX4 Aux4_1, const AUX4 Aux4_2 );

void storageStation( const GRID * const grid, const int NT, const int stationNum,
					 const STATION station, const WAVE W, const int it,
					 const float ev, const float es );


void data2D_XYZ_out( const MPI_COORD * const thisMPICoord, const PARAMS * const params,
					 const GRID * const grid, const WAVE W, const SLICE slice,
					 const SLICE_DATA sliceData_dev, const SLICE_DATA sliceData_host,
					 const VTF vtf, const int it, const float ev, const float es );
void stationGPU2CPU( const STATION station_dev, const STATION station_host, const int stationNum, const int NT );
void writeStation( const PARAMS * const params, const GRID * const grid,
				   const MPI_COORD * const thisMPICoord, const STATION station,
				   const int NT, const int stationNum );


void mpiIrecv( const GRID * const grid, MPI_Comm comm_cart,
			   const MPI_NEIGHBOR * const mpiNeighbor, const SEND_RECV_DATA sr,
               MPI_Request * sendrecvRequest );

// real data
void preprocessTerrain( PARAMS params, MPI_Comm comm_cart, MPI_COORD thisMPICoord, GRID grid, COORD coord );
void projTrans( double lon_0, double lat_0, GRID grid, COORD coord, LONLAT LonLat );
double interp2d(double x[2], double y[2], double z[4], double x_, double y_ );
void readCrustal_1( PARAMS params, GRID grid, MPI_COORD thisMPICoord, COORD coord, STRUCTURE structure );
void readWeisenShenModel( PARAMS params, GRID grid, MPI_COORD thisMPICoord, COORD coord, STRUCTURE structure );

void init_MultiSource( PARAMS params, GRID grid, MPI_COORD thisMPICoord, COORD coord, long long ** srcIndex,
                       MOMENT_RATE * momentRate, MOMENT_RATE * momentRateSlice, SOURCE_FILE_INPUT * ret_src_in );
void calculateMomentRate( SOURCE_FILE_INPUT src_in, STRUCTURE structure_dev, float * Jac, MOMENT_RATE momentRate, long long * srcIndex, float DH );
void getMultiSourceMax( SOURCE_FILE_INPUT src_in, MOMENT_RATE momentRate, const float DT, float * source_max, float * es, const float sf );
void addMomenteRate( const GRID * const grid, SOURCE_FILE_INPUT src_in, const WAVE hW, const FLOAT * Jac, 
					 long long * srcIndex, MOMENT_RATE momentRate, MOMENT_RATE momentRateSlice, 
					 const int it, const int irk, const float DT, const float DH, const float * gaussFactor,
					 const int flagSurf, const float es );

#endif  // __FUNCTIONS__