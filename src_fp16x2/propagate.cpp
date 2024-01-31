#include "header.h"

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
                const int IsFreeSurface, const int stationNum, const float ev, const float es)
{
    int this_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &this_rank );

    float DT = params->DT;
	int NT = params->TMAX / DT;
	int IT_SKIP = params->IT_SKIP;
	float DH = grid->DH;

    int useMultiSource = params->useMultiSource;
	int useSingleSource = params->useSingleSource;
    int sliceFreeSurf = params->sliceFreeSurf;
    
    int it = 0, irk = 0;
	// float A, B;	// LSRK coefficient
	int FB1 = 0;    int FB2 = 0;	int FB3 = 0;	// forward, backward
	int FB[8][3] =
	{
		{ -1, -1, -1 },
		{  1,  1, -1 },
		{  1,  1,  1 },
		{ -1, -1,  1 },
		{ -1,  1, -1 },
		{  1, -1, -1 },
		{  1, -1,  1 },
		{ -1,  1,  1 },
	};	// F = 1, B = -1

    long long midClock = clock();
    long long stepClock = 0;
    long long startTime = 0, endTime = 0;
    startTime = clock();
    // time loop for wave propagete
    for (it = 0; it < NT; it++)
    {
        if( useSingleSource )
			loadPointSource( grid, S, W, Jac, it, DT, DH, params->rickerfc, es );
        if ( useMultiSource )
            addMomenteRate( grid, src_in, W, Jac, srcIndex, momentRate, momentRateSlice,
                            it, 0, DT, DH, gaussFactor, IsFreeSurface, es );

        FB1 = FB[it % 8][0];    FB2 = FB[it % 8][1];    FB3 = FB[it % 8][2];
        // Runge-Kutta loop
        for ( irk = 0; irk < 4; irk++ )
        {

            MPI_Barrier( comm_cart );
			mpiSendRecv( grid, comm_cart, mpiNeighbor, W, sr );
#ifdef PML
            waveDeriv( grid, h_W, W, con, medium, pml_beta, irk,
                       FB1, FB2, FB3, DT, IsFreeSurface );

            if ( IsFreeSurface )
                freeSurfaceDeriv( grid, h_W, W, con, medium, Jac,
                                  _rDZ_DX, _rDZ_DY, pml_beta, irk,
                                  FB1, FB2, FB3, DT );
            
            pmlDeriv( grid, h_W, W, con, medium, Aux4_1, Aux4_2,
                      pml_alpha, pml_beta, pml_d, border,
                      FB1, FB2, FB3, DT );

            if ( IsFreeSurface )
                pmlFreeSurfaceDeriv( grid, h_W, W, con, medium,
                                     Aux4_1, Aux4_2, _rDZ_DX, _rDZ_DY,
                                     pml_d, border, FB1, FB2, DT );
            
            waveRk( grid, irk, h_W.Vx, W.Vx, t_W.Vx, m_W.Vx );
			pmlRk( grid, border, irk, Aux4_1, Aux4_2 );

#else
            waveDeriv( grid, h_W, W, con, medium, irk,
                       FB1, FB2, FB3, DT, IsFreeSurface );

            if ( IsFreeSurface )
                freeSurfaceDeriv( grid, h_W, W, con, medium, Jac,
                                  _rDZ_DX, _rDZ_DY, irk,
                                  FB1, FB2, FB3, DT );

            waveRk( grid, irk, h_W.Vx, W.Vx, t_W.Vx, m_W.Vx );
#endif

            FB1 *= -1;  FB2 *= -1;  FB3 *= -1;  // reverse

        }   // For RK loop: Range Kutta Four Step

        if ( stationNum > 0 )
            storageStation( grid, NT, stationNum, station_dev, W, it, ev, es );

        if ( it % IT_SKIP == 0  )
		{
			//V mean data dump Vx Vy Vz. T means data dump Txx Tyy Tzz Txy Txz Tzz
            //F means data dump FreeSurfVx FreeSurfVy FreeSurfVz
            data2D_XYZ_out( thisMPICoord, params, grid, W, slice,
                            sliceData_dev, sliceData_host, stress, it, ev, es );
			data2D_XYZ_out( thisMPICoord, params, grid, W, slice,
                            sliceData_dev, sliceData_host, velocity, it, ev, es );
			if ( sliceFreeSurf && IsFreeSurface )
				data2D_XYZ_out( thisMPICoord, params, grid, W, freeSurfSlice,
                                freeSurfData_dev, freeSurfData_host, freesurf, it, ev, es );
		}

        MPI_Barrier( comm_cart );
        if ( ( 0 == this_rank ) && ( it % 10 == 0 ) )
        {
            printf( "it = %8d. ", it );
            stepClock = clock( ) - midClock;
            midClock  = stepClock + midClock;
            printf("Step time loss: %8.3lfs. Total time loss: %8.3lfs.\n", \
                    stepClock * 1.0 / (CLOCKS_PER_SEC * 1.0), midClock * 1.0 / (CLOCKS_PER_SEC * 1.0) );
        }

    }   // For it loop: The time iterator of NT steps
    MPI_Barrier( comm_cart );
    endTime = clock( ) - startTime;
    if ( 0 == this_rank ) {
        printf("Total elapsed time: %8.3lfs.\n", endTime * 1.0 / (CLOCKS_PER_SEC * 1.0));
    }

    if ( stationNum > 0 )
	{
		stationGPU2CPU( station_dev, station_host, stationNum, NT );
		writeStation( params, grid, thisMPICoord, station_host, NT, stationNum );
	}



}