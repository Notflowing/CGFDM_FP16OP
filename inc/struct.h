#ifndef __STRUCT__
#define __STRUCT__
#pragma once

/*
 * Define struct PARAMS
 * read parameter variables from parameters json file
 * set some basic parameters such as
 * grid meshing, MPI partition
 * time step, data output setting
 * source parameters
 * wave propagate configuration
 */
typedef struct PARAMS {
    double TMAX;
    
	double DT;
    double DH;
    
	int NX;
    int NY;
    int NZ;

    int PX;
    int PY;
    int PZ;

	int centerX;
	int centerY;

	double centerLatitude; 
	double centerLongitude;
    
	int sourceX; 
	int sourceY;
	int sourceZ;

	int IT_SKIP;
	
	int sliceX; 
	int sliceY;
	int sliceZ;
	int sliceFreeSurf;

	int nPML;

	int gauss_hill;
	int useTerrain;
	int useMedium;
	int useMultiSource;
	int useSingleSource;
	float rickerfc;

	int ShenModel;
	int Crust_1Medel;

	int itSlice;
	int itStep;
	char waveOutput[64];
	char sliceName[64];
	int itStart;
	int itEnd;
	int igpu;
	char OUT[256];

	char TerrainDir[256];
	
	int SRTM90;
	
	int lonStart;
	int latStart;
	int blockX;
	int blockY;
		
	float Depth;

	float MLonStart;
	float MLatStart;
	float MLonEnd;
	float MLatEnd;

	float MLonStep;
	float MLatStep;

	float CrustLonStep;
	float CrustLatStep;

	float MVeticalStep;
	
	char MediumDir[256];
	char crustDir[256];

	char sourceFile[256];
	char sourceDir[256];
}PARAMS;

// Grid point coordinate position
typedef struct COORD {	
	float * x;
	float * y;
	float * z;
}COORDINATE, COORD;

// Grid point latitude and longitude value
typedef struct LONLAT {
	double * lon;
	double * lat;
	double * depth;
}LONLAT;

/* 
 *Grid basic parameters
 * NX, NY, NZ represents the number of overall grid points
 * nx, ny, nz represents the number of local grid points after MPI partition
 * _nx_ -> halo + nx + halo
 * _nx  -> halo + nx
 * frontNX represents the grid points number in front of the starting point of the X-axis of this MPI coord
 */
typedef struct GRID
{
	int PX;
	int PY;
	int PZ;

	int _NX_;
	int _NY_;
	int _NZ_;

	int _NX;
	int _NY;
	int _NZ;

	int NX;
	int NY;
	int NZ;

	int _nx_;
	int _ny_;
	int _nz_;

	int _nx;
	int _ny;
	int _nz;

	int nx;
	int ny;
	int nz;

	int frontNX;
	int frontNY;
	int frontNZ;

	int _frontNX;
	int _frontNY;
	int _frontNZ;

	int originalX;
	int originalY;

	int _originalX;
	int _originalY;
	//int originalZ;

	int halo;

	int nPML;
	
	float DH;
	float rDH;
}GRID;

// The relative position of the MPI block 
typedef struct MPI_NEIGHBOR
{
	int X1;	// left
	int X2;	// right

	int Y1; // front
	int Y2; // back

	int Z1; // down
	int Z2; // up
}MPI_NEIGHBOR;

// If use netCDF format, define NCFILE
typedef struct NCFILE
{
	int ncID;

	int ntDimID;	
	int nzDimID;	
	int nyDimID;	
	int nxDimID;
	
	int VxVarID;
	int VyVarID;
	int VzVarID;
	
	int coordXVarID;
	int coordYVarID;
	int coordZVarID;
	
	int lonVarID;
	int latVarID;
}NCFILE;

typedef struct SOURCE_FILE_INPUT {
	long long npts;	// source point number
	int nt;   		// number of source time sequences of every point
	float dt; 		// time sample interval

	float * lon;
	float * lat;
	float * coordZ;

	float * area;
	float * strike;
	float * dip;

	float * rake;
	float * rate;
}SOURCE_FILE_INPUT;


typedef struct POINT_INDEX {
	int X;
	int Y;
	int Z;
}POINT_INDEX;

// define 9 seismic wave variables
// FLOAT or half
typedef struct WAVE {
	FLOAT * Vx;
	FLOAT * Vy; 
	FLOAT * Vz; 
	FLOAT * Txx;
	FLOAT * Tyy;
	FLOAT * Tzz;
	FLOAT * Txy;
	FLOAT * Txz;
	FLOAT * Tyz;
}WAVE;

typedef struct WAVE_fp32 {
	float * Vx; 
	float * Vy; 
	float * Vz; 
	float * Txx;
	float * Tyy;
	float * Tzz;
	float * Txy;
	float * Txz;
	float * Tyz;
}WAVE_fp32;

// station data output
typedef struct STATION {
	int * X;
	int * Y;
	int * Z;
#ifdef FP16
	WAVE_fp32 wave;
#else
	WAVE wave;
#endif
}STATION;

// wave for Runge-Kutta time integration
typedef struct WAVE4VAR {
	WAVE h_W;
	WAVE W;
	WAVE t_W;
	WAVE m_W;
}WAVE4VAR, WAVE4;

// ADE CFS-PML 	for seismic wave modeling
typedef struct AUXILIARY {
	FLOAT * Vx; 
	FLOAT * Vy; 
	FLOAT * Vz; 
	FLOAT * Txx;
	FLOAT * Tyy;
	FLOAT * Tzz;
	FLOAT * Txy;
	FLOAT * Txz;
	FLOAT * Tyz;
}AUXILIARY, AUX;

// PML enbeded RK4
typedef struct AUXILIARY4VAR {
	AUX h_Aux_x;
	AUX   Aux_x;
	AUX t_Aux_x;
	AUX m_Aux_x;

	AUX h_Aux_y;
	AUX   Aux_y;
	AUX t_Aux_y;
	AUX m_Aux_y;

	AUX h_Aux_z;
	AUX   Aux_z;
	AUX t_Aux_z;
	AUX m_Aux_z;
}AUXILARY4VAR, AUXILIARY4, AUX4;

typedef struct PML_ALPHA {
	float * x; 
	float * y; 
	float * z; 
}PML_ALPHA;

typedef struct PML_BETA {
	float * x; 
	float * y; 
	float * z; 
}PML_BETA;

typedef struct PML_D {
	float * x;	
	float * y;	
	float * z;	
}PML_D;

typedef struct PML_ALPHA_FLOAT {
	FLOAT * x; 
	FLOAT * y; 
	FLOAT * z; 
}PML_ALPHA_FLOAT;

typedef struct PML_BETA_FLOAT {
	FLOAT * x; 
	FLOAT * y; 
	FLOAT * z; 
}PML_BETA_FLOAT;

typedef struct PML_D_FLOAT {
	FLOAT * x;	
	FLOAT * y;	
	FLOAT * z;	
}PML_D_FLOAT;

// bool value for MPI borner
typedef struct MPI_BORDER {
	bool isx1; bool isx2;
	bool isy1; bool isy2;
	bool isz1; bool isz2;
}MPI_BORDER;

typedef struct CONTRAVARIANT_FLOAT {
	FLOAT * xi_x;
	FLOAT * xi_y;
	FLOAT * xi_z;
	FLOAT * et_x;
	FLOAT * et_y;
	FLOAT * et_z;
	FLOAT * zt_x;
	FLOAT * zt_y;
	FLOAT * zt_z;
}CONTRAVARIANT_FLOAT;

// coordinate conversion coefficient--contravariant
typedef struct CONTRAVARIANT {
	float * xi_x;
	float * xi_y;
	float * xi_z;
	float * et_x;
	float * et_y;
	float * et_z;
	float * zt_x;
	float * zt_y;
	float * zt_z;
}CONTRAVARIANT;

// free surface velocity conponent coefficient matrix
typedef struct Mat3x3 {
	FLOAT * M11; FLOAT * M12; FLOAT * M13;
	FLOAT * M21; FLOAT * M22; FLOAT * M23;
	FLOAT * M31; FLOAT * M32; FLOAT * M33;
}Mat3x3;

// 2D slice position
typedef struct SLICE {
	int X;
	int Y;
	int Z;
}SLICE;

// 2D slice data output
// typedef struct SLICE_DATA {
// 	FLOAT * x;
// 	FLOAT * y;
// 	FLOAT * z;
// }SLICE_DATA;

typedef struct SLICE_DATA {
	float * x;
	float * y;
	float * z;
}SLICE_DATA;

// source position
typedef struct SOURCE {
	int X;
	int Y;
	int Z;
}SOURCE;

// source position
typedef struct SOURCE_INDEX {
	int * X;
	int * Y;
	int * Z;
}SOURCE_INDEX;

// medium parameters: mu, lambda, buoyancy
typedef struct MEDIUM_FLOAT {
	FLOAT * mu;
	FLOAT * lambda;
	FLOAT * buoyancy;
}MEDIUM_FLOAT;

typedef struct MEDIUM
{
	float * mu;
	float * lambda;
	float * buoyancy;
}MEDIUM;

// medium parameters: Vs, Vp, rho
typedef struct STRUCTURE {
	float * Vs;
	float * Vp;
	float * rho;
}STRUCTURE;

// MPI send and recv data 
typedef struct SEND_RECV_DATA {
	FLOAT * thisXSend1;
	FLOAT * thisXRecv1;
	FLOAT * thisYSend1;
	FLOAT * thisYRecv1;
	FLOAT * thisZSend1;
	FLOAT * thisZRecv1;

	FLOAT * thisXSend2;
	FLOAT * thisXRecv2;
	FLOAT * thisYSend2;
	FLOAT * thisYRecv2;
	FLOAT * thisZSend2;
	FLOAT * thisZRecv2;
}SEND_RECV_DATA;


typedef struct  WSLICE {
	FLOAT * sliceX;
	FLOAT * sliceY;
	FLOAT * sliceZ;
}WSLICE;  


typedef struct MPI_COORDINATE {
	int X;
	int Y;
	int Z;
}MPI_COORDINATE, MPI_COORD;


typedef struct DELTA_H_RANGE {
	float * DT_min;
	float * DT_max;
}DELTA_H_RANGE;

typedef struct POINT_OR_VECTOR {
	double x;
	double y;
	double z;
}POINT_OR_VECTOR;


typedef struct SOURCE_INFO {	
	int npts;
	int nt;
	float dt;
}SOURCE_INFO;

// source moment rate
typedef struct MOMENT_RATE {	
	float * Mxx;
	float * Myy;
	float * Mzz;
	float * Mxy;
	float * Mxz;
	float * Myz;
}MOMENT_RATE;

// PGV--peek ground velocity
typedef struct PGV {
	float * pgvh;
	float * pgv;
}PGV;


#endif //__STRUCT__
