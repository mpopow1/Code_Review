#pragma once

#include <stdio.h>

//Conversions
#define PI 3.1415926535897932384626433832795
#define DEG_TO_RAD (PI/180.0)
#define RAD_TO_DEG (180.0/PI)
#define RAD_TO_ARC (206264.806247)
#define ARC_TO_RAD (0.0000048481368110954)
#define FEET_TO_METER (0.3048)
#define METER_TO_FEET (3.28084)
#define TEN_THOUSANDTH_DEGREE_IN_RAD (0.000001276)
#define THOUSANDTH_DEGREE_IN_RAD (0.00001745329)

//----------------------------------------------------------
// GPS WGS84 Earth ellipsoid constants - https://en.wikipedia.org/wiki/World_Geodetic_System
// semi-major axis in meters
#define WGS84_A (6378137.0)
// semi-minor axis in meters
#define WGS84_B (6356752.3142)
// flattening
#define WGS84_F (1/298.257223563)
// first eccentricity squared
#define WGS84_E2 (0.00669437999014)
//----------------------------------------------------------

#define ROTATION_MATRIX_NUM_ELEMENTS   (9)
#define ROTATION_MATRIX_SIZE           (ROTATION_MATRIX_NUM_ELEMENTS * sizeof(float))
#define LOS_VECTOR_MATRIX_NUM_ELEMENTS (3)
#define LOS_VECTOR_MATRIX_SIZE         (LOS_VECTOR_MATRIX_NUM_ELEMENTS  * sizeof(float))

typedef struct {
	//Known Inputs
	float A_yaw;       //In Radians, clockwise 0-360, 0 is true north
	float A_pitch;     //In Radians, down is negative
	float A_roll;      //In Radians, counter clockwise is negative
	float A_azimuth;   //In Radians, clockwise 0-360, 0 is forward relative to aircraft body
	float A_elevation; //In Radians, down is negative, relative to aircraft body
	float A_latitude;  //In Radians
	float A_longitude; //In Radians
	float A_altitude;  //Mean Sea Level, In meters

	//Intermediate Inputs - calculated from the above inputs
	float LOS_Rotation[3]; //aligns aircraft LOS to the Earth frame of reference
	float RN;		// reference slant range
	float RM;       // reference slant range

	//Target Location Data - Unknowns - Will be calculated by GPU kernel
	float slantRange;
	float T_latitude;
	float T_longitude;
	float T_altitude; //a.k.a. terrain elevation
}Aircraft_Data;

#define NUM_BLOCKS 1
#define NUM_THREADS 512
#define OUTPUT_SIZE (NUM_BLOCKS * NUM_THREADS)
//Sets the size of each step in meters along the LOS vector, default = 1m
#define RANGE_RESOLUTION 1
//Sets the number of iterations to perform along the LOS vector. Default 10 covers about 5km
#define MAX_ITERATIONS 100

#define VISIBILITY_SCAN_NUM_THREADS 512

typedef struct {
	float lat[OUTPUT_SIZE];
	float lon[OUTPUT_SIZE];
	float alt[OUTPUT_SIZE];
	float ele[OUTPUT_SIZE];
	float rng[OUTPUT_SIZE];
	int min_index = (NUM_BLOCKS * NUM_THREADS);
	int iteration = 0;
	bool found = false;
}SlantRangeData;

typedef struct
{
	int lat_degrees;      //Earth longitude degree of the current DTED file
	int lon_degrees;      //Earth latitude degree of the current DTED file
	int lon_interval;     //Number of longitude lines
	int lat_interval;     //Number of latitude posts
	int lon_count;        //Interval in arcseconds between longitude posts
	int lat_count;        //Interval in arcseconds between latitude posts
	char lat_hemisphere;  //Latitudal Hemisphere of the current DTED file, N/S
	char lon_hemisphere;  //Longitudinal Hemisphere of the current DTED file, W/E
	short* elevations;    //List of elevation samples from the file (size = lon_count * lat_count)
}DTEDFile;

typedef struct {
	double lat_arcseconds; //latitude of this post in arcseconds
	double lon_arcseconds; //longitude of this post in arcseconds
	char lat_hemisphere;   //DTED file hemispheres
	char lon_hemisphere;
	short elevation;       //Elevation at this post
	int lat_degrees;       //DTED file SW corner latitude
	int lon_degrees;       //DTED file SW corner longitude
}DTEDPost;

extern "C" void calculateABC(Aircraft_Data* Data);
extern "C" void printMatrix(float* matrix, int height, int width);
extern "C" void CalcTargetLocation(Aircraft_Data* AV_Data, DTEDFile* DTED_Data);
extern "C" void LoadElevationData(DTEDFile* DTED_Data);
extern "C" void CalcAreaVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data);

