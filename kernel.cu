#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cublas.h>
#include <curand_kernel.h>
#include "kernel.h"

//Device memory pointers on the device (for convenience)
__device__ DTEDFile *dted;
__device__ short    *elevations; //required because data pointer cannot be passed to dted.elevations from the host
__device__ bool     *visible; //parallel to elevations
__device__ Aircraft_Data *avData; // Yes, I am aware that it is more efficient to use a struct of arrays rather than an array of structs
__device__ SlantRangeData *rangeData;

//Device memory pointers on the host
static DTEDFile* d_DTEDFile;
static short* d_elevations;
static bool* d_visible;
static Aircraft_Data* d_AV_Data;
static SlantRangeData* d_rangeData;

__device__ void GetNearestPost(DTEDPost *post, double target_lat_seconds, double target_lon_seconds)
{
	//peg lat and lon to the nearest post based on file resolution (post interval)
	//This location is relative to the DTED file space, not the Earth as a whole
	int post_lon = 0; //arcseconds of longitude into the DTED square with respect to hemisphere direction
	int post_lat = 0; //arcseconds of latitude into the DTED square with respect to hemisphere direction
	int lat_offset;
	int lon_offset;

	if (dted->lat_hemisphere == 'N')
	{
		//target latitude in arcseconds - southwest corner latitude in arcseconds + half the latitude interval and truncated (cast to int)
		int rounded_lat = (target_lat_seconds - dted->lat_degrees * 60 * 60) + (dted->lat_interval / 2.0);

		//previous result - interval remainder gives us the latitude of the post
		post_lat = rounded_lat - (rounded_lat % dted->lat_interval);

		//Determine the latitude position in the elevetion buffer relative to the longitude record
		lat_offset = (post_lat / dted->lat_interval);
	}
	else
	{
		//Because DTED is always read form the SW corner, need to reverse the latitude direction for southern hemisphere
		int rounded_lat = (target_lat_seconds - (dted->lat_degrees - 1) * 60 * 60) + (dted->lat_interval / 2.0);
		post_lat = rounded_lat - (rounded_lat % dted->lat_interval);

		lat_offset = ((dted->lat_count - 1) - (post_lat / dted->lat_interval));
	}

	if (dted->lon_hemisphere == 'E')
	{
		unsigned int rounded_lon = static_cast<unsigned int>((target_lon_seconds - dted->lon_degrees * 60 * 60) + (dted->lon_interval / 2.0));
		post_lon = rounded_lon - (rounded_lon % dted->lon_interval);
		lon_offset = dted->lon_count * (post_lon / dted->lon_interval);
	}
	else
	{
		//Because DTED is always read form the SW corner, need to reverse the longitude direction for western hemisphere
		unsigned int rounded_lon = static_cast<unsigned int>((target_lon_seconds - (dted->lon_degrees - 1) * 60 * 60) + (dted->lon_interval / 2.0));
		post_lon = rounded_lon - (rounded_lon % dted->lon_interval);
		lon_offset = dted->lon_count * ((dted->lon_count - 1) - (post_lon / dted->lon_interval));
	}

	post->elevation = elevations[lon_offset + lat_offset];
	post->lat_arcseconds = (dted->lat_hemisphere == 'S') ? (dted->lat_degrees - 1) * 60 * 60 + post_lat : dted->lat_degrees * 60 * 60 + post_lat;
	post->lon_arcseconds = (dted->lon_hemisphere == 'W') ? (dted->lon_degrees - 1) * 60 * 60 + post_lon : dted->lon_degrees * 60 * 60 + post_lon;
}

__device__ double GetElevationAt(double target_lat_seconds, double target_lon_seconds)
{
	//==============================================================================
	// STEP 4
	// Determine the starting post which is the DTED post nearest to the target location.
	DTEDPost start_post;
	GetNearestPost(&start_post, target_lat_seconds, target_lon_seconds);

	//==============================================================================
	// STEP 5
	// Dtermine the latitude post. This is the DTED post which is directly north or
	// south of the starting post in the latitude direction of the target location.
	DTEDPost lat_post;
	if (start_post.lat_arcseconds < target_lat_seconds)
	{
		// target is further toward the pole from the starting post
		GetNearestPost(&lat_post, start_post.lat_arcseconds + 3, start_post.lon_arcseconds);
	}
	else if (start_post.lat_arcseconds > target_lat_seconds)
	{
		// target is further toward the equator from the starting post
		GetNearestPost(&lat_post, start_post.lat_arcseconds - 3, start_post.lon_arcseconds);
	}
	else
	{
		// target is exactly on the starting post
		lat_post = start_post;
	}

	//==============================================================================
	// STEP 6
	// Determine the longitude post. This is the DTED post which is directly west or
	// east of the starting post in the longitude direction of the target location.
	DTEDPost lon_post;
	if (start_post.lon_arcseconds < target_lon_seconds)
	{
		// target is further toward the antimeridian from the starting post
		GetNearestPost(&lon_post, start_post.lat_arcseconds, start_post.lon_arcseconds + 3);
	}
	else if (start_post.lon_arcseconds > target_lon_seconds)
	{
		// target is further toward the prime meridian from the starting post
		GetNearestPost(&lon_post, start_post.lat_arcseconds, start_post.lon_arcseconds - 3);
	}
	else
	{
		// target is exactly on the starting post
		lon_post = start_post;
	}

	//==============================================================================
	// STEP 7
	// Determine the diagonal post. This is the DTED post which is directly diagonal
	// from the starting post in the latitude and longitude direction of the target
	// location.
	DTEDPost diagonal_post;
	GetNearestPost(&diagonal_post, lat_post.lat_arcseconds, lon_post.lon_arcseconds);

	//==============================================================================
	// STEP 8
	// Calculate the elevation directly west and east of the target location between
	// the (starting and latitude posts) and (longitude and diagonal posts) respectively.
	// Use simple line slope equation where the terrain elevation is the y and latitude
	// distance from the starting post (or longitude post) to the target location is
	// the x

	//                y       =                              m                                                   x                     +          b
	double mid_lat_elevation1 = ((lat_post.elevation - start_post.elevation) / 3.0) * abs(target_lat_seconds - start_post.lat_arcseconds) + start_post.elevation;
	double mid_lat_elevation2 = ((diagonal_post.elevation - lon_post.elevation) / 3.0) * abs(target_lat_seconds - lon_post.lat_arcseconds) + lon_post.elevation;

	//==============================================================================
	// STEP 9
	// Calculate the elevation of the target location between the two points from
	// step 6. Use simply line slope equation where the terrain elevation is the y
	// and the longitude distance from the starting post to the target location is
	// the x.  The outcome looks like:
	/*
	 *    X               X
	 *    |               |
	 *    |               |
	 *    |               |
	 *    E----O----------E
	 *    |               |
	 *    X               X
	 *
	 * X - DTED post. In this scenario, bottom left would be the starting post,
	 * top left is the latitude post, bottom right is the longitude post and top
	 * right is the diagonal post.
	 * E - point of calculated elevation
	 * O - Target Location
	 */

	double final_elevation = ((mid_lat_elevation2 - mid_lat_elevation1) / 3.0) * abs(target_lon_seconds - start_post.lon_arcseconds) + mid_lat_elevation1;

	//printf("Lat Distance: %f | Lon Distance: %f | Mid elevation 1: %f | Mid elevation 2: %f | Target Elevation: %f\n", abs(start_post.lat_arcseconds - target_lat_seconds), abs(start_post.lon_arcseconds - target_lon_seconds), mid_lat_elevation1, mid_lat_elevation2, final_elevation);

	return final_elevation;
}

/*===================================================================================================================================================
	computeSlantRange(Aircraft_Data *adata, SlantRangeData *rdata = rangeData, int caller_index = 0)

Description
	This kernel computes the target location given an aircraft position and LOS vector from the aircraft. The target location is the point of 
	intersection between the LOS vector and Earth's terrain. It accomplishes this by testing a point along the LOS vector (one point per thread)
	until the first point which is less than the terrain elevation at the same location relative to the LOS vector is found. Points along the LOS
	vector are tested in segments the size of the thread block. If the point of intersect is not found, another iteration is performed further down
	the LOS vector. This is repeated until either the target location is found or the number of iterations is exhausted (such as when looking up into the sky)
Parameters
	adata        - pointer to an Aircraft Data structure containing input data
	rdata        - pointer to a Slant Range Data structure which is used to contain intermediate data in the calculation
	caller_index - the index of the parent thread (with respect to its grid) which launched this kernel. Debug purposes only
Output
	This kernel outputs to the following Aircraft Data structure fields:
	adata->slantRange  - slant range from the aircraft to the Earth (distance along the LOS vector from the aircraft to the target)
	adata->T_altitude  - target location altitude or the terrain elevation (height in Mean Sea Level) at the target location
	adata->T_longitude - target location longitude
	adata->T_latitude  - target location latitude
*/
__global__ void computeSlantRange(Aircraft_Data *adata, SlantRangeData *rdata = rangeData, int caller_index = 0)
{
	const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//This data is uesd by all the threads simultaneously in this kernel. Reset to a known state and synchronize
	if (index == 0)
	{
		rdata->iteration = 0;
		rdata->min_index = (NUM_BLOCKS * NUM_THREADS);
		rdata->found = false;
	}
	
	__syncthreads();
	
	while (!rdata->found && rdata->iteration < MAX_ITERATIONS)
	{
		rdata->rng[index] = RANGE_RESOLUTION * (double)index + (NUM_THREADS * rdata->iteration);

		rdata->lat[index] = adata->A_latitude  + (adata->LOS_Rotation[0] * rdata->rng[index]) / adata->RM;
		rdata->lon[index] = adata->A_longitude + (adata->LOS_Rotation[1] * rdata->rng[index]) / (adata->RN * cos(adata->A_latitude));
		rdata->alt[index] = adata->A_altitude  - (adata->LOS_Rotation[2] * rdata->rng[index]);
		rdata->ele[index] = (float)GetElevationAt( abs(rdata->lat[index]) * RAD_TO_ARC, abs(rdata->lon[index]) * RAD_TO_ARC );
		
		//Print for debugging
		//printf("Caller ID: %d | Iteration: %d | Index: %d | Range: %dm | Lat: %f | Lon: %f | Alt: %f | Elevation: %fm\n", caller_index, rdata->iteration, index, (int)(rdata->rng[index]), rdata->lat[index] * RAD_TO_DEG, rdata->lon[index] * RAD_TO_DEG, rdata->alt[index], rdata->ele[index]);

		__syncthreads(); //synch threads for upcoming conditional statement

		//Need the smallest altitude which is larger than the largest elevation along the vector
		//Can't be the first index because that's the airplane. If it were, then the airplane has collided with the terrain
		//If the current altitude is less than the elevation (underground) and the previous altitude was not...
		if (index > 0 && rdata->alt[index] < rdata->ele[index] && rdata->alt[index - 1] > rdata->ele[index - 1])
		{
			//...then record the index only if it is the smallest index
			//It is implied that the smaller the index, the shorter the range.  The shortest possible range is desired because we don't want the location on the otherside of a mountain.
			atomicMin(&(rdata->min_index), index); //FYI, If using visual studio, intellisense will flag atomic operations as errors.
			rdata->found = true; //Indicate that the intersection has been found - this will terminate the loop
		}
		__syncthreads(); //Syncthreads for upcoming check and potential assignment operation

		if (rdata->found && index == rdata->min_index)
		{
			adata->slantRange  = rdata->rng[rdata->min_index];
			adata->T_altitude  = rdata->alt[rdata->min_index];
			adata->T_longitude = rdata->lon[rdata->min_index];
			adata->T_latitude  = rdata->lat[rdata->min_index];

			//Print for debugging
			//printf("FOUND TARGET: Caller Index: %d | Iteration: %d | Thread Index: %d | Range: %dm | Lat: %f | Lon: %f | Alt: %f | Elevation: %f\n", caller_index, rdata->iteration, index, (int)(adata->slantRange), adata->T_latitude * RAD_TO_DEG, adata->T_longitude * RAD_TO_DEG, rdata->alt[index], adata->T_altitude);
		}
		else
		{
			if (index == 0) rdata->iteration += 1; //increment the number of iterations performed
			__syncthreads(); //sync of the next iteration
		}
	}
}

extern "C" void CalcTargetLocation(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)
{
	Aircraft_Data* d_AV_Data;
	cudaMalloc((void **)&d_AV_Data, sizeof(Aircraft_Data));
	cudaMemcpy(d_AV_Data, AV_Data, sizeof(Aircraft_Data), cudaMemcpyHostToDevice);

	printf("\n\nRunning Slant Range Kernel\n");
	computeSlantRange <<<NUM_BLOCKS, NUM_THREADS >>> (d_AV_Data, d_rangeData ,0);
	cudaDeviceSynchronize();
	printf("\nKernel Finished\n\n");

	cudaMemcpy(AV_Data, d_AV_Data, sizeof(Aircraft_Data), cudaMemcpyDeviceToHost);
	printf("TARGET LOCATION: Latitude: %f | Longitude: %f | Altitude: %fm | Range: %dm\n", AV_Data->T_latitude * RAD_TO_DEG, AV_Data->T_longitude * RAD_TO_DEG, AV_Data->T_altitude, (int)(AV_Data->slantRange));
	cudaFree(d_AV_Data);
	
}

//Returns the bearing from point 1 to point 2 on the Earth assuming both points are known.
//Bearing is relative to true north (so it is aligned with our aircraft yaw)
//All inputs and outputs are in radians
//Makes no distinction between hemispheres (be careful if each point is in a different hemisphere)
__device__ float GetBearingBetweenTwoPoints(float lat1, float lon1, float lat2, float lon2)
{
	lat1 = abs(lat1);
	lon1 = abs(lon1);
	lat2 = abs(lat2);
	lon2 = abs(lon2);
	float delta = abs(lon1 - lon2);
	float x = cos(lat2) * sin(lon2 - lon1);
	float y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1);
	return (2 * PI) - atan2(x, y);
}

//Returns an estimated surface distance in meters from point 1 to point 2 using a smooth sphere Earth model
//A distance function such as this should be accurate enough for very low altitudes (< 16000ft)
__device__ float GetGroundDistance_Haversine(float lat1, float lon1, float lat2, float lon2)
{
	const float EARTH_MEAN_RADIUS = 6371000; //meters
	float delta_lat = sin((lat2 - lat1) / 2);
	float delta_lon = sin((lon2 - lon1) / 2);
	float a = delta_lat * delta_lat + cos(lat1) * cos(lat2) * delta_lon * delta_lon;
	return EARTH_MEAN_RADIUS * 2 * atan2(sqrt(a), sqrt(1 - a));
}

__global__ void computeVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)
{
	const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < DTED_Data->lat_count) //if thread index is out of range, let it finish immediately.
	{
		for (int i = 0; i < DTED_Data->lon_count; i++)
		{
			avData[index].A_altitude = AV_Data->A_altitude;
			avData[index].A_latitude = AV_Data->A_latitude;
			avData[index].A_longitude = AV_Data->A_longitude;
			avData[index].RM = AV_Data->RM;
			avData[index].RN = AV_Data->RN;
			avData[index].H_elevation = index;

			//Assume a level state
			avData[index].A_yaw = 0;
			avData[index].A_pitch = 0;
			avData[index].A_roll = 0;

			//Calculate target location in arcseconds
			float latitude;
			float longitude;

			if (DTED_Data->lat_hemisphere == 'S')
				latitude = (DTED_Data->lat_degrees * 3600 - DTED_Data->lat_interval * (index % DTED_Data->lat_count)) * ARC_TO_RAD * -1;
			else
				latitude = (DTED_Data->lat_degrees * 3600 + DTED_Data->lat_interval * (index % DTED_Data->lat_count)) * ARC_TO_RAD;

			if (DTED_Data->lon_hemisphere == 'W')
				longitude = (DTED_Data->lon_degrees * 3600 - DTED_Data->lon_interval * i) * ARC_TO_RAD * -1;
			else
				longitude = (DTED_Data->lon_degrees * 3600 + DTED_Data->lon_interval * i) * ARC_TO_RAD;

			//Get azimuth angle (bearing) to the target
			avData[index].A_azimuth = GetBearingBetweenTwoPoints(avData[index].A_latitude, avData[index].A_longitude, latitude, longitude);

			//Get elevation/depression angle to the target
			float groundDistance = GetGroundDistance_Haversine(AV_Data->A_latitude, AV_Data->A_longitude, latitude, longitude);
			avData[index].A_elevation = (PI / 2) - atan(groundDistance / (AV_Data->A_altitude - elevations[i * DTED_Data->lat_count + index]));

			//Since we assumed a level state
			avData[index].LOS_Rotation[0] = cos(avData[index].A_azimuth) * cos(avData[index].A_elevation);
			avData[index].LOS_Rotation[1] = sin(avData[index].A_azimuth) * cos(avData[index].A_elevation);
			avData[index].LOS_Rotation[2] = sin(avData[index].A_elevation);

			computeSlantRange << <NUM_BLOCKS, NUM_THREADS >> > (avData + index, rangeData + index, index); //launching kernel from kernel
			cudaDeviceSynchronize(); //Wait for all the child kernels to complete before proceeding.

			//printf("KNOWN TARGET: Iteration: %d | Index: %d | Az: %f | El: %f | Latitude: %f | Longitude: %f | Altitude: %dm\n", i, index, avData[index].A_azimuth * RAD_TO_DEG, avData[index].A_elevation * RAD_TO_DEG, latitude * RAD_TO_DEG, longitude * RAD_TO_DEG, elevations[index]);
			//printf("INTERSECTION: Latitude: %f | Longitude: %f | Altitude: %f | Range: %dm\n", avData[index].T_latitude * RAD_TO_DEG, avData[index].T_longitude * RAD_TO_DEG, avData[index].T_altitude * METER_TO_FEET, (int)(avData[index].slantRange));

			//printf("Known Lat: %f Lon: %f | Calculated: Lat: %f Lon: %f | Diff: Lat: %f Lon: %f\n",latitude * RAD_TO_ARC, longitude * RAD_TO_ARC, avData[index].T_latitude * RAD_TO_ARC, avData[index].T_longitude * RAD_TO_ARC, abs(latitude - avData[index].T_latitude) * RAD_TO_ARC, abs(longitude - avData[index].T_longitude) * RAD_TO_ARC);

			if (abs(latitude - avData[index].T_latitude) * RAD_TO_ARC <= DTED_Data->lat_interval && abs(longitude - avData[index].T_longitude) * RAD_TO_ARC <= DTED_Data->lon_interval)
			{
				visible[index + DTED_Data->lat_count * i] = true;
			}
			else
			{
				visible[index + DTED_Data->lat_count * i] = false;
			}
		}
	}
}

extern "C" void printVisibleArea(DTEDFile *data)
{
	FILE* file;
	file = fopen("visibility_output.txt","w");
	int num_elements = data->lon_count * data->lat_count;
	bool* h_visible = (bool*)malloc(num_elements * sizeof(bool));
	cudaMemcpy(h_visible, d_visible, num_elements * sizeof(bool), cudaMemcpyDeviceToHost);

	for (int r = data->lat_count - 1; r >= 0 ; r--)
	{
		for (int c = 0; c < data->lon_count; c++)
		{
			fprintf(file, "%c ", (h_visible[c * data->lat_count + r] == true) ? 'X' : '-');
		}
		fprintf(file, "\n");
	}

	fclose(file);
	free(h_visible);
}

extern "C" void CalcAreaVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)
{
	int num_blocks = DTED_Data->lat_count / VISIBILITY_SCAN_NUM_THREADS;      //Number of blocks needed for computation
	if (DTED_Data->lat_count % VISIBILITY_SCAN_NUM_THREADS > 0) num_blocks++; //in case the number of threads needed is not evenly divisble by the number of blocks

	//Transfer the Aircraft Data
	Aircraft_Data* d_AV_Data;
	cudaMalloc((void**)& d_AV_Data, sizeof(Aircraft_Data));
	cudaMemcpy(d_AV_Data, AV_Data, sizeof(Aircraft_Data), cudaMemcpyHostToDevice);

	//Transfer the DTED File Data
	DTEDFile* d_DTED_Data;
	cudaMalloc((void**)& d_DTED_Data, sizeof(DTEDFile));
	cudaMemcpy(d_DTED_Data, DTED_Data, sizeof(DTEDFile), cudaMemcpyHostToDevice);

	//printf("\n\nLocation Lat: %f | Lon: %f | Alt: %f\n",AV_Data->A_latitude * RAD_TO_DEG, AV_Data->A_longitude * RAD_TO_DEG, AV_Data->A_altitude);
	printf("\n\nRunning Visibility Scan Kernel\n");
	computeVisibility <<< num_blocks, VISIBILITY_SCAN_NUM_THREADS >>> (d_AV_Data, d_DTED_Data);
	printf("\nKernel Finished. Check file visibility_output.txt\n");

	printVisibleArea(DTED_Data);

	cudaFree(d_AV_Data);
	cudaFree(d_DTED_Data);
}

//Allocates device memory and copies DTED File elevation data to the device
//Copy memory address of allocated memory to global pointers only has to be done once.
extern "C" void LoadElevationData(DTEDFile* DTED_Data)
{
	int total_elements = DTED_Data->lon_count * DTED_Data->lat_count;

	int size = sizeof(DTEDFile);
	cudaMalloc((void**)& d_DTEDFile, size); //Allocate device memory
	cudaMemcpy(d_DTEDFile, DTED_Data, size, cudaMemcpyHostToDevice); //Copy data over to device
	cudaMemcpyToSymbol(dted, &d_DTEDFile, sizeof(DTEDFile*)); //Assign pointer to global __device__ symbol

	//Couldn't figure out how to copy directly to dted->elevations without doing it in a kernel so made a separate global memory pointer instead
	size = total_elements * sizeof(short);
	cudaMalloc((void**)& d_elevations, size); //Allocate device memory
	cudaMemcpy(d_elevations, DTED_Data->elevations, size, cudaMemcpyHostToDevice); //Copy data over to device
	cudaMemcpyToSymbol(elevations, &d_elevations, sizeof(short*)); //Assign pointer to global __device__ symbol

	size = total_elements * sizeof(bool);
	cudaMalloc((void**)& d_visible, size);
	cudaMemset(d_visible, 0, size);
	cudaMemcpyToSymbol(visible, &d_visible, sizeof(bool*));

	size = DTED_Data->lat_count * sizeof(Aircraft_Data); //Not the most memory efficient
	cudaMalloc((void**)& d_AV_Data, size);
	cudaMemset(d_AV_Data, 0, size);
	cudaMemcpyToSymbol(avData, &d_AV_Data, sizeof(Aircraft_Data*));
	
	size = DTED_Data->lat_count * sizeof(SlantRangeData);
	cudaMalloc((void**)& d_rangeData, size);
	cudaMemcpyToSymbol(rangeData, &d_rangeData, sizeof(SlantRangeData*)); //Copy this struct pointer to one in global memory so we don't have to pass it in kernel calls*/
}

extern "C"  void printMatrix(float* matrix, int height, int width)
{
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			printf("| %f |",matrix[c * width + r] * RAD_TO_DEG);
		}
		printf("\n");
	}
}

//Calculate the rotation matrix which is a 3x1 matrix containing values yaw (A), pitch (B) and roll (C)
//In memory this is stored simply as a 3 element array
extern "C"  void calculateABC(Aircraft_Data *Data)
{
	/*

	Yaw Rotation Matrix					|	Pitch Rotation Matrix			|	Roll Rotation Matrix			|	LOS Vector
		cos(yaw)   -sin(yaw)		0	|	 cos(pitch)	  0		sin(pitch)	|		1		0			0		|	cos(azimuth) * cos(elevation)
		sin(yaw)	cos(yaw)		0	|		0		  1		   0		|		0	 cos(roll)	-sin(roll)	|	sin(azimuth) * cos(elevation)
		0				0			1	|	-sin(pitch)	  0		cos(pitch)	|		0	 sin(roll)	 cos(roll)	|	sin(elevation)

	*/
	// IMPORTANT: In CUDA, matricies are stored in column order format
	float h_Matrix_Yaw[ROTATION_MATRIX_NUM_ELEMENTS]   = { cos(Data->A_yaw), sin(Data->A_yaw), 0, -sin(Data->A_yaw), cos(Data->A_yaw), 0, 0, 0, 1 };
	float h_Matrix_Pitch[ROTATION_MATRIX_NUM_ELEMENTS] = { cos(Data->A_pitch), 0, -sin(Data->A_pitch), 0, 1, 0, sin(Data->A_pitch), 0, cos(Data->A_pitch) };
	float h_Matrix_Roll[ROTATION_MATRIX_NUM_ELEMENTS]  = { 1, 0, 0, 0, cos(Data->A_roll), sin(Data->A_roll), 0, -sin(Data->A_roll), cos(Data->A_roll) };
	float h_Matrix_LOS_Vector[LOS_VECTOR_MATRIX_NUM_ELEMENTS] = { cos(Data->A_azimuth) * cos(-1 * Data->A_elevation), sin(Data->A_azimuth) * cos(-1 * Data->A_elevation), sin(-1 * Data->A_elevation) };
	float* d_Matrix_Yaw;
	float* d_Matrix_Pitch;
	float* d_Matrix_YawPitchProduct;
	float* d_Matrix_Roll;
	float* d_Matrix_YawPitchRollProduct;
	float* d_Matrix_LOS_Vector;
	float* d_new_LOS_Vector;

	cublasInit();

	/*ALLOCATE ON THE DEVICE*/
	cublasAlloc(ROTATION_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_Yaw);
	cublasAlloc(ROTATION_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_Pitch);
	cublasAlloc(ROTATION_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_Roll);
	cublasAlloc(LOS_VECTOR_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_LOS_Vector);
	cublasAlloc(ROTATION_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_YawPitchProduct);
	cublasAlloc(ROTATION_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_Matrix_YawPitchRollProduct);
	cublasAlloc(LOS_VECTOR_MATRIX_NUM_ELEMENTS, sizeof(float), (void**)& d_new_LOS_Vector);

	/*SET MATRIX*/
	cublasSetMatrix(3, 3, sizeof(float), h_Matrix_Yaw, 3, d_Matrix_Yaw, 3);
	cublasSetMatrix(3, 3, sizeof(float), h_Matrix_Pitch, 3, d_Matrix_Pitch, 3);
	cublasSetMatrix(3, 3, sizeof(float), h_Matrix_Roll, 3, d_Matrix_Roll, 3);
	cublasSetMatrix(3, 1, sizeof(float), h_Matrix_LOS_Vector, 3, d_Matrix_LOS_Vector, 3);

	/*KERNEL*/
	//Perform the operation:
	// ((Yaw Matrix * Pitch Matrix) * Roll Matrix) * LOS Vector
	// The result changes the frame of reference of the LOS vector from the aircraft body to the horizontal plane (relative to Earth) 
	// at the aircraft body. This makes further calculations much easier.
	cublasSgemm('n', 'n', 3, 3, 3, 1, d_Matrix_Yaw, 3, d_Matrix_Pitch, 3, 0, d_Matrix_YawPitchProduct, 3);
	cublasSgemm('n', 'n', 3, 3, 3, 1, d_Matrix_YawPitchProduct, 3, d_Matrix_Roll, 3, 0, d_Matrix_YawPitchRollProduct, 3);
	cublasSgemm('n', 'n', 3, 1, 3, 1, d_Matrix_YawPitchRollProduct, 3, d_Matrix_LOS_Vector, 3, 0, d_new_LOS_Vector, 3);
	cublasGetMatrix(3, 1, sizeof(float), d_new_LOS_Vector, 3, h_Matrix_LOS_Vector, 3);

	/* Shutdown */
	cublasShutdown();

	//Copy the end result to host memory
	memcpy(Data->LOS_Rotation, h_Matrix_LOS_Vector, 3*sizeof(float));

	//In a practical real-time application, such as if this code were running on an aircraft computer,
	//these arrays would be allocated and re-used, and probably never freed. 
	//However, this demonstration is a single execution and not a real-time application.
	cublasFree(d_Matrix_Yaw);
	cublasFree(d_Matrix_Pitch);
	cublasFree(d_Matrix_YawPitchProduct);
	cublasFree(d_Matrix_Roll);
	cublasFree(d_Matrix_YawPitchRollProduct);
	cublasFree(d_Matrix_LOS_Vector);
	cublasFree(d_new_LOS_Vector);
}

