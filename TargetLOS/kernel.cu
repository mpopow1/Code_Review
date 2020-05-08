#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cublas.h>
#include <curand_kernel.h>
#include "kernel.h"

//Device memory pointers on the device (for convenience)
__device__ DTEDFile *dted;            // Points to the parameters from the DTED file
__device__ short    *elevations;      // Contains all the elevation samples from the DTED file, and directly maps to the area covered by the file.
__device__ bool     *visible;         // Parallel to elevations, true indicates that that location is visible from the aircraft's current location
__device__ Aircraft_Data *avData;     // Contains the aircraft data (location, orientation and target location result).  Yes, I am aware that it is more efficient to use a struct of arrays rather than an array of structs
__device__ SlantRangeData *rangeData; // Contains components for the slant range calculation

//Device memory pointers on the host
static DTEDFile* d_DTEDFile;
static short* d_elevations;
static bool* d_visible;
static Aircraft_Data* d_AV_Data;
static SlantRangeData* d_rangeData;

//==================================================================================================================================================
//------------------------------------------ Computing Slant Range kernels and functions ------------------------------------------------------------

/*===================================================================================================================================================
	GetNearestPost(DTEDPost *post, double target_lat_seconds, double target_lon_seconds)

Description
	This device function determines the nearest location relative to the DTED elevation posts. It is a component to the algorithm GetElevationAt()
	for extrapolating the terrain elevation at the target location.
Parameters
	post               - pointer to the output DTED post (the elevation sample and corresponding location from the DTED file)
	target_lat_seconds - the target location latitude in arc seconds
	target_lon_seconds - the target location longitude in arc seconds
*/
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

/*===================================================================================================================================================
	GetElevationAt(double target_lat_seconds, double target_lon_seconds)

Description
	This device function calculates the terrain elevation at the target location using the elevation samples from a DTED file. Since DTED files have
	a certain resolution, that is the terrain is unknown between elevation sample posts, the terrain elevation must be extrapolated from nearby posts
	if the target location is somewhere between the posts. Therefore, this algorithm notes the locations and elevations of the four posts surrounding
	the actual target location.  Next a linear function of the elevation is computed between opposing pairs from these four posts such that they are 
	parallel either in the latitude or longitude direction.  Finally a linear function of the elevation is computed which connects these two lines 
	together while intersecting the actual target location.  The elevation on the linear function at the target location is the extrapolated terrain
	elevation at the target location.
Parameters
	target_lat_seconds   - expected target location latitude in arcseconds
	target_lon_seconds   - expected target location longitude in arcseconds
Output
	An extrapolated teraain elevation for the anticipated target location.
*/
__device__ double GetElevationAt(double target_lat_seconds, double target_lon_seconds)
{
	//==============================================================================
	// Determine the starting post which is the DTED post nearest to the target location.
	DTEDPost start_post;
	GetNearestPost(&start_post, target_lat_seconds, target_lon_seconds);

	//==============================================================================
	// Determine the latitude post. This is the DTED post which is directly north or
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
	// Determine the diagonal post. This is the DTED post which is directly diagonal
	// from the starting post in the latitude and longitude direction of the target
	// location.
	DTEDPost diagonal_post;
	GetNearestPost(&diagonal_post, lat_post.lat_arcseconds, lon_post.lon_arcseconds);

	//==============================================================================
	// Calculate the elevation directly west and east of the target location between
	// the (starting and latitude posts) and (longitude and diagonal posts) respectively.
	// Use simple line slope equation where the terrain elevation is the y and latitude
	// distance from the starting post (or longitude post) to the target location is
	// the x

	//                y       =                              m                                                   x                     +          b
	double mid_lat_elevation1 = ((lat_post.elevation - start_post.elevation) / 3.0) * abs(target_lat_seconds - start_post.lat_arcseconds) + start_post.elevation;
	double mid_lat_elevation2 = ((diagonal_post.elevation - lon_post.elevation) / 3.0) * abs(target_lat_seconds - lon_post.lat_arcseconds) + lon_post.elevation;

	//==============================================================================
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

	//Debug
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
	the LOS vector. That is the segment is shifted segment size down the LOS vector. This is repeated until either the target location is found or 
	the number of iterations is exhausted (such as when looking up into the sky).
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

	//This data is used by all the threads simultaneously in this kernel. Reset to a known state and synchronize
	if (index == 0)
	{
		rdata->iteration = 0;
		rdata->min_index = (NUM_BLOCKS * NUM_THREADS);
		rdata->found = false;
	}
	
	__syncthreads();
	
	//Each thread operates on a specific location along the LOS vector within a "batch" of samples or sampling threads
	//If no thread finds the LOS intersect with the terrain, the batch is moved further along the LOS vector on the next iteration
	//This is repeated for a limited number of iterations until a thread finds the intersection
	while (!rdata->found && rdata->iteration < MAX_ITERATIONS)
	{
		//Calculate the distance from the aircraft along the LOS vector - dependent on the current thread and the resolution of the 
		//"step" along the LOS vector.  By default each "step" size is 1 meter. So, each thread is testing a location that is
		//thread index meters from the aircraft with respect to the current iteration.
		rdata->rng[index] = RANGE_RESOLUTION * (double)index + (NUM_THREADS * rdata->iteration);

		//Calculate the latitude and longitude of the point on the ground beneath the test point on the LOS vector
		rdata->lat[index] = adata->A_latitude  + (adata->LOS_Rotation[0] * rdata->rng[index]) / adata->RM;
		rdata->lon[index] = adata->A_longitude + (adata->LOS_Rotation[1] * rdata->rng[index]) / (adata->RN * cos(adata->A_latitude));

		//Calculate the altitude of the test point on the LOS vector
		rdata->alt[index] = adata->A_altitude  - (adata->LOS_Rotation[2] * rdata->rng[index]);

		//Calculate the terrain elevation of the point on the ground beneath the test point on the LOS vector
		rdata->ele[index] = (float)GetElevationAt( abs(rdata->lat[index]) * RAD_TO_ARC, abs(rdata->lon[index]) * RAD_TO_ARC );
		
		//Print for debugging
		//printf("Caller ID: %d | Iteration: %d | Index: %d | Range: %dm | Lat: %f | Lon: %f | Alt: %f | Elevation: %fm\n", caller_index, rdata->iteration, index, (int)(rdata->rng[index]), rdata->lat[index] * RAD_TO_DEG, rdata->lon[index] * RAD_TO_DEG, rdata->alt[index], rdata->ele[index]);

		__syncthreads(); //synch threads for upcoming conditional statement

		//Intersection is determined by the smallest altitude which is larger than the largest elevation along the LOS vector
		//Can't be the first index because that's the aircraft. If it were, then the aircraft has collided with the terrain

		//If the current altitude is less than the elevation (underground) and the previous altitude was not...
		if (index > 0 && rdata->alt[index] < rdata->ele[index] && rdata->alt[index - 1] > rdata->ele[index - 1])
		{
			//...then record the index only if it is the smallest index. It is implied that the smaller the index, the shorter the range.
			//The shortest possible range is desired because we don't want the location on the otherside of a mountain (imagine drawing the LOS vector through a mountain such that it emerges on the other side).

			atomicMin(&(rdata->min_index), index); //FYI, If using visual studio, intellisense will flag atomic operations as errors.
			rdata->found = true; //Indicate that the intersection has been found - this will terminate the loop
		}
		__syncthreads(); //Syncthreads for upcoming check and potential assignment operation

		if (rdata->found && index == rdata->min_index) //Make sure this is the thread that found the point of intersection.
		{
			//Assign its sample location data to the target location output
			adata->slantRange  = rdata->rng[rdata->min_index];
			adata->T_altitude  = rdata->alt[rdata->min_index];
			adata->T_longitude = rdata->lon[rdata->min_index];
			adata->T_latitude  = rdata->lat[rdata->min_index];

			//Print for debugging
			//printf("FOUND TARGET: Caller Index: %d | Iteration: %d | Thread Index: %d | Range: %dm | Lat: %f | Lon: %f | Alt: %f | Elevation: %f\n", caller_index, rdata->iteration, index, (int)(adata->slantRange), adata->T_latitude * RAD_TO_DEG, adata->T_longitude * RAD_TO_DEG, rdata->alt[index], adata->T_altitude);
		}
		else //If the point of intersection hasn't been found yet...
		{
			if (index == 0) rdata->iteration += 1; //...increment the number of iterations performed
			__syncthreads();                       //sync of the next iteration
		}
	}
}

/*===================================================================================================================================================
	CalcTargetLocation(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)

Description
	This host function is a wrapper for launching the computeSlantRange() kernel.  It loads the known aircraft data to the device, calls the kernel
	and retrieves the result from the device. By default, it launches computeSlantRange() with 1 block of 512 threads. 
Parameters
	adata        - pointer to an Aircraft Data structure containing input data
	rdata        - pointer to a Slant Range Data structure which is used to contain intermediate data in the calculation
	caller_index - the index of the parent thread (with respect to its grid) which launched this kernel. Debug purposes only
Output
	This function prints out the target location to the screen including latitude, longitude, altitude (terrain elevation) and range from the aircraft.
*/
extern "C" void CalcTargetLocation(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)
{
	Aircraft_Data* d_AV_Data;
	cudaMalloc((void **)&d_AV_Data, sizeof(Aircraft_Data));
	cudaMemcpy(d_AV_Data, AV_Data, sizeof(Aircraft_Data), cudaMemcpyHostToDevice);

	printf("\n\nRunning Slant Range Kernel\n");
	computeSlantRange <<< NUM_BLOCKS, NUM_THREADS >>> (d_AV_Data, d_rangeData ,0);
	cudaDeviceSynchronize();
	printf("\nKernel Finished\n\n");

	cudaMemcpy(AV_Data, d_AV_Data, sizeof(Aircraft_Data), cudaMemcpyDeviceToHost);
	printf("TARGET LOCATION: Latitude: %f | Longitude: %f | Altitude: %fm | Range: %dm\n", AV_Data->T_latitude * RAD_TO_DEG, AV_Data->T_longitude * RAD_TO_DEG, AV_Data->T_altitude, (int)(AV_Data->slantRange));
	cudaFree(d_AV_Data);
	
}

//===================================================================================================================================================
//----------------------------------- Perform Location Visibility Scan functions and kernels --------------------------------------------------------

/*===================================================================================================================================================
	GetBearingBetweenTwoPoints(float lat1, float lon1, float lat2, float lon2)

Description
	This device function calculates the bearing between two points assuming both points are known. The bearing is relative to true north 
	(so it is aligned with aircraft	yaw). Makes no distinction between hemispheres. (Be careful if each point is in a different hemisphere)
Parameters
	lat1 / lon1 - latitude and longitude of the first point in radians
	lat2 / lon2 - latitude and longitude of the second point in radians
Output
	Returns the bearing from point 1 to point 2
*/
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

/*===================================================================================================================================================
	GetGroundDistance_Haversine(float lat1, float lon1, float lat2, float lon2)

Description
	This device function calculates an estimated surface distance in meters from point 1 to point 2 using a smooth sphere Earth model.  
	The accuracy of this function should be sufficient for very low altitudes (< 16000ft) and points that are too far apart. Otherwise, 
	this would not be recommended in practice.
Parameters
	lat1 / lon1 - latitude and longitude of the first point in radians
	lat2 / lon2 - latitude and longitude of the second point in radians
Output
	Returns the distance from point 1 to point 2 along a smooth sphere
*/
__device__ float GetGroundDistance_Haversine(float lat1, float lon1, float lat2, float lon2)
{
	const float EARTH_MEAN_RADIUS = 6371000; //meters
	float delta_lat = sin((lat2 - lat1) / 2);
	float delta_lon = sin((lon2 - lon1) / 2);
	float a = delta_lat * delta_lat + cos(lat1) * cos(lat2) * delta_lon * delta_lon;
	return EARTH_MEAN_RADIUS * 2 * atan2(sqrt(a), sqrt(1 - a));
}

/*===================================================================================================================================================
	computeVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)

	TODO:
	Currently, this function works for some scenarios and doesn't for others.  Not sure why, I might be inadvertently placing the aircraft 
	"underground" somehow.  Need to investigate.

Description
	This kernel is used to perform a scan of the entire area covered by the DTED file and marks which locations in the "visible" buffer can be visible 
	from the aircraft's current location. Each element in the "visible" buffer corresponds to a DTED post (so each element is a known location)
	within the 1 degree x 1 degree square just like the "elevations" buffer.  Both of these buffers are single dimension and represent each location
	from the DTED file in order of longitude record of latitude posts.  In this kernel, each thread processes a single DTED post location within the 
	longitude record. This is repeated for each longitude record across the entire file area. A location is considered visible if the resulting target
	after calling computeSlantRange() is within a certain threshold of accuracy.

	Visibility is determined by launching the computeSlantRange() kernel for each location tested. This function demonstrates launching a kernel from
	a kernel.  This also results in a tremendous number of threads and can cause the display driver to restart especially for higher resolution DTED.
Parameters
	AV_Data   - pointer to an Aircraft Data structure containing input data (i.e. the aircraft's current location)
	DTED_Data - pointer to a DTED File Data structure containing data describing the area covered by the DTED file
Output
	Each location is marked in the "visible" buffer whether it is visible (true) or not (false).
*/
__global__ void computeVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)
{
	const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < DTED_Data->lat_count) //if thread index is out of range, let it finish immediately.
	{
		for (int i = 0; i < DTED_Data->lon_count; i++)
		{
			//An Aircraft_Data structure is maintained for each latitude post in the current longitude record under test
			//The current AV location is copied to it here where it will be used in the computeSlantRange() kernel
			avData[index].A_altitude = AV_Data->A_altitude;
			avData[index].A_latitude = AV_Data->A_latitude;
			avData[index].A_longitude = AV_Data->A_longitude;
			avData[index].RM = AV_Data->RM;
			avData[index].RN = AV_Data->RN;

			//Assume a level state - we want to know if the location can be visible, not is it visible
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

			//Since we assumed a level state - do a simplified version of the matrix multiplication from calculateABC()
			avData[index].LOS_Rotation[0] = cos(avData[index].A_azimuth) * cos(avData[index].A_elevation);
			avData[index].LOS_Rotation[1] = sin(avData[index].A_azimuth) * cos(avData[index].A_elevation);
			avData[index].LOS_Rotation[2] = sin(avData[index].A_elevation);

			//Launch kernel from this kernel to test whether the target location can be seen from the current aircraft location
			computeSlantRange << <NUM_BLOCKS, NUM_THREADS >> > (avData + index, rangeData + index, index);
			cudaDeviceSynchronize(); //Wait for all the child kernels to complete before proceeding.

			//Debug
			//printf("KNOWN TARGET: Iteration: %d | Index: %d | Az: %f | El: %f | Latitude: %f | Longitude: %f | Altitude: %dm\n", i, index, avData[index].A_azimuth * RAD_TO_DEG, avData[index].A_elevation * RAD_TO_DEG, latitude * RAD_TO_DEG, longitude * RAD_TO_DEG, elevations[index]);
			//printf("INTERSECTION: Latitude: %f | Longitude: %f | Altitude: %f | Range: %dm\n", avData[index].T_latitude * RAD_TO_DEG, avData[index].T_longitude * RAD_TO_DEG, avData[index].T_altitude * METER_TO_FEET, (int)(avData[index].slantRange));

			//If the level of accuracy is within the interval between DTED posts, then assume the location is visible
			//For level 0 DTED, the inerval is 30 arcseconds.
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

/*===================================================================================================================================================
	printVisibleArea(DTEDFile *data)

Description
	Copies the "visible" buffer from the device to the host and writes it to a file.

Parameters
	data - pointer to a DTED File Data structure containing data describing the area covered by the DTED file
Output
	The file visibility_output.txt representing the locations from the DTED file. A '-' means that location is not visible and a 'X' means that location
	is visible from the aircraft's current position.
*/
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

/*===================================================================================================================================================
	CalcAreaVisibility(Aircraft_Data* AV_Data, DTEDFile* DTED_Data)

Description
	Sets up and launches the computeVisibility() kernel. By default, this kernel launches with 512 threads per block.  The number of blocks is 
	determined by the number of latitude posts in the longitude record of the DTED.  Each thread processes each latitude post in the longitude 
	record. The longitude record is advanced on each iteration within the kernel. Each thread has its own Aircraft_Data and DTED_Data structures to
	facilitate computation.  Each thread also launches the computeSlantRange() kernel to determine if its corresponding location is visible from
	the aircraft's position.

	DTED level 0 contains 121 latitude posts per longitude record and each thread launches another kernel of default 512 threads bringing the default
	total thread count to 77312. For higher resolution such as level 1, this would be 1201 * 512 = 614912 threads.
Parameters
	AV_Data   - pointer to an Aircraft Data structure containing input data (i.e. the aircraft's current location)
	DTED_Data - pointer to a DTED File Data structure containing data describing the area covered by the DTED file
Output
	Calls printVisibility() which outputs the "visible" buffer to a file.
*/
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

/*===================================================================================================================================================
	LoadElevationData(DTEDFile* DTED_Data)

Description
	Utility function which allocates device memory for Aircraft data, DTED data, elevation and visible buffers.	Allocates device memory for the 
	computeSlantRange() and computeVisibility() functions. Copies DTED elevations to device. Copies addresses of device memory buffer to pre-defined
	device pointers so that the buffers are easier to access later.
Parameters
	DTED_Data - pointer to a DTED File Data structure containing data describing the area covered by the DTED file
*/
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

//===================================================================================================================================================
//-------------------------------------------------- Aircraft LOS Vector Rotation Functions ---------------------------------------------------------

/*===================================================================================================================================================
	printMatrix(float* matrix, int height, int width)

Description
	Utility for printing out a column ordered matrix. Be aware that the aircraft data is in radians and this function converts to degrees after
	That is, cos(0) which is 1 in both radians and degrees will display as 180/PI
Parameters
	matrix - pointer to the matrix to print out
	height - height of the matrix
	width  - width of the matrix
*/
extern "C" void printMatrix(float* matrix, int height, int width)
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

/*===================================================================================================================================================
	calculateABC(Aircraft_Data *Data)

Description
	This function calculates the rotation matrix for the LOS vector based the aircraft attitude.  This function uses cuBLAS to perform the matrix
	multiplication.
Parameters
	Data - pointer to the aircraft data containing both the orientation data input (yaw, pitch, roll, azimuth and elevation)
Output
	The calculated rotation matrix for the LOS vector is saved to the LOS_Rotation buffer in the aircraft data.
	It is a 3x1 matrix containing values yaw (A), pitch (B) and roll (C) stored as a 3 element array.
*/
extern "C" void calculateABC(Aircraft_Data *Data)
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

