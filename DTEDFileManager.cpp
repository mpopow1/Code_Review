#include "DTEDFileManager.h"

DTEDFileManager::DTEDFileManager(){}
DTEDFileManager::~DTEDFileManager(){}

/*===================================================================================================================================================
	GetDTEDFileData(string filename)

Description
	Reads in the header data and elevations data from the Digital Terrain Elevation Data (DTED) file.
Parameters
	filename - name of the DTED file
*/
bool DTEDFileManager::GetDTEDFileData(string filename)
{
	//in bytes
	const int ELEVATION_DATA_START = 3436;
	const int RECORD_HEADER_SIZE = 12;

	cout << "Opening File: " << filename << endl;
	file.open(filename, ifstream::binary);

	if (file.is_open())
	{
		//Get the longitude coordinate of the southwest corner
		file.seekg(4);
		file >> data.lon_degrees;
		data.lon_degrees /= 10000; //Remove minutes and seconds after reading in. This method is easier than messing with substrings.
		//Get the longitude hemisphere of the southwest corner
		file.seekg(11);
		file >> data.lon_hemisphere;
		//Get the latitude coordinate of the southwest corner
		file >> data.lat_degrees;
		data.lat_degrees /= 10000; //Remove minutes and seconds after reading in. This method is easier than messing with substrings.
		//Get the latitude hemisphere of the southwest corner
		file.seekg(19);
		file >> data.lat_hemisphere;
		//Get the interval between each longitude record in arcseconds
		file.seekg(20);
		char bytes[4];
		file.read(bytes,4);
		data.lon_interval = atoi(bytes) / 10; //Resolution of tenths of arcseconds in the file, convert to whole arcseconds
		//Get the interval between each latitude post in arcseconds
		file.read(bytes, 4);
		data.lat_interval = atoi(bytes) / 10; //Resolution of tenths of arcseconds in the file, convert to whole arcseconds
		//Get the number of longitude records and latitude posts per record in the file
		file.seekg(47);
		file.read(bytes, 4);
		data.lon_count = atoi(bytes);
		file.read(bytes, 4);
		data.lat_count = atoi(bytes);
		
		//Get the terrain elevation samples which will be stored as a single dimension array
		data.elevations = (short*)malloc(sizeof(short) * data.lon_count * data.lat_count);
		if (data.elevations)
		{
			char sample[2];
			for (int lon_record = 0; lon_record < data.lon_count; lon_record++)
			{
				file.seekg(ELEVATION_DATA_START + (data.lat_count * 2 + RECORD_HEADER_SIZE) * lon_record); //Move to the start of the elevation data for the current record

				for (int lat_post = 0; lat_post < data.lat_count; lat_post++)
				{
					file.read(&(sample[1]), 1); //In the DTED file, the elevation is stored in Big Endian format
					file.read(&(sample[0]), 1); //swap the bytes for each elevation sample.
					data.elevations[lon_record * data.lon_count + lat_post] = *((short*)sample);
				}
			}
		}
		else
		{
			printf("Could not allocate memory for elevation data. Terminating Program...\n");
		}

		file.close();
		return true;
	}
	else
	{
		printf("Could not open DTED file: %s\n", filename.c_str());
	}

	return false;
}

/*===================================================================================================================================================
	PrintFileData(int num_elevations)

Description
	Utility function to print out the data that was read in from the DTED file.
Parameters
	num_elevations - number of elevation samples to print out.
*/
void DTEDFileManager::PrintFileData(int num_elevations)
{
	int nw_corner_lat, nw_corner_lon;
	int ne_corner_lat, ne_corner_lon;
	int se_corner_lat, se_corner_lon;

	nw_corner_lon = data.lon_degrees;
	if (data.lon_hemisphere == 'W')
	{
		ne_corner_lon = data.lon_degrees - 1;
		se_corner_lon = data.lon_degrees - 1;
	}
	else
	{
		ne_corner_lon = data.lon_degrees + 1;
		se_corner_lon = data.lon_degrees + 1;
	}

	se_corner_lat = data.lat_degrees;
	if (data.lat_hemisphere == 'N')
	{
		nw_corner_lat = data.lat_degrees + 1;
		ne_corner_lat = data.lat_degrees + 1;
	}
	else
	{
		nw_corner_lat = data.lat_degrees - 1;
		ne_corner_lat = data.lat_degrees - 1;
	}

	cout << "=== DTED File Info ===\n\nArea bound by the file:" << endl;
	cout << nw_corner_lat << data.lat_hemisphere << " " << nw_corner_lon << data.lon_hemisphere << "-----" << ne_corner_lat << data.lat_hemisphere << " " << ne_corner_lon << data.lon_hemisphere << endl;
	cout << "   |           |\n   |           |\n   |           |\n   |           |\n";
	cout << data.lat_degrees << data.lat_hemisphere << " " << data.lon_degrees << data.lon_hemisphere << "-----" << se_corner_lat << data.lat_hemisphere << " " << se_corner_lon << data.lon_hemisphere << endl << endl;
	cout << "Longitude Record Interval: " << data.lon_interval << " arcseconds\nLatitude Post Interval: " << data.lat_interval << " arcseconds" << endl;
	cout << "Number of Longitude Records: " << data.lon_count << "\nNumber of Latitude Posts per Longitude Record: " << data.lat_count << endl;
	cout << "Total Elevation Samples: " << data.lon_count * data.lat_count << endl;

	int size = data.lat_count * data.lon_count;
	if (num_elevations < size && num_elevations > 0) size = num_elevations;
	cout << "\nPrinting the first " << size << " Elevation Samples in meters:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << data.elevations[i] << " ";
	}
	cout << endl;
}

