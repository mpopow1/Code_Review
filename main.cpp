
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "kernel.h"
#include "DTEDFileManager.h"

float DMS_to_DecimalDegrees(int degree, int minute, float second)
{
	return degree + (minute / 60.0) + (second / 3600.0);
}

// Rules:
//North and east are positive. South and west are negative
//Yaw is clockwise relative to true north
//Pitch up is positive, down is negative
//Roll clockwise is positive, counterclockwise is negative
//azimuth is clockwise relative to aircraft forward
//elevation below the aircraft horizontal plane is negative

int  main(int argc, char** argv)
{
	//Arbitrary location within the Grand Canyon was selected
	//Below are modifiable aircraft parameters
	Aircraft_Data Data;
	Data.A_yaw       = DEG_TO_RAD * 0;
	Data.A_pitch     = DEG_TO_RAD * 0;
	Data.A_roll      = DEG_TO_RAD * 0;
	Data.A_azimuth   = DEG_TO_RAD * 279.81;
	Data.A_elevation = DEG_TO_RAD * -19.73;
	Data.A_latitude  = DEG_TO_RAD * DMS_to_DecimalDegrees(36, 8, 44);
	Data.A_longitude = DEG_TO_RAD * DMS_to_DecimalDegrees(112, 15, 46) * -1;
	Data.A_altitude  = FEET_TO_METER * 4000;
	Data.RN = WGS84_A / sqrt(1 - WGS84_E2 * sin(Data.A_latitude) * sin(Data.A_latitude));
	Data.RM = Data.RN * (1 - WGS84_E2) / (1 - WGS84_E2 * sin(Data.A_latitude) * sin(Data.A_latitude));

	calculateABC(&Data);
	printf("Aircraft\nYaw: %f\nPitch: %f\nRoll: %f\n",Data.A_yaw * RAD_TO_DEG, Data.A_pitch * RAD_TO_DEG, Data.A_roll * RAD_TO_DEG);
	printf("Rotation:\nYaw: %f\nPitch: %f\nRoll %f\n",Data.LOS_Rotation[0] * RAD_TO_DEG, Data.LOS_Rotation[1] * RAD_TO_DEG, Data.LOS_Rotation[2] * RAD_TO_DEG);
	
	static DTEDFileManager manager;
	if (manager.GetDTEDFileData("GrandCanyon-36N-113W.DT0"))
	{
		manager.PrintFileData(50);
		LoadElevationData(&(manager.data));

		CalcTargetLocation(&Data, &(manager.data));
		Data.A_altitude = FEET_TO_METER * 6000;
		CalcTargetLocation(&Data, &(manager.data));
		Data.A_altitude  = FEET_TO_METER * 8000;
		CalcTargetLocation(&Data, &(manager.data));

		Data.A_altitude  = FEET_TO_METER * 4000;
		CalcAreaVisibility(&Data, &(manager.data));
	}

	printf("\nPress ENTER to exit...\n");
	getchar();

	return EXIT_SUCCESS;
}
