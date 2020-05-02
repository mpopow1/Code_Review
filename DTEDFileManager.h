#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "kernel.h"

using namespace std;

class DTEDFileManager
{
public:

	DTEDFileManager();
	~DTEDFileManager();
	bool GetDTEDFileData(string);
	void PrintFileData(int);

	DTEDFile data;

private:
	ifstream file;
};

