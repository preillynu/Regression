
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <Windows.h>
using namespace std;


//Set grid and block size for the kernels that run and sets the number of neighbors desired
float minAlpha = 0.01;
int maxIter = 150;
float setWeight = 1.0;

float sigmoid(float in){
	return 1.0 / (1 + exp(-1 * in));
}

int main()
{
	//Input file to read the labeled data
	ifstream inputData;

	//The arrays that will hold the labeled data
	float** coors;

	//Arrays to hold the distances computed via the GPU and CPU
	float* weights;

	//An array to hold the unsorted labels
	int* labels;
	int* sortedlabels;

	//Variables used to hold one data point
	float* dataPoint;

	//Variables to hold the number of labeled data points and the number or points being entered
	int numInput;
	int numPoints;
	int numDim;
	int numLabel;
	float alpha = minAlpha;

	//Opens a file whose first line is the number of elements and every subsequent line is an x coordinate, y coordinate,
	//and label that are seperated by spaces
	inputData.open("Synthesized_Data_100.txt");

	//Make sure the file opens correctly
	if (!inputData.is_open()){
		cout << "Something went wrong while reading in the data. Check where it is located again." << endl;
		exit(0);
	}

	//Store the number of labeled data points, the number of dimensions and the total number of labels
	inputData >> numInput >> numDim >> numLabel;

	//Prompt the user for the number of points being classified
	cout << "How many points do you want to read? ";
	cin >> numPoints;

	//Set up the arrays to have a max capacity equal to the sum of the number of labeled and unlabeled points
	coors = new float*[numDim+1];
	labels = new int[numInput + numPoints];
	sortedlabels = new int[numDim];
	weights = new float[numDim + 1];
	dataPoint = new float[numDim+1];

	//Set up the 2D coors array for the number of dimensions each point is
	for (int i = 0; i < numDim+1; i++){
		coors[i] = new float[numInput + numPoints];
		weights[i] = setWeight;
	}

	for (int i = 0; i < numInput; i++){
		coors[0][i] = 1.0;
		for (int j = 1; j < numDim+1; j++){
			inputData >> coors[j][i];
		}
		inputData >> labels[i];
	}

	//Close the input file
	inputData.close();

	//Collect the data points that the user wants classified
	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: " << endl;
		coors[0][i + numInput] = 0.0;
		for (int j = 1; j < numDim+1; j++){
			cout << j << " dim: ";
			cin >> coors[j][i + numInput];
		}
		cout << endl;
	}

	//Get the coordinates of the point to be classified
	for (int i = 0; i < numDim + 1; i++){
		dataPoint[i] = coors[i][numInput];
	}

	//Number of ticks per second
	LARGE_INTEGER frequency;
	//Measure times
	LARGE_INTEGER t1, t2;
	//Store time
	double elapsedTime;

	//Fill the frequency variable
	QueryPerformanceFrequency(&frequency);

	//Get the first time
	QueryPerformanceCounter(&t1);

	//Run the code for the first point
	for (int i = 0; i < maxIter; i++){
		for (int j = 0; j < numInput; j++){
			alpha = 4 / (1.0 + i + j) + minAlpha;
			float h = 0.0;
			for (int k = 0; k < numInput; k++){
				for (int l = 0; l < numDim + 1; l++){
					h += coors[l][k] * weights[l];
				}
			}
			h = sigmoid(h);
			float error = labels[j] - h;
			for (int k = 0; k < numDim + 1; k++){
				weights[k] += alpha*error*coors[k][j];
			}
		}
	}

	float classify = 0.0;
	for (int k = 0; k < numDim + 1; k++){
		classify += weights[k] * dataPoint[k];
	}
	classify = sigmoid(classify);
	if (classify > 0.5){
		labels[numInput] = 1;
	}
	else{
		labels[numInput] = 0;
	}
	cout << "Point " << 0 << " is classified with a: " << labels[numInput] << endl;

	for (int k = 0; k < numDim + 1; k++){
		cout << "Weight " << k << " is: " << weights[k] << endl;
	}

	//Get the second time
	QueryPerformanceCounter(&t2);

	//Get the elapsed time in milliseconds
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << elapsedTime << " milliseconds for sequential run.\n" << endl;

	//Run the Regression code for all extra points using online learning algorithm
	for (int z = 1; z < numPoints; z++){

		//Get the coordinates of the point to be classified
		for (int i = 0; i < numDim+1; i++){
			dataPoint[i] = coors[i][numInput + z];
		}
		cout << z << " data point: " << endl;

		//Time the sequential version using Windows's QueryPerfomanceCounter()

		//Number of ticks per second
		LARGE_INTEGER frequency;
		//Measure times
		LARGE_INTEGER t1, t2;
		//Store time
		double elapsedTime;

		//Fill the frequency variable
		QueryPerformanceFrequency(&frequency);

		//Get the first time
		QueryPerformanceCounter(&t1);

		alpha = minAlpha;

		for (int i = 0; i < numInput + z; i++){
			float h = 0.0;
			for (int j = 0; j < numInput + z; j++){
				for (int k = 0; k < numDim + 1; k++){
					h += coors[k][j]*weights[k];
				}
			}
			h = sigmoid(h);
			float error = labels[i] - h;
			for (int k = 0; k < numDim + 1; k++){
				weights[k] += alpha*error*coors[k][i];
			}
		}

		float classify = 0.0;
		for (int k = 0; k < numDim + 1; k++){
			classify += weights[k] * dataPoint[k];
		}
		classify = sigmoid(classify);
		if (classify > 0.5){
			labels[numInput + z] = 1;
		}
		else{
			labels[numInput + z] = 0;
		}
		cout << "Point " << z << " is classified with a: " << labels[numInput + z] << endl;

		for (int k = 0; k < numDim + 1; k++){
			cout << "Weight " << k << " is: " << weights[k] << endl;
		}
		//Get the second time
		QueryPerformanceCounter(&t2);

		//Get the elapsed time in milliseconds
		elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		cout << elapsedTime << " milliseconds for sequential run.\n" << endl;

	
	}

	//Create an ofstream to print the data to so that it can be used as a labeled data set for another run
	ofstream outputData;

	//Open a file and make sure it is empty before writing to it
	outputData.open("Regression_Test_Updated.txt", ios::out | ios::trunc);

	//Make sure the file opens correctly
	if (!outputData.is_open()){
		cout << "Something went wrong with opening the output file. Check where it is located again." << endl;
		exit(0);
	}

	//Put the total number of data points at the top of the file
	outputData << (numInput + numPoints) << " " << numDim << " " << numLabel << endl;

	//Print each point and its correspoding label
	for (int i = 0; i < numInput + numPoints; i++){
		for (int j = 1; j < numDim+1; j++){
			outputData << coors[j][i] << " " << endl;
		}
		outputData << labels[i] << endl;
	}

	//Close the file once it is written
	outputData.close();

	//Free remaining arrays
	for (int i = 0; i < numDim+1; i++){
		free(coors[i]);
	}
	free(coors);
	free(dataPoint);
	free(sortedlabels);
	free(weights);
	free(labels);

	//Pause on Windows machines to view output
	system("pause");
	return 0;
}
