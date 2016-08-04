
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
int maxIter2 = 150;
float setWeight = 1.0;

int blockSize = 1024;
int blockSide = 1024;



float sigmoid(float in){
	return 1.0 / (1 + exp(-1 * in));
}

__global__ void MatrixMultiplyKernel(float *A, float *B, float *C, int rows, int cols, int k, float alpha, float beta)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y*blockDim.y;

	if ((idx < cols) && (idy < rows)){
		float sum = 0.0;
		for (int i = 0; i < k; i++){
			sum += A[idy*k + i] * B[idx + cols*i];
		}
		C[idy*cols + idx] = (C[idy*cols + idx] * beta) + alpha*sum;
	}

}

__global__ void sigmoidKernel(float *Weight, int rows)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	//Only calculate the distance if the thread corresponds to an existing element
	if (i < rows){
		Weight[i] = (1.0 / (1 + exp(-1 * Weight[i])));
	}
}

__global__ void subtractKernel(float *A, float *B, float *C,int rows)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	//Only calculate the distance if the thread corresponds to an existing element
	if (i < rows){
		C[i] = A[i] - B[i];
	}
}

int main()
{
	//Input file to read the labeled data
	ifstream inputData;

	//The arrays that will hold the labeled data
	float** coors;
	float* CudaCoors;
	float* CudaCoorsT;

	//Arrays to hold the distances computed via the GPU and CPU
	float* weights;
	float* CudaWeights;
	float* CudaH;
	float* CudaError;

	//An array to hold the unsorted labels
	int* labels;
	int* sortedlabels;

	//Variables used to hold one data point
	float* dataPoint;

	//Dev Vars
	float *devCudaWeights, *devCudaH, *devCudaError, *devCudaCoors, *devCoors, *devLabels;

	//Variables to hold the number of labeled data points and the number or points being entered
	int numInput;
	int numPoints;
	int numDim;
	int numLabel;
	float alpha = minAlpha;

	//Opens a file whose first line is the number of elements and every subsequent line is an x coordinate, y coordinate,
	//and label that are seperated by spaces
	inputData.open("Synthesized_Data_10000.txt");

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

	//Create best Grid Size
	int gridSize = (numInput + blockSize - 1) / blockSize;

	dim3 block(1, blockSize, 1);
	dim3 grid(1, gridSize, 1);

	//Set up the arrays to have a max capacity equal to the sum of the number of labeled and unlabeled points
	coors = new float*[numDim + 1];
	CudaCoors = new float[numInput*(numDim + 1)];
	CudaCoorsT = new float[numInput*(numDim + 1)];
	labels = new int[numInput + numPoints];
	sortedlabels = new int[numDim];
	weights = new float[numDim + 1];
	CudaWeights = new float[numDim + 1];
	CudaH = new float[numInput + numPoints];
	CudaError = new float[numInput + numPoints];
	dataPoint = new float[numDim + 1];

	//Set up the 2D coors array for the number of dimensions each point is
	for (int i = 0; i < numDim + 1; i++){
		coors[i] = new float[numInput];
		weights[i] = ((double)rand() / (RAND_MAX));
		CudaWeights[i] = weights[i];
	}

	for (int i = 0; i < numInput; i++){
		coors[0][i] = 1.0;
		CudaCoors[i*(numDim + 1)] = 1.0;
		CudaCoorsT[i] = 1.0;
		for (int j = 1; j < numDim + 1; j++){
			inputData >> coors[j][i];
			CudaCoors[i*(numDim + 1) + j] = coors[j][i];
			CudaCoorsT[j*numInput + i] = coors[j][i];
		}
		inputData >> labels[i];
		CudaH[i] = (float)labels[i];
	}

	//Close the input file
	inputData.close();

	//Collect the data points that the user wants classified
	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: " << endl;
		//coors[0][i + numInput] = 0.0;
		//CudaCoors[i + numInput][0] = 0.0;
		for (int j = 1; j < numDim + 1; j++){
			cout << j << " dim: ";
			cin >> dataPoint[i];
			//cin >> coors[j][i + numInput];
			//CudaCoors[i + numInput][j] = coors[j][i + numInput];
		}
		cout << endl;
	}

	//Get the coordinates of the point to be classified
	//for (int i = 0; i < numDim + 1; i++){
	//	dataPoint[i] = coors[i][numInput];
	//}

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

	//Create cuBlas handles to use cuBlas
	cublasHandle_t h;
	cublasCreate(&h);

	const float alphaCuda = 1.0f;
	const float alphaCudaNeg = -1.0f;
	const float varAlpha = 0.0001f;
	const float BetaCuda = 0.0f;

	cudaMalloc((void**)&devCudaWeights, (numDim + 1)*sizeof(float));
	cudaMalloc((void**)&devCudaH, (numInput)*sizeof(float));
	cudaMalloc((void**)&devLabels, (numInput)*sizeof(float));
	cudaMalloc((void**)&devCudaError, (numInput)*sizeof(float));
	cudaMalloc((void**)&devCudaCoors, (numInput*(numDim + 1))*sizeof(float));
	cudaMalloc((void**)&devCoors, (numInput*(numDim + 1))*sizeof(float));

	cudaMemcpy(devCudaWeights, CudaWeights, (numDim + 1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCudaH, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devLabels, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCudaError, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCoors, CudaCoorsT, (numInput*(numDim + 1))*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCudaCoors, CudaCoors, (numInput*(numDim + 1))*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start timer
	cudaEventRecord(start);

	/*for (int i = 0; i < 150; i++){
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, numInput, 1, numDim + 1, &alphaCuda, devCoors, numInput, devCudaWeights, numDim + 1, &BetaCuda, devCudaH, numInput);
	sigmoidKernel << <grid, block >> >(devCudaH, numInput);
	cublasSaxpy(h, numInput, &alphaCudaNeg, devCudaH, 1, devCudaError, 1);
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, numDim + 1, 1, numInput, &varAlpha, devCudaCoors, numDim + 1, devCudaError, numInput, &alphaCuda, devCudaWeights, numDim + 1);
	cudaMemcpy(devCudaError, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	}*/

	dim3 MMBlock(1, blockSide, 1);
	dim3 MMBlock2(1, numDim + 1, 1);
	int gridCoorX = 1;
	int gridCoorY = (numInput + blockSide - 1) / blockSide;
	dim3 gridCoor(gridCoorX, gridCoorY, 1);

	int gridCoorTranspose = (numDim + blockSide) / blockSide;
	dim3 transposeCoorGrid = (1, gridCoorTranspose, 1);

	for (int i = 0; i < maxIter2; i++){

		MatrixMultiplyKernel << <gridCoor, MMBlock >> >(devCudaCoors, devCudaWeights, devCudaH, numInput, 1, numDim + 1, 1.0, 0.0);

		sigmoidKernel << <grid, block >> >(devCudaH, numInput);

		subtractKernel << <grid, block >> >(devLabels, devCudaH, devCudaError, numInput);

		MatrixMultiplyKernel << <transposeCoorGrid, MMBlock2 >> >(devCoors, devCudaError, devCudaWeights, numDim + 1, 1, numInput, 0.0001, 1.0);

		cudaDeviceSynchronize();
	}

	cudaMemcpy(CudaWeights, devCudaWeights, (numDim + 1)*sizeof(float), cudaMemcpyDeviceToHost);

	classify = 0.0;
	for (int k = 0; k < numDim + 1; k++){
		classify += CudaWeights[k] * dataPoint[k];
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
		cout << "Weight " << k << " is: " << CudaWeights[k] << endl;
	}

	//Finish timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "GPU Version runtime: " << milliseconds << "\n";

	cudaFree(devCudaWeights);
	cudaFree(devCudaCoors);
	cudaFree(devCoors);
	cudaFree(devCudaH);
	cudaFree(devLabels);
	cudaFree(devCudaError);

	//Run the Regression code for all extra points using online learning algorithm
	/*for (int z = 1; z < numPoints; z++){
	//Get the coordinates of the point to be classified
	for (int i = 0; i < numDim + 1; i++){
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
	h += coors[k][j] * weights[k];
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
	}*/

	//Create an ofstream to print the data to so that it can be used as a labeled data set for another run
	ofstream outputData;

	//Open a file and make sure it is empty before writing to it
	outputData.open("Regression_Test_Updated_10000.txt", ios::out | ios::trunc);

	//Make sure the file opens correctly
	if (!outputData.is_open()){
		cout << "Something went wrong with opening the output file. Check where it is located again." << endl;
		exit(0);
	}

	//Put the total number of data points at the top of the file
	outputData << (numInput + numPoints) << " " << numDim << " " << numLabel << endl;

	//Print each point and its correspoding label
	for (int i = 0; i < numInput; i++){
		for (int j = 1; j < numDim + 1; j++){
			outputData << coors[j][i] << " " << endl;
		}
		outputData << labels[i] << endl;
	}

	//Close the file once it is written
	outputData.close();

	//Free remaining arrays
	for (int i = 0; i < numDim + 1; i++){
		free(coors[i]);
	}
	free(coors);
	free(CudaCoors);
	free(CudaCoorsT);
	free(dataPoint);
	free(sortedlabels);
	free(weights);
	free(CudaError);
	free(CudaWeights);
	free(CudaH);
	free(labels);
	free(devLabels);
	//Pause on Windows machines to view output
	system("pause");
	return 0;
}
