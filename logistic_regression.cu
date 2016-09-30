
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
#include <time.h>


#define BLOCKSIZE  32
using namespace std;

//User defined params to alter how the regression is done (how many runs and how much weight to each run)
float minAlpha = 0.01;
int maxIter = 150;
int maxIter2 = 150;

/* PARAMS FOR HALTING VERSION
int numChecks = 10;
float haltThresh = 0.005;
*/

//Set block size for the multiplies
int blockSize = BLOCKSIZE;

//Sigmoid function for logistic regression
float sigmoid(float in){
	return 1.0 / (1 + exp(-1 * in));
}

//Tiled version of matrix multiply
__global__ void MatrixMultiplyKernel(float *devA, float *devB, float *devC, int rows, int cols, int k, float alpha, float beta)
{
	//Get the thread's x and y locations for its run
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	//Allocate shared memory to hold parts of A and B
	__shared__ float tileA[BLOCKSIZE][BLOCKSIZE];
	__shared__ float tileB[BLOCKSIZE][BLOCKSIZE];
	
	//Use sum to get the result for a specific element
	float sum = 0.0;
	
	//Use iter to see if the loop should be run again
	int iter = 0;

	do{
		//Check if the x thread falls within bounds of the matrices
		if ((idy < rows) && (threadIdx.x + BLOCKSIZE*iter < k)){
			tileA[threadIdx.y][threadIdx.x] = devA[threadIdx.x + idy*k + BLOCKSIZE*iter];
		}
		else {
			tileA[threadIdx.y][threadIdx.x] = 0.0;
		}

		//Check if the y thread falls within bounds of the matrices
		if ((threadIdx.y + BLOCKSIZE*iter < k) && (idx < cols)){
			tileB[threadIdx.y][threadIdx.x] = devB[idx + (threadIdx.y + BLOCKSIZE*iter)*cols];
		}
		else {
			tileB[threadIdx.y][threadIdx.x] = 0.0;
		}
		
		//Sync to ensure that all of the data has been grabbed for the tiles in this warp
		__syncthreads();

		//Sum the elements related to the element in C corresponding to idx and idy
		for (int i = 0; i < BLOCKSIZE; i++){
			sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
		}
		
		//Iterate the number done
		iter++;
		
		//Sync the threads again to ensure they have all done their work before going through the loop to get data
		__syncthreads();
		
	//Check if the tiles have covered all of C
	} while (BLOCKSIZE*iter < k);

	//If the thread falls within the matrix C, fill in its element, scaled by alpha and beta
	if ((idy < rows) && (idx < cols)){
		devC[idx + idy*cols] = sum * alpha + devC[idx + idy*cols] * beta;
	}
}

//Edited matrix multiply to calculate distance
__global__ void distKernel(float *devA, float *devB, float *devC, int rows, int cols, int K)
{
 	//Get the thread's x and y locations for its run
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

	//Allocate shared memory to hold parts of A and B
        __shared__ float tileA[BLOCKSIZE][BLOCKSIZE];
        __shared__ float tileB[BLOCKSIZE][BLOCKSIZE];
	
       	//Use sum to get the result for a specific element
	float sum = 0.0;
        
	//Use iter to see if the loop should be run again
	int iter = 0;

        do{
		//Check if the x thread falls within bounds of the matrices
                if ((idy < rows) && (threadIdx.x + BLOCKSIZE*iter < K)){
                        tileA[threadIdx.y][threadIdx.x] = devA[threadIdx.x + idy*K + BLOCKSIZE*iter];
                }
                else {
                        tileA[threadIdx.y][threadIdx.x] = 0.0;
                }

		//Check if the y thread falls within bounds of the matrices
                if ((threadIdx.y + BLOCKSIZE*iter < K) && (idx < cols)){
                        tileB[threadIdx.y][threadIdx.x] = devB[idx + (threadIdx.y + BLOCKSIZE*iter)*cols];
                }
                else {
                        tileB[threadIdx.y][threadIdx.x] = 0.0;
                }
		
		//Sync to ensure that all of the data has been grabbed for the tiles in this warp
                __syncthreads();

		//Sum the squared distance between the terms
                for (int i = 0; i < BLOCKSIZE; i++){
                        sum += (tileA[threadIdx.y][i] - tileB[i][threadIdx.x])*(tileA[threadIdx.y][i] - tileB[i][threadIdx.x]);
                }
		
		//Iterate the number done
                iter++;
		
		//Sync the threads again to ensure they have all done their work before going through the loop to get data
                __syncthreads();
		
	//Check if the tiles have covered all of C	
        } while (BLOCKSIZE*iter < K);

	//If the thread falls within the matrix C, fill in its element, scaled by alpha and beta
        if ((idy < rows) && (idx < cols)){
                 devC[idx + idy*cols] = sum;
        }
}


//Run every element of the matrix Weight through the sigmoid function
__global__ void sigmoidKernel(float *Weight, int rows)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	
	//Ensure the thread is in bounds
	if (i < rows){
		Weight[i] = (1.0 / (1 + exp(-1 * Weight[i])));
	}
}

//Element wise subtraction of matrix A and B, stored in matrix C
__global__ void subtractKernel(float *A, float *B, float *C, int rows)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	
	//Ensure the thread is in bounds
	if (i < rows){
		C[i] = A[i] - B[i];
	}
}

int main()
{
	//Input file to read the labeled data
	ifstream inputData;

	//The matrices that will hold the labeled data
	float** coors;
	float* CudaCoors;
	float* CudaCoorsT;

	//Matrices for weights and kernel outputs
	float* weights;
	float* CudaWeights;
	float* CudaH;
	float* CudaError;

	//Arrays to hold the unsorted labels
	int* labels;
	int* sortedlabels;

	//Variable used to hold one data point
	float* dataPoint;

	//Dev vars for kernels
	float *devCudaWeights, *devOldWeights, *devCudaH, *devCudaError, *devCudaCoors, *devCoors, *devLabels;

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

	//Prompt the user for the number of points being classified for future multiPoint version
	//cout << "How many points do you want to read? ";
	//cin >> numPoints;
	numPoints = 1;

	//Set up the matrices to have a max capacity equal to the sum of the number of labeled and unlabeled points
	//Coordinate matrices
	coors = new float*[numDim + 1];
	CudaCoors = new float[numInput*(numDim + 1)];
	CudaCoorsT = new float[numInput*(numDim + 1)];
	
	//Label matrices
	labels = new int[numInput + numPoints];
	sortedlabels = new int[numDim];
	
	//Weight matrices
	weights = new float[numDim + 1];
	CudaWeights = new float[numDim + 1];
	
	//Additional matrices for kernels 
	CudaH = new float[numInput + numPoints];
	CudaError = new float[numInput + numPoints];
	
	//Used to hold one data point
	dataPoint = new float[numDim + 1];

	//Set up the 2D coors array for the number of dimensions each point is
	//Also randomly set up the weights with values 0-1, as those arrays are the same length
	for (int i = 0; i < numDim + 1; i++){
		coors[i] = new float[numInput];
		weights[i] = ((double)rand() / (RAND_MAX));
		CudaWeights[i] = weights[i];
	}

	//Fill in the coors matrix along with the one-dimensional versions for the kernels
	for (int i = 0; i < numInput; i++){
		
		//First dimension always 1 to improve performance
		coors[0][i] = 1.0;
		
		//Set up the cuda arrays
		CudaCoors[i*(numDim + 1)] = 1.0;
		CudaCoorsT[i] = 1.0;
		
		//Go through the input data and add it to the arrays
		for (int j = 1; j < numDim + 1; j++){
			inputData >> coors[j][i];
			CudaCoors[i*(numDim + 1) + j] = coors[j][i];
			CudaCoorsT[j*numInput + i] = coors[j][i];
		}
		
		//Add the labels to the related arrays
		inputData >> labels[i];
		CudaH[i] = (float)labels[i];
	}

	//Close the input file
	inputData.close();

	//Collect the data points that the user will use to classify
	for (int i = 0; i < numPoints; i++){
		cout << i << " data point: " << endl;
		for (int j = 1; j < numDim + 1; j++){
			cout << j << " dim: ";
			cin >> dataPoint[i];
		}
		cout << endl;
	}

	//Get the coordinates of the point to be classified
	//for (int i = 0; i < numDim + 1; i++){
	//	dataPoint[i] = coors[i][numInput];
	//}

	//Clock to time sequential run
	clock_t t;
        t = clock();

	//SEQUENTIAL CODE
	
	//Run the code for maxIter times
	for (int i = 0; i < maxIter; i++){
		for (int j = 0; j < numInput; j++){
			//Slightly change alpha based on how many times the code has run
			alpha = 4 / (1.0 + i + j) + minAlpha;
			
			//use h to hold the sum of a matrix multiply
			float h = 0.0;
			
			//Calculate h for this run
			for (int k = 0; k < numInput; k++){
				for (int l = 0; l < numDim + 1; l++){
					h += coors[l][k] * weights[l];
				}
			}
			
			//Use h to calculate the error of the weights related to this row of data
			h = sigmoid(h);
			float error = labels[j] - h;
			
			//Update the weights based upon the error and the related coordinate
			for (int k = 0; k < numDim + 1; k++){
				weights[k] += alpha*error*coors[k][j];
			}
		}
	}

	//Use classify to classify the point
	float classify = 0.0;

	for (int k = 0; k < numDim + 1; k++){
		classify += weights[k] * dataPoint[k];
	}

	//Classify the point by summing the products of corresponding weights and data dimensions
	classify = sigmoid(classify);

	//Classify the point
	if (classify > 0.5){
		labels[numInput] = 1;
	}
	else{
		labels[numInput] = 0;
	}

	//Print out the classification
	cout << "Point " << 0 << " is classified with a: " << labels[numInput] << endl;

	//Print out the weights
	for (int k = 0; k < numDim + 1; k++){
		cout << "Weight " << k << " is: " << weights[k] << endl;
	}

	//Get the elapsed time in milliseconds
        t = clock() - t; 
        cout << (((float) t)/CLOCKS_PER_SEC * 1000) << " milliseconds for sequential run.\n" << endl;

	/* CUBLAS params
	
	//Create cuBlas handles to use cuBlas
	cublasHandle_t h;
	cublasCreate(&h);

	
	const float alphaCuda = 1.0f;
	const float alphaCudaNeg = -1.0f;
	const float varAlpha = 0.0001f;
	const float BetaCuda = 0.0f;
	*/
		
	//Allocate room for all of the required matrices
	//Weights
	cudaMalloc((void**)&devCudaWeights, (numDim + 1)*sizeof(float));
	cudaMalloc((void**)&devOldWeights, (numDim + 1)*sizeof(float));
	
	//Errors and labels
	cudaMalloc((void**)&devCudaH, (numInput)*sizeof(float));
	cudaMalloc((void**)&devLabels, (numInput)*sizeof(float));
	cudaMalloc((void**)&devCudaError, (numInput)*sizeof(float));
	
	//Coordinates
	cudaMalloc((void**)&devCudaCoors, (numInput*(numDim + 1))*sizeof(float));
	cudaMalloc((void**)&devCoors, (numInput*(numDim + 1))*sizeof(float));

	
	//Copy over the necessary data from the host
	//Weights
	cudaMemcpy(devCudaWeights, CudaWeights, (numDim + 1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devOldWeights, CudaWeights, (numDim + 1)*sizeof(float), cudaMemcpyHostToDevice);
	
	//Errors and Labels
	cudaMemcpy(devCudaH, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devLabels, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCudaError, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	
	//Coordinates
	cudaMemcpy(devCoors, CudaCoorsT, (numInput*(numDim + 1))*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devCudaCoors, CudaCoors, (numInput*(numDim + 1))*sizeof(float), cudaMemcpyHostToDevice);

	//Set up variables to time the CUDA version
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start timer
	cudaEventRecord(start);

	//CUBLAS VERSION
	//Same steps as the kernel version below
	
	/*for (int i = 0; i < 150; i++){
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, numInput, 1, numDim + 1, &alphaCuda, devCoors, numInput, devCudaWeights, numDim + 1, &BetaCuda, devCudaH, numInput);
	sigmoidKernel << <grid, block >> >(devCudaH, numInput);
	cublasSaxpy(h, numInput, &alphaCudaNeg, devCudaH, 1, devCudaError, 1);
	cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, numDim + 1, 1, numInput, &varAlpha, devCudaCoors, numDim + 1, devCudaError, numInput, &alphaCuda, devCudaWeights, numDim + 1);
	cudaMemcpy(devCudaError, CudaH, (numInput)*sizeof(float), cudaMemcpyHostToDevice);
	}*/
	
	//Set up the variety of grids and blocks needed to effciently complete the kernels
	int gridSize = (numInput + blockSize - 1) / blockSize;

	//Needed for sigmoid and subtraction kernels, which are element-wise on one column matrices
	dim3 block(1, blockSize, 1);
	dim3 grid(1, gridSize, 1);
	dim3 MMBlock(blockSide, blockSide, 1);
	
	//Needed for the first matrix multiply
	int gridCoorX = 1;
	int gridCoorY = (numInput + blockSide - 1) / blockSide;
	dim3 gridCoor(gridCoorX, gridCoorY, 1);

	//Needed for the second matrix multiply
	int gridCoorTranspose = (numDim + blockSide) / blockSide;
	dim3 transposeCoorGrid = (1, gridCoorTranspose, 1);

	//Run as many times as specified by maxIter2
	for (int i = 0; i < maxIter2; i++){
		
		//Multiply the coordinates by the weights, store the result in devCudaH 
		MatrixMultiplyKernel << <gridCoor, MMBlock >> >(devCudaCoors, devCudaWeights, devCudaH, numInput, 1, numDim + 1, 1.0, 0.0);

		//Run the result of the first multiply through the sigmoid function, and calculate the error between
		// actual and expected labels
		sigmoidKernel << <grid, block >> >(devCudaH, numInput);
		subtractKernel << <grid, block >> >(devLabels, devCudaH, devCudaError, numInput);
		
		/* HALTING VERSION - GATHER OLD WEIGHTS
		if(((i % (maxIter2/numChecks)) == 0) && (i != maxIter2)){
                        cudaMemcpy(devOldWeights, devCudaWeights, (numDim+1)*sizeof(float), cudaMemcpyDeviceToDevice);
		}
		*/
		
		//Multiply the coordinates by the error, and use this to update the weights
		MatrixMultiplyKernel << <transposeCoorGrid, MMBlock >> >(devCoors, devCudaError, devCudaWeights, numDim + 1, 1, numInput, 0.0001, 1.0);
	
		/*
		// HALTING VERSION THAT WILL LET THE USER SPECIFY A LEVEL OF ACCURACY REQUIRED BEFORE FINISHING
		//Check to see if the code should be run based off of numChecks
		if(((i % (maxIter2/numChecks)) == 0) && (i != maxIter2)){
		
			//Get the distance between the old weights and the new weights to see if the change is 
			//below the user's threshold
			
                        distKernel << < grid1, MMBlock>> >(devCudaWeights, devOldWeights, devCudaH, 1, 1, numDim + 1);
                        
			//Copy the result back to the host for value checking
			cudaMemcpy(&weightChange, devCudaH, sizeof(float), cudaMemcpyDeviceToHost);
			
			//Halt if valid
                        if(sqrt(weightChange) < haltThresh*(numDim+1)){
                                cout << "Broken on the " << i << "th iteration\n\n";
                                break;
                        }
                }
		*/
	}

	//Copy the weights back to the host
	cudaMemcpy(CudaWeights, devCudaWeights, (numDim + 1)*sizeof(float), cudaMemcpyDeviceToHost);

	//Classify the point by summing the products of corresponding weights and data dimensions
	classify = 0.0;
	for (int k = 0; k < numDim + 1; k++){
		classify += CudaWeights[k] * dataPoint[k];
	}
	
	//Classify the point 
	classify = sigmoid(classify);
	if (classify > 0.5){
		labels[numInput] = 1;
	}
	else{
		labels[numInput] = 0;
	}
	
	//Print out the classifications of the points
	cout << "Point " << 0 << " is classified with a: " << labels[numInput] << endl;

	//Print out the weights calculated
	for (int k = 0; k < numDim + 1; k++){
		cout << "Weight " << k << " is: " << CudaWeights[k] << endl;
	}

	//Finish timing, and calculate runtime
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	//Output runtime
	cout << "GPU Version runtime: " << milliseconds << "\n";

	//Free the CUDA Arrays
	cudaFree(devCudaWeights);
	cudaFree(devOldWeights);
	cudaFree(devCudaCoors);
	cudaFree(devCoors);
	cudaFree(devCudaH);
	cudaFree(devLabels);
	cudaFree(devCudaError);

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
	
	return 0;
}
