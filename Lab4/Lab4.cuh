/*
Author: Kaiwen Gong
Class: ECE6122
Last Date Modified: 11/1/2024
Description:
implementing random walk with cuda to improve performance
*/

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

using namespace std;
using namespace chrono;

// Define grid and block size for CUDA
#define BLOCK_SIZE 256

// Prototypes
__global__ void randomWalk(int numSteps, int *x, int *y, unsigned int seed);
double computeAverageDistance(int numWalkers, int *x, int *y);
void simulateWalkUsingCudaMalloc(int numWalkers, int numSteps);
void simulateWalkUsingCudaMallocHost(int numWalkers, int numSteps);
void simulateWalkUsingCudaMallocManaged(int numWalkers, int numSteps);

#ifndef UNTITLED_LAB4_CUH
#define UNTITLED_LAB4_CUH


class Lab4 {

};


#endif //UNTITLED_LAB4_CUH
