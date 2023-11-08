/*
Author: Kaiwen Gong
Class: ECE6122
Last Date Modified: 11/1/2024
Description:
implementing random walk with cuda to improve performance
*/
#include <iostream>
#include <chrono>
#include <curand_kernel.h>
#include <std>
#define BLOCK_SIZE 256
using namespace std;
__global__ void randomWalk(int numWalkers, int numSteps, int *d_x, int *d_y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(1234, idx, 0, &state);

    int x = 0;
    int y = 0;
    for (int j = 0; j < numSteps; j++) {
        float random_val = curand_uniform(&state);
        if (random_val < 0.25) x++;
        else if (random_val < 0.5) x--;
        else if (random_val < 0.75) y++;
        else y--;
    }

    d_x[idx] = x;
    d_y[idx] = y;
}

int main(int argc, char **argv) {
    int numWalkers = 0, numSteps = 0;

    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-W") == 0) {
            numWalkers = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-I") == 0) {
            numSteps = atoi(argv[i + 1]);
        }
    }

    if (numWalkers == 0 || numSteps == 0) {
        std::cerr << "Invalid inputs.\n";
        return -1;
    }

    int *h_x = new int[numWalkers];
    int *h_y = new int[numWalkers];
    int *d_x, *d_y;

    // Using cudaMalloc
    {
        cudaMalloc(&d_x, numWalkers * sizeof(int));
        cudaMalloc(&d_y, numWalkers * sizeof(int));

        auto start = std::chrono::high_resolution_clock::now();

        randomWalk<<<(numWalkers + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numWalkers, numSteps, d_x, d_y);

        cudaMemcpy(h_x, d_x, numWalkers * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, numWalkers * sizeof(int), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;

        double avg_distance = 0;
        for (int i = 0; i < numWalkers; i++) {
            avg_distance += sqrt(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        }
        avg_distance /= numWalkers;

        std::cout << "Normal CUDA memory Allocation:\n";
        std::cout << "    Time to calculate(microsec): " << elapsed.count() << "\n";
        std::cout << "    Average distance from origin: " << avg_distance << "\n";

        cudaFree(d_x);
        cudaFree(d_y);
    }

    // Using cudaMallocHost
    {
        cudaMallocHost(&h_x, numWalkers * sizeof(int));
        cudaMallocHost(&h_y, numWalkers * sizeof(int));
        cudaMalloc(&d_x, numWalkers * sizeof(int));
        cudaMalloc(&d_y, numWalkers * sizeof(int));

        auto start = std::chrono::high_resolution_clock::now();

        randomWalk<<<(numWalkers + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numWalkers, numSteps, d_x, d_y);

        cudaMemcpy(h_x, d_x, numWalkers * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, numWalkers * sizeof(int), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;

        double avg_distance = 0;
        for (int i = 0; i < numWalkers; i++) {
            avg_distance += sqrt(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        }
        avg_distance /= numWalkers;

        std::cout << "Pinned CUDA memory Allocation:\n";
        std::cout << "    Time to calculate(microsec): " << elapsed.count() << "\n";
        std::cout << "    Average distance from origin: " << avg_distance << "\n";

        cudaFreeHost(h_x);
        cudaFreeHost(h_y);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // Using cudaMallocManaged
    {
        cudaMallocManaged(&h_x, numWalkers * sizeof(int));
        cudaMallocManaged(&h_y, numWalkers * sizeof(int));

        auto start = std::chrono::high_resolution_clock::now();

        randomWalk<<<(numWalkers + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(numWalkers, numSteps, h_x, h_y);

        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;

        double avg_distance = 0;
        for (int i = 0; i < numWalkers; i++) {
            avg_distance += sqrt(h_x[i] * h_x[i] + h_y[i] * h_y[i]);
        }
        avg_distance /= numWalkers;

        std::cout << "Managed CUDA memory Allocation:\n";
        std::cout << "    Time to calculate(microsec): " << elapsed.count() << "\n";
        std::cout << "    Average distance from origin: " << avg_distance << "\n";

        cudaFree(h_x);
        cudaFree(h_y);
    }

    std::cout << "Bye\n";

    return 0;
}
