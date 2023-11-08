/*
Author:  Kaiwen Gong
Date last modified: 09/22/2023
Organization: ECE6122 Class

Description:
The main class of the function
*/


#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <iomanip>
#include "ECE_PointCharge.h"
#include "ECE_ElectricField.h"
#include <cmath>

using namespace std;

vector<std::string> splitStr(string str);

bool split(const std::string &s, char delimiter, std::vector<std::string> &results);
void printInScientific(double value, const std::string& prefix);
void computeFields(const std::vector<std::vector<ECE_PointCharge>>& matrix,int M, int startRow, int endRow, double& ex, double& ey, double& ez, double qx, double qy, double qz, double charge);
int main() {
    unsigned int n = std::thread::hardware_concurrency();
    string strNandM, strXandY, strChargeC, strCharge, strContinue, strThreadNum;

    int N, M;
    int threadNum;
    double distx, disty, charge;
    double qx, qy, qz;

//    cout << "Your computer supports " << n << " concurrent threads." << endl;

    istringstream iss;


    while (true) {
        cout << "Please enter the number of concurrent threads to use: ";
        getline(cin, strThreadNum);
        istringstream iss(strThreadNum);
        if (iss >> threadNum && threadNum > 0) {
            break;
        } else {
            continue;
        }
    }


    while (true) {
        cout << "Please enter the number of rows and columns in the N x M array: ";
        getline(cin, strNandM);
        istringstream iss(strNandM);
        if (iss >> N >> M && N > 0 && M > 0) {
            break;
        } else {
            continue;
        }
    }

    while (true) {
        cout << "Please enter the x and y separation distances in meters: ";
        getline(cin, strXandY);
        istringstream iss(strXandY);
        if (iss >> distx >> disty && distx > 0 && disty > 0) {
            break;
        } else {
            continue;
        }
    }

    while (true) {
        cout << "Please enter the common charge on the points in micro C: ";
        getline(cin, strCharge);
        istringstream iss(strCharge);
        if(iss >> charge){
            break;
        }else{
            continue;
        }
    }

    vector<vector<ECE_PointCharge>> matrix(N,vector<ECE_PointCharge>(M));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double realX = (i - (N - 1) / 2.0) * distx;
            double realY = (j - (M - 1) / 2.0) * disty;
            matrix[i][j] = ECE_PointCharge(realX, realY, 0, charge);
        }
    }


    do {
        double ex = 0, ey = 0, ez = 0;
        while(true){
            cout << "Please enter the location in space to determine the electric field (x y z) in meters: ";
            getline(cin, strChargeC);
            istringstream iss(strChargeC);
            if (iss >> qx >> qy >> qz) {
                break;
            } else {
                continue;
            }
        }

        std::vector<std::thread> threads(threadNum) ;
        int rowsPerThread = N / (threadNum);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n - 1; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == n - 2) ? N : startRow + rowsPerThread; // Last thread takes remaining rows
            threads[i] = thread(computeFields, std::ref(matrix),M, startRow, endRow, std::ref(ex), std::ref(ey), std::ref(ez), qx, qy, qz, charge);
        }
        for (auto& t : threads) {
            t.join();
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        cout << "The electric field at (" << qx << ", " << qy << ", " << qz << ") in V/m is:" << std::endl;
        printInScientific(ex, "Ex");
        printInScientific(ey, "Ey");
        printInScientific(ez, "Ez");

        double eMagnitude = sqrt(ex*ex + ey*ey + ez*ez);
        printInScientific(eMagnitude, "|E|");

        cout << "The calculation took "<< duration.count() <<" microsec!"<< endl;

        cout << "Do you want to enter a new location (Y/N)? "<< endl;
        getline(cin, strContinue);

    } while (strContinue != "N");
    cout << "Bye!";
    return 0;
}

void printInScientific(double value, const std::string& prefix) {
    if (value == 0) {
        std::cout << prefix << " = " << "0.000001 * 10^0" << std::endl;
        return;
    }

    int exponent = std::floor(log10(std::abs(value)));
    double mantissa = value / pow(10, exponent);

    std::cout << prefix << " = " << std::fixed << std::setprecision(4) << mantissa << " * 10^" << exponent << std::endl;
}
void computeFields(const std::vector<std::vector<ECE_PointCharge>>& matrix,int M, int startRow, int endRow, double& ex, double& ey, double& ez, double qx, double qy, double qz, double charge) {
    double localEx = 0, localEy = 0, localEz = 0;
    for (int row = startRow; row < endRow; ++row) {
        for (int col = 0; col < M; ++col) {
            ECE_PointCharge ep = matrix[row][col];
            ECE_ElectricField eef(ep.getX(), ep.getY(), ep.getZ(), charge);
            eef.computeFieldAt(qx, qy, qz);
            double tempEx, tempEy, tempEz;
            eef.getElectricField(tempEx, tempEy, tempEz);
            localEx += tempEx;
            localEy += tempEy;
            localEz += tempEz;
        }
    }

    // Lock mutex and update the shared variables

    ex += localEx;
    ey += localEy;
    ez += localEz;

}