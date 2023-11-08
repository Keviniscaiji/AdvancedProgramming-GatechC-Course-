/*
Author:  Kaiwen Gong
Date last modified: 10/2/2023
Organization: ECE6122 Class

Description:
Implementation of calculation of the electric field at a point in space.
*/

# pragma once
# include <string>




#include "ECE_PointCharge.h"

class ECE_ElectricField : public ECE_PointCharge {
protected:
    double Ex;
    double Ey;
    double Ez;

public:
    ECE_ElectricField(double x_val, double y_val, double z_val, double q_val);
    ECE_ElectricField();
    void computeFieldAt(double x_val, double y_val, double z_val);
    void getElectricField(double &Ex_val, double &Ey_val, double &Ez_val);


};




