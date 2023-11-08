/*
Author:  Kaiwen Gong
Date last modified: 10/2/2023
Organization: ECE6122 Class

Description:
Implementation of initiation of the electric field
*/



#include "ECE_ElectricField.h"
#include <cmath> // for hypothetical formula

ECE_ElectricField::ECE_ElectricField(double x_val, double y_val, double z_val, double q_val)
        : ECE_PointCharge(x_val, y_val, z_val, q_val), Ex(0), Ey(0), Ez(0) {}


ECE_ElectricField::ECE_ElectricField() {

}
void ECE_ElectricField::computeFieldAt(double x_val, double y_val, double z_val) {
    const double k = 9e9;
    q = 1e-6 * q;
    double r = sqrt((x-x_val)*(x-x_val) + (y-y_val)*(y-y_val) + (z-z_val)*(z-z_val));

    // Now, use the modified formula for the electric field
    double factor = (k * q) / (r * r * r);

    Ex = factor * (x_val - x);
    Ey = factor * (y_val - y);
    Ez = factor * (z_val - z);
}


void ECE_ElectricField::getElectricField(double &Ex_val, double &Ey_val, double &Ez_val) {
    Ex_val = Ex;
    Ey_val = Ey;
    Ez_val = Ez;
}


