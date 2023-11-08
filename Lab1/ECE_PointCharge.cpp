/*
Author:  Kaiwen Gong
Date last modified: 09/22/2023
Organization: ECE6122 Class

Description:
Calculation of the electric field at a point in space.
*/


#include "ECE_PointCharge.h"

ECE_PointCharge::ECE_PointCharge(double x_val, double y_val, double z_val, double q_val)
        : x(x_val), y(y_val), z(z_val), q(q_val) {}

ECE_PointCharge::ECE_PointCharge(){}

void ECE_PointCharge::setLocation(double x_val, double y_val, double z_val) {
    x = x_val;
    y = y_val;
    z = z_val;
}

void ECE_PointCharge::setCharge(double q_val) {
    q = q_val;
}

double ECE_PointCharge:: getX(){
    return x;
}
double ECE_PointCharge:: getY(){
    return y;
}
double ECE_PointCharge:: getZ(){
    return z;
}
