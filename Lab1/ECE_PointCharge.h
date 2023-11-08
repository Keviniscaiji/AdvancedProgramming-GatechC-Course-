/*
Author:  Kaiwen Gong
Date last modified: 09/22/2023
Organization: ECE4122 Class

Description:
Initiation of the electric field
*/
# pragma once
# include <string>



class ECE_PointCharge {
protected:
    double x;
    double y;
    double z;
    double q;

public:
    ECE_PointCharge(double x_val, double y_val, double z_val, double q_val);
    ECE_PointCharge();
    void setLocation(double x_val, double y_val, double z_val);
    void setCharge(double q_val);

    double getX();

    double getY();

    double getZ();
};




