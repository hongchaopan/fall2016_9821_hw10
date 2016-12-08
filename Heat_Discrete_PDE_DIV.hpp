//
// Created by hongchao on 12/6/16.
//

#ifndef MTH9821_HW10_HEAT_DISCRETE_PDE_DIV_HPP
#define MTH9821_HW10_HEAT_DISCRETE_PDE_DIV_HPP

#include <iostream>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>
#include <functional>
#include "LinearSolver.hpp"

using namespace Eigen;

class EU_Call_Div_PDE{
public:
    //ctor
    EU_Call_Div_PDE(double S, double K, double T, double vol, double r, double q_div, double t_div, int M1,
                    double alpha1);
    // Exact function f
    double f(double x)const;
    // Boundary conditions
    double g_left(double tau)const{return 0;}
    double g_right1(double tau)const;
    double g_right2(double tau)const;

    // Finite difference methods
    std::tuple<MatrixXd, MatrixXd> Forward_Euler();
    std::tuple<MatrixXd, MatrixXd> Crank_Nicolson_LU();

    // Greeks
    std::tuple<double, double, double> Greeks(const MatrixXd &u_approx);

    // Values
    double u_value(const MatrixXd &u_approx);
    double Option_Value(const MatrixXd &u_approx);

private:
    double S, K, T, vol, r;
    double q_div, t_div;
    int M1;
    double alpha1;

public:
    int M2, N, N_left, N_right;
    double a, b, tau_div, delta_x;
    double x_left, x_left_new, x_right, x_right_new, x_compute, x_compute_new;
    double tau_final, delta_tau1, delta_tau2, alpha2;
    double x_left_tilde, x_right_tilde;

};



#endif //MTH9821_HW10_HEAT_DISCRETE_PDE_DIV_HPP
