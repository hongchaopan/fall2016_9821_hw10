//
// Created by hongchao on 12/6/16.
//

#include "Heat_Discrete_PDE_DIV.hpp"

EU_Call_Div_PDE::EU_Call_Div_PDE(double S, double K, double T, double vol, double r, double q_div, double t_div, int M1,
                                 double alpha1) : S(S), K(K), T(T), vol(vol), r(r), q_div(q_div), t_div(t_div), M1(M1),
                                                  alpha1(alpha1) {
    // Other variables
    a=r/(vol*vol)-0.5;
    b=std::pow((r/(vol*vol)+0.5),2.0);
    tau_div=0.5*(T-t_div)*vol*vol;
    tau_final=0.5*T*vol*vol;
    x_compute=std::log(S/K)+std::log(1.0-q_div);
    delta_tau1=tau_div/ static_cast<double>(M1);
    delta_x=std::sqrt(delta_tau1/alpha1);

    x_left_tilde=std::log(S/K)+(r-0.5*vol*vol)*T-3*vol*std::sqrt(T);
    x_right_tilde=std::log(S/K)+(r-0.5*vol*vol)*T+3*vol*std::sqrt(T);

    N_left= static_cast<int>(ceil((x_compute-x_left_tilde)/delta_x));
    N_right= static_cast<int>(ceil((x_right_tilde-x_compute)/delta_x));
    N=N_left+N_right;

    x_left=x_compute-N_left*delta_x;
    x_right=x_compute+N_right*delta_x;

    // Domain tau_div_-, tau_final
    x_left_new=x_left-std::log(1.0-q_div);
    x_right_new=x_right-std::log(1.0-q_div);
    x_compute_new=std::log(S/K);

    M2= static_cast<int>(ceil((tau_final-tau_div)/delta_tau1));   // Delta_tau1=alpha1*(delta_x)^2

    delta_tau2=(tau_final-tau_div)/ static_cast<double>(M2);
    alpha2=delta_tau2/std::pow(delta_x, 2.0);

}

double EU_Call_Div_PDE::f(double x) const {
    return K*std::exp(a*x)*std::max(std::exp(x)-1.0,0.0);
}

double EU_Call_Div_PDE::g_right1(double tau) const {

    return K*exp(a*x_right+b*tau)*(exp(x_right)-exp(-2.0*r*tau/(vol*vol)));
}

double EU_Call_Div_PDE::g_right2(double tau) const {
    return K*exp(a*x_right_new+b*tau)*(exp(x_right_new)-exp(-2.0*r*tau/(vol*vol)));
}

std::tuple<MatrixXd, MatrixXd> EU_Call_Div_PDE::Forward_Euler() {
    MatrixXd A1(N - 1, N - 1), A2(N - 1, N - 1);
    A1.setZero();
    A2.setZero();
    for (int n = 0; n < N - 1; ++n) {
        A1(n, n) = 1.0 - 2.0 * alpha1;
        A2(n, n) = 1.0 - 2.0 * alpha2;
        if (n > 0) {
            A1(n, n - 1) = alpha1;
            A1(n - 1, n) = alpha1;
            A2(n, n - 1) = alpha2;
            A2(n - 1, n) = alpha2;
        }
    }
    // from 0 to tau_div_+
    VectorXd U(N - 1);
    for (int n = 0; n < N - 1; ++n) {
        U(n) = f(x_left + delta_x * ((double)n + 1));
    }
    MatrixXd res1(M1 + 1, N + 1);
    res1(0, 0) = g_left(0);
    res1(0, N) = g_right1(0);
    res1.block(0, 1, 1, N - 1) = U.transpose();
    VectorXd bv(N - 1);
    bv.setZero();
    for (int m = 0; m < M1; ++m) {
        bv(0) = alpha1 * g_left(m * delta_tau1);
        bv(N - 2) = alpha1 * g_right1(m * delta_tau1);
        U = A1 * U + bv;
        res1(m + 1, 0) = g_left((m + 1.0) * delta_tau1);
        res1.block(m + 1, 1, 1, N - 1) = U.transpose();
        res1(m + 1, N) = g_right1((m + 1.0) * delta_tau1);
    }
    // from tau_div_- to tau_final
    MatrixXd res2(M2 + 1, N + 1);
    res2(0, 0) = g_left(tau_div);
    res2(0, N) = g_right2(tau_div);
    res2.block(0, 1, 1, N - 1) = U.transpose();
    bv.setZero();
    for (int m = 0; m < M2; ++m) {
        bv(0) = alpha2 * g_left(tau_div + m * delta_tau2);
        bv(N - 2) = alpha2 * g_right2(tau_div + m * delta_tau2);
        U = A1 * U + bv;
        res2(m + 1, 0) = g_left(tau_div + (m + 1.0) * delta_tau2);
        res2.block(m + 1, 1, 1, N - 1) = U.transpose();
        res2(m + 1, N) = g_right2(tau_div + (m + 1.0) * delta_tau2);
    }
    return std::make_tuple(res1, res2);

}

std::tuple<MatrixXd, MatrixXd> EU_Call_Div_PDE::Crank_Nicolson_LU() {
    MatrixXd A1(N - 1, N - 1), A2(N - 1, N - 1), B1(N - 1, N - 1), B2(N - 1, N - 1);
    A1.setZero();
    A2.setZero();
    B1.setZero();
    B2.setZero();
    for (int n = 0; n < N - 1; ++n) {
        A1(n, n) = 1.0 + alpha1;
        A2(n, n) = 1.0 + alpha2;
        B1(n, n) = 1.0 - alpha1;
        B2(n, n) = 1.0 - alpha2;
        if (n > 0) {
            A1(n, n - 1) = -alpha1 / 2.0;
            A1(n - 1, n) = -alpha1 / 2.0;
            B1(n, n - 1) = alpha1 / 2.0;
            B1(n - 1, n) = alpha1 / 2.0;
            A2(n, n - 1) = -alpha2 / 2.0;
            A2(n - 1, n) = -alpha2 / 2.0;
            B2(n, n - 1) = alpha2 / 2.0;
            B2(n - 1, n) = alpha2 / 2.0;
        }
    }
    // from 0 to tau_div_+
    VectorXd U(N - 1);
    for (int n = 0; n < N - 1; ++n) {
        U(n) = f(x_left + delta_x * ((double)n + 1));
    }
    MatrixXd res1(M1 + 1, N + 1);
    res1(0, 0) = g_left(0);
    res1(0, N) = g_right1(0);
    res1.block(0, 1, 1, N - 1) = U.transpose();
    MatrixXd LL1, UU1;
    std::tie(LL1, UU1) = lu_no_pivoting_tridiag(A1);
    VectorXd bv(N - 1);
    bv.setZero();
    VectorXd y;
    for (int m = 0; m < M1; ++m) {
        bv = B1 * U;
        bv(0) += alpha1 / 2.0 *
                 (g_left((m + 1.0) * delta_tau1) + g_left(m * delta_tau1));
        bv(N - 2) += alpha1 / 2.0 * (g_right1((m + 1.0) * delta_tau1) +
                                     g_right1(m * delta_tau1));
        y = forward_subst(LL1, bv);
        U = backward_subst(UU1, y);
        res1(m + 1, 0) = g_left((m + 1.0) * delta_tau1);
        res1.block(m + 1, 1, 1, N - 1) = U.transpose();
        res1(m + 1, N) = g_right1((m + 1.0) * delta_tau1);
    }
    // from tau_div to tau_final
    MatrixXd res2(M2 + 1, N + 1);
    res2(0, 0) = g_left(tau_div);
    res2(0, N) = g_right2(tau_div);
    res2.block(0, 1, 1, N - 1) = U.transpose();
    bv.setZero();
    MatrixXd LL2, UU2;
    std::tie(LL2, UU2) = lu_no_pivoting_tridiag(A2);
    for (int m = 0; m < M2; ++m) {
        bv = B2 * U;
        bv(0) += alpha2 / 2.0 * (g_left(tau_div + (m + 1.0) * delta_tau2) +
                                 g_left(tau_div + m * delta_tau2));
        bv(N - 2) +=
                alpha2 / 2.0 * (g_right2(tau_div + (m + 1.0) * delta_tau2) +
                                g_right2(tau_div + m * delta_tau2));
        y = forward_subst(LL2, bv);
        U = backward_subst(UU2, y);
        res2(m + 1, 0) = g_left(tau_div + (m + 1.0) * delta_tau2);
        res2.block(m + 1, 1, 1, N - 1) = U.transpose();
        res2(m + 1, N) = g_right2(tau_div + (m + 1.0) * delta_tau2);
    }
    return std::make_tuple(res1, res2);
}

std::tuple<double, double, double> EU_Call_Div_PDE::Greeks(const MatrixXd &u_approx) {
    double S_1 = K * exp(x_compute_new - delta_x);
    double S0 = K * exp(x_compute_new);
    double S1 = K * exp(x_compute_new + delta_x);
    double V_1 = exp(-a * (x_compute_new - delta_x) - b * tau_final) *
                 u_approx(M2, N_left - 1);
    double V0 = exp(-a * x_compute_new - b * tau_final) * u_approx(M2, N_left);
    double V1 = exp(-a * (x_compute_new + delta_x) - b * tau_final) *
                u_approx(M2, N_left + 1);

    double Delta = (V1 - V_1) / (S1 - S_1);
    double Gamma = ((S0 - S_1) * V1 - (S1 - S_1) * V0 + (S1 - S0) * V_1) /
                   ((S0 - S_1) * (S1 - S0) * (S1 - S_1) * 0.5);
    double delta_t = -2 * delta_tau2 / (vol * vol);
    double V_approx =
            exp(-a * x_compute_new - b * tau_final) * u_approx(M2, N_left);
    double V_approx_dt =
            exp(-a * x_compute_new - b * (tau_final - delta_tau2)) *
            u_approx(M2 - 1, N_left);
    double Theta = (V_approx - V_approx_dt) / delta_t;
    return std::make_tuple(Delta, Gamma, Theta);
}

double EU_Call_Div_PDE::u_value(const MatrixXd &u_approx) {
    return u_approx(M2,N_left);
}

double EU_Call_Div_PDE::Option_Value(const MatrixXd &u_approx) {
    return exp(-a*x_compute_new-b*tau_final)*u_approx(M2,N_left);
}
