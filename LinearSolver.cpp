#include "LinearSolver.hpp"
#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <fstream>
/* Define MatrixXd and VectorXd class
These classes are derived from MatrixXdXd and VectorXdXd
They support index beginning from 1 to n,
while in Eigen index begin with 0
 */

// Forward Substitution

VectorXd forward_subst(const MatrixXd& L, const VectorXd& b) {

    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());
    VectorXd x(b.rows());
    double sum;
    int n = L.cols();
    x(0) = b(0) / L(0, 0);
    for (int j = 1; j < n; j++) {
        sum = 0;
        for (int k = 0; k < j; k++) {
            sum += L(j, k) * x(k);
        }
        x(j) = (b(j) - sum) / L(j, j);
    }
    return x;
}

// Backward Substitution

VectorXd backward_subst(const MatrixXd& U, const VectorXd& b) {

    assert(U.rows() == U.cols());
    assert(U.rows() == b.rows());
    VectorXd x(b.rows());
    double sum;
    int n = U.cols();
    x(n - 1) = b(n - 1) / U(n - 1, n - 1);
    for (int j = n - 2; j >= 0; j--) {
        sum = 0;
        for (int k = j + 1; k < n; k++) {
            sum += U(j, k) * x(k);
        }
        x(j) = (b(j) - sum) / U(j, j);
    }
    return x;
}

// Forward substitution for lower triangular bidiagonal matrix

VectorXd forward_subst_bidiag(const MatrixXd& L, const VectorXd& b) {

    assert(L.rows() == L.cols());
    assert(L.rows() == b.rows());
    const int n = b.rows();
    VectorXd x(n);
    x(0) = b(0) / L(0, 0);
    for (int j = 1; j < n; j++) {
        x(j) = (b(j) - L(j, j - 1) * x(j - 1)) / L(j, j);
    }
    return x;
}

// Backward Substitution for upper triangular bidiagonal matrix

VectorXd backward_subst_bidiag(const MatrixXd& U, const VectorXd& b) {

    assert(U.rows() == U.cols());
    assert(U.rows() == b.rows());
    VectorXd x(b.rows());
    int n = U.cols();
    x(n - 1) = b(n - 1) / U(n - 1, n - 1);
    for (int j = n - 2; j >= 0; j--) {
        x(j) = (b(j) - U(j, j + 1) * x(j + 1)) / U(j, j);
    }
    return x;
}

// Forward substitution for lower triangular banded matrix

VectorXd forward_subst_banded(const MatrixXd& A, const VectorXd& b, int m) {//Forward_substitution for Lower_Triangular MatrixXd Ax = b
    //m is the band length of the lower diagonal part of A.
    assert(A.rows() == b.size());
    assert(A.isLowerTriangular());
    VectorXd v(b.size());
    //v(0) = b(0) / A(0, 0);
    for (int i = 0; i < A.rows(); ++i) {
        double sum = 0;
        for (int j = std::max(0, i - m - 1); j < i; ++j) {
            sum += v(j) * A(i, j);
        }
        v(i) = (b(i) - sum) / A(i, i);
    }
    return v;
}

// Backward Substitution for upper tiangular banded matrix

VectorXd backward_subst_banded(const MatrixXd& U, const VectorXd& b, int m) {

    assert(U.rows() == U.cols());
    assert(U.rows() == b.rows());
    VectorXd x(b.rows());
    double sum;
    int n = U.cols();
    x(n - 1) = b(n - 1) / U(n - 1, n - 1);
    for (int j = n - 2; j >= 0; j--) {
        sum = 0;
        int ub = std::min(n, j + m + 1);
        for (int k = j + 1; k < ub; k++) {
            sum += U(j, k) * x(k);
        }
        x(j) = (b(j) - sum) / U(j, j);
    }
    return x;
}

void lu_helper(MatrixXd &A, MatrixXd &L, MatrixXd &U, int i, int n) {
    if (A(i, i) == 0) {
        throw std::overflow_error("Devided By 0 Error");
    }
    for (int k = i; k < n; k++) {
        U(i, k) = A(i, k);
        L(k, i) = A(k, i) / U(i, i);
    }
    for (int j = i + 1; j < n; j++) {
        for (int k = i + 1; k < n; k++) {
            A(j, k) -= L(j, i) * U(i, k);
        }
    }
}

/* LU decomposition without pivoting
A = LU
 */
std::tuple<MatrixXd, MatrixXd> lu_no_pivoting(MatrixXd A) {

    assert(A.cols() == A.rows());
    const int n = A.cols();
    MatrixXd L = MatrixXd::Zero(n, n);
    MatrixXd U = MatrixXd::Zero(n, n);
    for (int i = 0; i < n - 1; i++) {
        lu_helper(A, L, U, i, n);
    }
    L(n - 1, n - 1) = 1;
    U(n - 1, n - 1) = A(n - 1, n - 1);
    return std::make_tuple(std::move(L), std::move(U));
}

/* LU decomposition with pivoting
A = LU
 */
std::tuple<MatrixXd, MatrixXd, MatrixXd> lu_row_pivoting(MatrixXd A) {

    assert(A.cols() == A.rows());
    const int n = A.cols();
    MatrixXd L = MatrixXd::Zero(n, n);
    MatrixXd U = MatrixXd::Zero(n, n);
    MatrixXd P = MatrixXd::Identity(n, n);

    // Following part is given by psudocode
    for (int i = 0; i < n - 1; i++) {

        int i_max_in_block;
        A.col(i).bottomRows(n - i).cwiseAbs().maxCoeff(&i_max_in_block);
        int i_max = i + i_max_in_block;

        // Switch rows i and i_max of A and P
        A.row(i).swap(A.row(i_max));
        P.row(i).swap(P.row(i_max));
        L.row(i).swap(L.row(i_max));
        lu_helper(A, L, U, i, n);
    }
    L(n - 1, n - 1) = 1;
    U(n - 1, n - 1) = A(n - 1, n - 1);
    return std::make_tuple(std::move(P), std::move(L), std::move(U));
}

/* LU decomposition with pivoting for banded matrix
A = LU
 */
std::tuple<MatrixXd, MatrixXd, MatrixXd> lu_row_pivoting_banded(MatrixXd A, int m) {

    assert(A.cols() == A.rows());
    const int n = A.cols();
    MatrixXd L = MatrixXd::Zero(n, n);
    MatrixXd U = MatrixXd::Zero(n, n);
    MatrixXd P = MatrixXd::Identity(n, n);

    // Following part is given by psudocode
    for (int i = 0; i < n - 1; i++) {

        int i_max_in_block;
        A.col(i).bottomRows(n - i).cwiseAbs().maxCoeff(&i_max_in_block);
        int i_max = i + i_max_in_block;

        // Switch rows i and i_max of A and P
        A.row(i).swap(A.row(i_max));
        P.row(i).swap(P.row(i_max));
        L.row(i).swap(L.row(i_max));
        if (A(i, i) == 0) {
            throw std::overflow_error("Devided By 0 Error");
        }
        int ub = std::min(n, i + m + 1);
        for (int k = i; k < ub; k++) {
            U(i, k) = A(i, k);
            L(k, i) = A(k, i) / U(i, i);
        }
        for (int j = i + 1; j < n; j++) {
            int ub = std::min(n, j + m + 1);
            for (int k = i + 1; k < ub; k++) {
                A(j, k) -= L(j, i) * U(i, k);
            }
        }
    }
    L(n - 1, n - 1) = 1;
    U(n - 1, n - 1) = A(n - 1, n - 1);
    return std::make_tuple(std::move(P), std::move(L), std::move(U));
}

std::tuple<MatrixXd, MatrixXd> lu_no_pivoting_tridiag(MatrixXd A) {
    /*
    LU decomposition without pivoting for tridiagonal matrix
     */
    int n = A.rows();
    MatrixXd L(n, n), U(n, n);
    L.setZero();
    U.setZero();
    for (int i = 0; i < n - 1; i++) {
        U(i, i) = A(i, i);
        U(i, i + 1) = A(i, i + 1);
        L(i, i) = 1;
        L(i + 1, i) = A(i + 1, i) / U(i, i);
        A(i + 1, i + 1) = A(i + 1, i + 1) - L(i + 1, i) * U(i, i + 1);
    }
    L(n - 1, n - 1) = 1;
    U(n - 1, n - 1) = A(n - 1, n - 1);
    return std::make_tuple(L, U);
}

/* LU lineaer solver*/
VectorXd lu_linear_solver(MatrixXd A, VectorXd b) {
    int size = b.size();
    MatrixXd P(size, size), L(size, size), U(size, size);
    lu_row_pivoting(A);
    std::tie(P, L, U) = lu_row_pivoting(A);

    VectorXd y(size, 1), x(size, 1);
    y = forward_subst(L, P * b);
    x = backward_subst(U, y);

    return x;
}

MatrixXd cholesky(MatrixXd A) {

    assert(A.cols() == A.rows());
    if (!A.isApprox(A.transpose())) {
        throw std::overflow_error("The matrix is not symmetric");
    }
    const int n = A.cols();
    MatrixXd U = MatrixXd::Zero(n, n);
    for (int i = 0; i < n - 1; i++) {
        if (A(i, i) <= 0) {
            throw std::overflow_error("The matrix is not SPD");
        }
        U(i, i) = sqrt(A(i, i));
        for (int k = i + 1; k < n; k++) {
            U(i, k) = A(i, k) / U(i, i);
        }
        for (int j = i + 1; j < n; j++)
            for (int k = j; k < n; k++) {
                A(j, k) -= U(i, j) * U(i, k);
            }
    }
    if (A(n - 1, n - 1) <= 0) {
        throw std::overflow_error("The matrix is not SPD");
    }
    U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
    return U;
}

MatrixXd cholesky_banded(MatrixXd A, int m) {

    assert(A.cols() == A.rows());
    if (!A.isApprox(A.transpose())) {
        throw std::overflow_error("The matrix is not symmetric");
    }
    const int n = A.cols();
    MatrixXd U = MatrixXd::Zero(n, n);
    for (int i = 0; i < n - 1; i++) {
        if (A(i, i) <= 0) {
            throw std::overflow_error("The matrix is not SPD");
        }
        U(i, i) = sqrt(A(i, i));
        int ub = std::min(n, i + m + 1);
        for (int k = i + 1; k < ub; k++) {
            U(i, k) = A(i, k) / U(i, i);
        }
        ub = std::min(n, i + m + 2);
        for (int j = i + 1; j < ub; j++)
            for (int k = j; k < ub; k++) {
                A(j, k) -= U(i, j) * U(i, k);
            }
    }
    if (A(n - 1, n - 1) <= 0) {
        throw std::overflow_error("The matrix is not SPD");
    }
    U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
    return U;
}

MatrixXd cholesky_tridiag_spd(MatrixXd A) {

    assert(A.cols() == A.rows());
    if (!A.isApprox(A.transpose())) {
        throw std::overflow_error("The matrix is not symmetric");
    }
    const int n = A.cols();
    MatrixXd U = MatrixXd::Zero(n, n);
    for (int i = 0; i < n - 1; i++) {
        if (A(i, i) <= 0) {
            throw std::overflow_error("The matrix is not SPD");
        }
        U(i, i) = sqrt(A(i, i));
        U(i, i + 1) = A(i, i + 1) / U(i, i);
        A(i + 1, i + 1) -= U(i, i + 1) * U(i, i + 1);
    }
    if (A(n - 1, n - 1) <= 0) {
        throw std::overflow_error("The matrix is not SPD");
    }
    U(n - 1, n - 1) = sqrt(A(n - 1, n - 1));
    return U;
}

VectorXd linear_solve_cholesky(const MatrixXd& A, const VectorXd& b) {
    assert(A.cols() == A.rows());
    assert(A.rows() == b.rows());
    MatrixXd U = cholesky(A);
    VectorXd y = forward_subst(U.transpose(), b);
    VectorXd x = backward_subst(U, y);
    return x;
}

VectorXd linear_solve_cholesky_tridiag(const MatrixXd& A, const VectorXd& b) {

    assert(A.cols() == A.rows());
    assert(A.rows() == b.rows());
    MatrixXd U = cholesky_tridiag_spd(A);
    VectorXd y = forward_subst_bidiag(U.transpose(), b);
    VectorXd x = backward_subst_bidiag(U, y);
    return x;
}

VectorXd linear_solve_cholesky_banded(const MatrixXd & A, const VectorXd & b, int m) {

    assert(A.cols() == A.rows());
    assert(A.rows() == b.rows());
    MatrixXd U = cholesky_banded(A, m);
    VectorXd y = forward_subst_banded(U.transpose(), b, m);
    VectorXd x = backward_subst_banded(U, y, m);
    return x;
}

std::tuple<VectorXd, int> linear_solve_jacobi_iter(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
        double tol, Criterion_Type type) {
    VectorXd x_new = x0;
    // Init value of x_old that dissatisfy stop criterion
    VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
    VectorXd r = b - A * x0;
    MatrixXd Dinv(A.diagonal().asDiagonal().inverse());
    MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
    MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
    VectorXd b_new = Dinv * b;
    int ic = 0;

    StopCriterion stop_crtr(tol, type, r);

    while (stop_crtr(x_old, x_new, r)) {
        x_old = x_new;
        x_new = -Dinv * (L * x_old + U * x_old) + b_new;
        r = b - A * x_new;
        if (ic <= 3) {
            std::cout << std::setprecision(9) << "counter: " << ic << std::endl << x_old << std::endl;
            output(x_old, "Q3.csv", "Jacobi_x_old", 'a');
        }
        ic++;
    }
    return std::make_tuple(x_new, ic);
}

std::tuple<VectorXd, int> linear_solve_gs_iter(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
        double tol, Criterion_Type type) {
    VectorXd x_new = x0;
    // Init value of x_old that dissatisfy stop criterion
    VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
    VectorXd r = b - A * x0;
    MatrixXd D(A.diagonal().asDiagonal());
    MatrixXd Dinv(D.inverse());
    MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
    MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
    VectorXd b_new = forward_subst(D + L, b);
    int ic = 0;

    StopCriterion stop_crtr(tol, type, r);

    while (stop_crtr(x_old, x_new, r)) {
        x_old = x_new;
        x_new = -forward_subst(D + L, U * x_old) + b_new;
        r = b - A * x_new;
        if (ic <= 3) {
            std::cout << std::setprecision(9) << "counter: " << ic << std::endl << x_old << std::endl;
            output(x_old, "Q3.csv", "GS_x_old", 'a');
        }
        ic++;
    }
    return std::make_tuple(x_new, ic);
}

std::tuple<VectorXd, int> linear_solve_sor_iter(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
        double w, double tol, Criterion_Type type) {
    assert(w > 0 && w < 2);
    VectorXd x_new = x0;
    // Init value of x_old that dissatisfy stop criterion
    VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
    VectorXd r = b - A * x0;
    MatrixXd D(A.diagonal().asDiagonal());
    MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
    MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
    VectorXd b_new = w * forward_subst(D + w*L, b);
    int ic = 0;

    StopCriterion stop_crtr(tol, type, r);

    while (stop_crtr(x_old, x_new, r)) {
        x_old = x_new;
        x_new = forward_subst(D + w*L, (1 - w) * D * x_old - w * U * x_old) + b_new;
        ic++;
    }
    return std::make_tuple(x_new, ic);
}

void output(const MatrixXd& A, const std::string& filename, const std::string& title, char flag) {
    std::ofstream myfile;
    if (flag == 'a') {
        myfile.open(filename, std::ios_base::app);
    } else
        myfile.open(filename);
    myfile << std::fixed << std::setprecision(9);
    myfile << title << "\n";
    for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.cols(); j++)
            myfile << A(i, j) << ",";
        myfile << "\n";
    }
    myfile << "\n";
    myfile.close();
}

std::tuple<VectorXd, int> linear_solve_sor_iter_proj(const MatrixXd & A, const VectorXd & b, const VectorXd & x0,
        double w, double tol, Criterion_Type type, std::function<double(double) > cons) {
    assert(w > 0 && w < 2);
    VectorXd x_new = x0;
    // Init value of x_old that dissatisfy stop criterion
    VectorXd x_old = x_new + VectorXd::Constant(x0.size(), tol);
    VectorXd r = b - A * x0;
    MatrixXd D(A.diagonal().asDiagonal());
    MatrixXd U = A.triangularView<Eigen::StrictlyUpper>();
    MatrixXd L = A.triangularView<Eigen::StrictlyLower>();
    VectorXd b_new = w * forward_subst(D + w*L, b);
    int ic = 0;

    StopCriterion stop_crtr(tol, type, r);

    while (stop_crtr(x_old, x_new, r)) {
        x_old = x_new;
        for(int j=0;j<A.rows();++j){
            double temp=0;
            for(int k=0;k<A.rows();++k){
                if(k<j)
                    temp = temp + A(j,k)*x_new(k);
                else if(k>j)
                    temp = temp + A(j,k)*x_old(k);
            }
            x_new(j) = (1-w)*x_old(j)-w/A(j,j)*temp+w*b(j)/A(j,j);
            x_new(j) = std::max(x_new(j), cons(j));
        }
//        x_new = forward_subst(D + w*L, (1 - w) * D * x_old - w * U * x_old) + b_new;
//        for (int i = 0; i < x_new.rows(); ++i){
//            x_new(i) = std::max(x_new(i), cons(i));
//        }
        ic++;
    }
    return std::make_tuple(x_new, ic);
}

void output(const VectorXd& V, const std::string& filename, const std::string& title, char flag) {
    std::ofstream myfile;
    if (flag == 'a') {
        myfile.open(filename, std::ios_base::app);
    } else
        myfile.open(filename);
    myfile << std::fixed << std::setprecision(9);
    myfile << title << "\n";
    for (int i = 0; i < V.rows(); i++) {
        myfile << V(i) << "\n";
    }
    myfile << "\n";
    myfile.close();
}
