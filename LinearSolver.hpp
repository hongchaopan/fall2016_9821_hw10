#ifndef LINEARSOLVER_HPP
#define LINEARSOLVER_HPP
#pragma once
#include <Eigen/Dense>
#include <tuple>
#include <iostream>
#include <string>
using namespace std;
using namespace Eigen;


// Forward Substitution
VectorXd forward_subst(const MatrixXd& L, const VectorXd& b);

// Backward Substitution
VectorXd backward_subst(const MatrixXd& U, const VectorXd& b);

// Forward substitution for lower triangular bidiagonal matrix
VectorXd forward_subst_bidiag(const MatrixXd& L, const VectorXd& b);

// Backward Substitution for upper tiangular bidiagonal matrix
VectorXd backward_subst_bidiag(const MatrixXd& U, const VectorXd& b);

// Forward substitution for lower triangular banded matrix
VectorXd forward_subst_banded(const MatrixXd& L, const VectorXd& b, int m);

// Backward Substitution for upper tiangular banded matrix
VectorXd backward_subst_banded(const MatrixXd& U, const VectorXd& b, int m);

/* LU decomposition without pivoting
	A = LU
	*/
std::tuple<MatrixXd, MatrixXd> lu_no_pivoting(MatrixXd A);
std::tuple<MatrixXd, MatrixXd> lu_no_pivoting_tridiag(MatrixXd A);
/* LU decomposition with pivoting
	A = LU
	*/
std::tuple<MatrixXd, MatrixXd, MatrixXd> lu_row_pivoting(MatrixXd A);

/* LU decomposition with pivoting for banded matrix
A = LU
*/
std::tuple<MatrixXd, MatrixXd, MatrixXd> lu_row_pivoting_banded(MatrixXd A, int m);

/* LU lineaer solver*/
VectorXd lu_linear_solver(MatrixXd A, VectorXd b);


/* Cholesky decomposition
	A = UtU
	*/
MatrixXd cholesky(MatrixXd A);

/* Cholesky decomposition for m-banded spd matrix
	A(i,j)=0 for abs(i-j)>m
	*/
MatrixXd cholesky_banded(MatrixXd A, int m);

/* Cholesky decomposition for tridiagonal spd matrix
	i.e. 1-banded spd matrix*/
MatrixXd cholesky_tridiag_spd(MatrixXd A);

/* Linear solver using Cholesky decomposition for
	spd matrix
	*/
VectorXd linear_solve_cholesky(const MatrixXd& A, const VectorXd& b);

/* Linear solver using Cholesky decomposition for
	tridiagonal spd matrix
	*/
VectorXd linear_solve_cholesky_tridiag(const MatrixXd& A, const VectorXd& b);

/* Linear solver using Cholesky decomposition for
	banded spd matrix
	*/
VectorXd linear_solve_cholesky_banded(const MatrixXd& A, const VectorXd& b, int m);


/* Define Stop Criterion
	- Residual-based stopping criterion
	- Consecutive approximation stopping criterion*/
enum class Criterion_Type { Resiudal_Based, Consec_Approx };

class StopCriterion {
public:
	StopCriterion(double _tol, Criterion_Type _type, const VectorXd& _r0) 
		: tol(_tol), stop_iter_residual(_tol*_r0.norm()), type(_type) {}
	
	bool operator()(const VectorXd& x_old, const VectorXd& x_new, const VectorXd& r) {

		if (type == Criterion_Type::Resiudal_Based)
			return r.norm() > stop_iter_residual;
		else {
			return (x_old - x_new).norm() > tol;
		}
	}

private:
	double tol;
	double stop_iter_residual;
	Criterion_Type type;
};

/* Jacobi Iteration*/
std::tuple<VectorXd, int> linear_solve_jacobi_iter(const MatrixXd& A, const VectorXd& b, 
	const VectorXd& x0, double tol, Criterion_Type type);

/* Gauss-Siedel Iteration*/
std::tuple<VectorXd, int> linear_solve_gs_iter(const MatrixXd& A, const VectorXd& b,
	const VectorXd& x0, double tol, Criterion_Type type);

/* SOR Iteration*/
std::tuple<VectorXd, int> linear_solve_sor_iter(const MatrixXd& A, const VectorXd& b,
	const VectorXd& x0, double tol, double w, Criterion_Type type);

std::tuple<VectorXd, int> linear_solve_sor_iter_proj(const MatrixXd& A, const VectorXd& b,
	const VectorXd& x0, double tol, double w, Criterion_Type type, std::function<double(double)> cons);

void output(const VectorXd& V, const std::string& filename, const std::string& title, char flag);
void output(const MatrixXd& A, const std::string& s, const std::string& title, char flag);

#endif
