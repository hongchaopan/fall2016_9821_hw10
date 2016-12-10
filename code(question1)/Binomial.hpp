#ifndef BINOMIAL_HPP
#define BINOMIAL_HPP

#include"BS_Pricer.hpp"
#include<iostream>
#include<string>
#include<math.h>
#include<Eigen/Dense>
#include<boost/tuple/tuple.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace std; 
using namespace Eigen; 

#define max(a,b) a>b?a:b

tuple <double, double, double, double> Binomial_European(const double& S, const double& K, const double& T, const double& sigma, const double& q, const double& r, const int& N, string type)
{
	double delta_t = T / N; 
	double u = exp(sigma*sqrt(delta_t)); 
	double d = exp(-sigma*sqrt(delta_t));
	double p_prime = (exp((r - q)*delta_t) - d) / (u - d);
	double q_prime = 1 - p_prime;
	VectorXd v_put(N+1); v_put.setZero(); 
	VectorXd v_call(N+1); v_call.setZero(); 
	double V22_P, V21_P, V20_P, V22_C, V21_C, V20_C, V11_P, V10_P, V11_C, V10_C, V00_P, V00_C; 
	V22_P = V21_P = V20_P = V22_C = V21_C = V20_C = V11_P = V10_P = V11_C = V10_C = V00_P = V00_C = 0;
	double S11, S10, S22, S21, S20; 
	double Delta_P, Delta_C, Gamma_P, Gamma_C, Theta_C, Theta_P; 

	for (int i = 0; i < N + 1; i++)
	{
		v_put(i) = max(0, K - S* pow(u, (N - i))*pow(d, i));
		v_call(i) = max(0, -(K - S* pow(u, (N - i))*pow(d, i)));
		
	}
	for (int j = N - 1; j > -1; j--)
	{
		for (int k = 0; k < j + 1; k++)
		{
			v_put(k) = (exp(-r*delta_t)*(v_put(k) * p_prime + v_put(k + 1) * q_prime)); 
			v_call(k) = (exp(-r*delta_t)*(v_call(k) * p_prime + v_call(k + 1) * q_prime));
			if (j == 2)
			{
				V22_P = v_put(0); 
				V21_P = v_put(1); 
				V20_P = v_put(2);

				V22_C = v_call(0); 
				V21_C = v_call(1); 
				V20_C = v_call(2); 
			}
			else if (j == 1)
			{
				V11_P = v_put(0); 
				V10_P = v_put(1); 

				V11_C = v_call(0); 
				V10_C = v_call(1); 
			}
			else if (j == 0)
			{
				V00_P = v_put(0); 

				V00_C = v_call(1);
			}
		}
		S11 = u*S; 
		S10 = d*S; 
		S22 = u*u*S; 
		S21 = u*d*S; 
		S20 = d*d*S; 
		
		Delta_P = (V10_P - V11_P) / (S10 - S11);
		Delta_C = (V10_C - V11_C) / (S10 - S11);
		Gamma_P = ((V20_P - V21_P) / (S20 - S21) - (V21_P - V22_P) / (S21 - S22)) / ((S20 - S22) / 2);
		Gamma_C = ((V20_C - V21_C) / (S20 - S21) - (V21_C - V22_C) / (S21 - S22)) / ((S20 - S22) / 2);
		Theta_P = (V21_P - V00_P) / (2 * delta_t); 
		Theta_C = (V21_C - V00_C) / (2 * delta_t);
	}

	if (type == "call")
	{
		return make_tuple(v_call(0), Delta_C, Gamma_C, Theta_C); 
	}
	else
	{
		return make_tuple(v_put(0), Delta_P, Gamma_P, Theta_P);
	}


}


tuple<double, double, double, double> BBS_European(double t, double S, double K, double T, double sigma, double q, double r, int N, string type)
{
	double delta_t = T / N; 
	double u = exp(sigma*sqrt(delta_t)); 
	//double d = exp(-sigma*sqrt(delta_t));
	double d = 1 / u; 
	double p_rn = (exp((r - q) * delta_t) - d) / (u - d); 
	double q_rn = 1 - p_rn; 
	VectorXd v_put(N + 1); v_put.setZero(); 
	VectorXd v_call(N + 1); v_call.setZero();  

	double S_t = 0;
	double V22_P, V21_P, V20_P, V22_C, V21_C, V20_C, V11_P, V10_P, V11_C, V10_C, V00_P, V00_C;
	V22_P = V21_P = V20_P = V22_C = V21_C = V20_C = V11_P = V10_P = V11_C = V10_C = V00_P = V00_C = 0;
	double S11, S10, S22, S21, S20;
	S11 = S10 = S22 = S21 = S20 = 0;
	double Delta_P, Delta_C, Gamma_P, Gamma_C, Theta_C, Theta_P;


	for (int i = 0; i < N; i++)
	{
		S_t = pow(u, (N - 1 - i))*pow(d, i)*S; 
		v_put(i) = (BS_pricer(t, S_t, K, delta_t, sigma, r, q, "put")); 
		v_call(i) = (BS_pricer(t, S_t, K, delta_t, sigma, r, q, "call")); 
	}

	for (int j = N - 2; j > -1; j--)
	{
		for (int k = 0; k < j + 1; k++)
		{
			v_put(k) = (exp(-r*delta_t)*(v_put(k) * p_rn + v_put(k + 1) * q_rn));
			v_call(k) = (exp(-r*delta_t)*(v_call(k) * p_rn + v_call(k + 1) * q_rn));
			if (j == 2)
			{
				V22_P = v_put(0);
				V21_P = v_put(1);
				V20_P = v_put(2);

				V22_C = v_call(0);
				V21_C = v_call(1);
				V20_C = v_call(2);
			}
			else if (j == 1)
			{
				V11_P = v_put(0);
				V10_P = v_put(1);

				V11_C = v_call(0);
				V10_C = v_call(1);
			}
			else if (j == 0)
			{
				V00_P = v_put(0);

				V00_C = v_call(1);
			}
		}
		S11 = u*S;
		S10 = d*S;
		S22 = u*u*S;
		S21 = u*d*S;
		S20 = d*d*S;

		Delta_P = (V10_P - V11_P) / (S10 - S11);
		Delta_C = (V10_C - V11_C) / (S10 - S11);
		Gamma_P = ((V20_P - V21_P) / (S20 - S21) - (V21_P - V22_P) / (S21 - S22)) / ((S20 - S22) / 2);
		Gamma_C = ((V20_C - V21_C) / (S20 - S21) - (V21_C - V22_C) / (S21 - S22)) / ((S20 - S22) / 2);
		Theta_P = (V21_P - V00_P) / (2 * delta_t);
		Theta_C = (V21_C - V00_C) / (2 * delta_t);
	}

	if (type == "call")
	{
		return make_tuple(v_call(0), Delta_C, Gamma_C, Theta_C);
	}
	else
	{
		return make_tuple(v_put(0), Delta_P, Gamma_P, Theta_P);
	}


}

tuple<double, double, double, double> BBSR_European(double t, double S, double K, double T, double sigma, double q, double r, int N, string type)
{
	auto x = BBS_European(t, S, K, T, sigma, q, r, N, type);
	auto y = BBS_European(t, S, K, T, sigma, q, r, floor(N / 2), type);
	double v_BBS1 = get<0>(x);
	double v_BBS2 = get<0>(y);
	double delta1 = get<1>(x);
	double delta2 = get<1>(y);
	double gamma1 = get<2>(x);
	double gamma2 = get<2>(y);
	double theta1 = get<3>(x);
	double theta2 = get<3>(y);

	double v_BBS = 2 * v_BBS1 - v_BBS2;
	double delta = 2 * delta1 - delta2;
	double gamma = 2 * gamma1 - gamma2;
	double theta = 2 * theta1 - theta2;

	return make_tuple(v_BBS, delta, gamma, theta);
}

tuple<double, double, double, double> Average_Binomeal_European(double S, double K, double T, double sigma, double q, double r, int N, string type)
{
	auto x = Binomial_European(S, K, T, sigma, q, r, N, type);
	auto y = Binomial_European(S, K, T, sigma, q, r, N + 1, type); 

	double v1 = get<0>(x); 
	double v2 = get<0>(y); 
	double delta1 = get<1>(x); 
	double delta2 = get<1>(y); 
	double gamma1 = get<2>(x); 
	double gamma2 = get<2>(y); 
	double theta1 = get<3>(x); 
	double theta2 = get<3>(y); 
	double v = (v1 + v2) / 2; 
	double delta = (delta1 + delta2) / 2; 
	double gamma = (gamma1 + gamma2) / 2; 
	double theta = (theta1 + theta2) / 2; 
	return make_tuple(v, delta, gamma, theta); 
}

double Binomial_European_Barrier(const double& S, const double& K, const double& B,const double& T, const double& sigma, const double& q, const double& r, const int& N, string type, string direction)
{
	double delta_t = T / N; 
	double u = exp(sigma*sqrt(delta_t)); 
	double d = exp(-sigma*sqrt(delta_t));
	double p_prime = (exp((r - q)*delta_t) - d) / (u - d);
	double q_prime = 1 - p_prime;
	VectorXd v_put(N+1); v_put.setZero(); 
	VectorXd v_call(N+1); v_call.setZero(); 
	double V22_P, V21_P, V20_P, V22_C, V21_C, V20_C, V11_P, V10_P, V11_C, V10_C, V00_P, V00_C; 
	V22_P = V21_P = V20_P = V22_C = V21_C = V20_C = V11_P = V10_P = V11_C = V10_C = V00_P = V00_C = 0;
	double S11, S10, S22, S21, S20; 
	double Delta_P, Delta_C, Gamma_P, Gamma_C, Theta_C, Theta_P; 

	for (int i = 0; i < N + 1; i++)
	{
		v_put(i) = max(0, K - S* pow(u, (N - i))*pow(d, i));
		v_call(i) = max(0, -(K - S* pow(u, (N - i))*pow(d, i)));
		if(direction == "dao" && S*pow(u,N-i) * pow(d,i) <=B) v_call(i) = 0; 
		else if (direction == "uao" && S* pow(u,N-1) * pow(d,i) >= B) v_call(i) = 0;
		
	}
	for (int j = N - 1; j > -1; j--)
	{
		for (int k = 0; k < j + 1; k++)
		{
			v_put(k) = (exp(-r*delta_t)*(v_put(k) * p_prime + v_put(k + 1) * q_prime)); 
			v_call(k) = (exp(-r*delta_t)*(v_call(k) * p_prime + v_call(k + 1) * q_prime));
			if(direction == "dao" && S*pow(u,j-k) * pow(d,k) <=B) v_call[k] = 0;
			else if(direction=="uao" && S * pow(u, j-k) * pow(d, k)>=B) v_call[k] = 0;				
			
			if (j == 2)
			{
				V22_P = v_put(0); 
				V21_P = v_put(1); 
				V20_P = v_put(2);

				V22_C = v_call(0); 
				V21_C = v_call(1); 
				V20_C = v_call(2); 
			}
			else if (j == 1)
			{
				V11_P = v_put(0); 
				V10_P = v_put(1); 

				V11_C = v_call(0); 
				V10_C = v_call(1); 
			}
			else if (j == 0)
			{
				V00_P = v_put(0); 

				V00_C = v_call(1);
			}
		}
		S11 = u*S; 
		S10 = d*S; 
		S22 = u*u*S; 
		S21 = u*d*S; 
		S20 = d*d*S; 
		
	}

	if (type == "call")
	{
		return v_call(0);
	}
	else
	{
		return v_put(0);
	}


}

double trinomial_European_Barrier(const double& S, const double& K, const double& B, const double& T, const double& sigma, const double& q, const double& r, const int& N, string type, string direction)
{
	double delta_t = T / N;
	double u = exp(sigma*sqrt(3 * delta_t));
	double d = exp(-sigma*sqrt(3 * delta_t));
	double p_u = 1.0 / 6 + (r - q - 0.5*sigma*sigma) * sqrt(delta_t / (12.0*sigma*sigma));
	double p_m = 2.0 / 3;
	double p_d = 1 - p_u - p_m;

	VectorXd v_put(2 * N + 1); v_put.setZero();
	VectorXd v_call(2 * N + 1); v_call.setZero();
	double S10, S12, S20, S22, S24;
	S10 = S*u; 
	S12 = S*d; 
	S20 = S*u*u;
	S22 = S; 
	S24 = S*d*d;
	
	MatrixXd VCall(3, 5); VCall.setZero();
	MatrixXd VPut(3, 5); VPut.setZero();
	double Delta_C, Delta_P, Gamma_C, Gamma_P, Theta_P, Theta_C;

	for (int i = 0; i < 2 * N + 1; i++)
	{
		v_put(i) = max(0, (K - S*pow(u, N - i)));
		v_call(i) = max(0, -(K - S*pow(u, N - i)));
		if (direction == "dao" && S* pow(u, N - i) <= B) v_call(i) = 0;
		else if (direction == "uao" && S* pow(u, N - i) >= B) v_call(i) = 0;
	}

	for (int j = N - 1; j > -1; j--)
	{
		for (int k = 0; k < 2 * j + 1; k++)
		{
			v_put(k) = exp(-r*delta_t) * (v_put(k) * p_u + v_put(k + 1) * p_m + v_put(k + 2) * p_d);
			v_call(k) = exp(-r*delta_t) * (v_call(k)*p_u + v_call(k + 1) * p_m + v_call(k + 2) *p_d);
			if (direction == "dao" && S*pow(u, j - k) <= B) v_call(k) = 0;
			else if (direction == "uao" && S*pow(u, j - k) >= B) v_call(k) = 0;
			if (j == 2)
			{
				VPut(2, 4) = v_put(0);
				VPut(2, 2) = v_put(2);
				VPut(2, 0) = v_put(4);

				VCall(2, 4) = v_call(0);
				VCall(2, 2) = v_call(2);
				VCall(2, 0) = v_call(4);

			}

			if (j == 1)
			{
				VPut(1, 2) = v_put(0);
				VPut(1, 1) = v_put(1);
				VPut(1, 0) = v_put(2);

				VCall(1, 2) = v_call(0);
				VCall(1, 1) = v_call(1);
				VCall(1, 0) = v_call(2);
			}

			if (j == 0)
			{
				VPut(0, 0) = v_put(0);
				VCall(0, 0) = v_call(0);

			}

		}
		S12 = u*S;
		S10 = d*S;
		S24 = u*u*S;
		S22 = u*d*S;
		S20 = d*d*S;

		Delta_P = (VPut(1, 0) - VPut(1, 2)) / (S10 - S12);
		Delta_C = (VCall(1, 0) - VCall(1, 2)) / (S10 - S12);
		Gamma_P = ((VPut(2, 0) - VPut(2, 2)) / (S20 - S22) - (VPut(2, 2) - VPut(2, 4)) / (S22 - S24)) / ((S10 - S12));
		Gamma_C = ((VCall(2, 0) - VCall(2, 2)) / (S20 - S22) - (VCall(2, 2) - VCall(2, 4)) / (S22 - S24)) / ((S10 - S12));
		Theta_P = (VPut(1, 1) - VPut(0, 0)) / (delta_t);
		Theta_C = (VCall(1, 1) - VCall(0, 0)) / (delta_t);

	}


	if (type == "call")
	{
		return v_call(0);
	}
	else
	{
		return v_put(0);
	}


}
#endif
