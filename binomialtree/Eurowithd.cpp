#include "stdafx.h"
#include "Eurowithd.hpp"
#include "math.h"
#include "vector"
#include "Eigen"
#include "iostream"

using namespace std;
using namespace Eigen;


Eurowithd::Eurowithd()
{
}

Eurowithd::Eurowithd(double s_val, double k_val, double T_val, double r_val, double sigma_val, vector<double> a_val) :s(s_val), k(k_val), T(T_val), r(r_val), sigma(sigma_val), a(a_val){

}

vector<vector<double>> Eurowithd::Treeprice(int N,string type) {
	vector<double> v;
	vector<vector<double>> vv;
	vector<double> ss;
	double snew = s;
	double dt = T / double(N);
	vector<int> p;
	for (int i = 0; i < 3; i++) {
		p.push_back(floor(double(i+1) /6.0 / dt));
		//cout << "pi:" << p[i] << endl;
	}
	double u = exp(sigma*sqrt(dt));
	double us = u*u;
	double d = 1.0 / u;
	double disc = exp(-r*dt);
	double Prn = (exp(r*dt) - d) / (u - d);
	for (int i = 0; i < floor(T*6.0); i++) {
		snew -= a[2 * i] * exp(-r*double(i) /6.0);
		//snew *=(1- a[2 * i + 1]);
		//cout << i << endl;
	}
	for (int i = 0; i < floor(T*6.0); i++) {
		//snew -= a[2 * i] * exp(-r*double(i) / 6.0);
		snew *=(1- a[2 * i + 1]);
		//cout << i << endl;
	}
	ss.push_back(snew*pow(d, N));
	for (int i = 1; i <=N; i++) {
		ss.push_back(ss[i - 1] * us);
	}
	for (int i = 0; i <= N; i++) {
		v.push_back(max(k - ss[i], 0.0));
	}
	if (type == "EP") {
		for (int j = N - 1; j >= 0;j--) {
			//if (j <= 2) vv.push_back(vector<double>());
			for (int i = 0; i <= j; ++i) {
				v[i]=disc*(v[i] *(1- Prn) + v[i + 1] * Prn);
			}
			if (j <= 2) vv.push_back(v);
			//vv.push_back(v);
		}
	}
	if (type == "AP") {
		int time = floor(T*6.0);
		double portion = 1.0;
		double fix = 0;
		//cout << "time:" << time << endl;
		//for (int m = 0; m < time; m++) {
			for (int j = N - 1; j >=p[time-1]; j--) {
				//if (j <= 2) vv.push_back(vector<double>());
				for (int i = 0; i <= j; i++) {
					//cout << m << j << i << endl;
					ss[i]= ss[i]*u;
					v[i] = max(k - ss[i], disc*(v[i + 1] * Prn + v[i] * (1 - Prn)));
					//cout <<k<<" "<<ss[i]<<" "<< v[i] << endl;
				}
				if (j <= 2) vv.push_back(v);
			}
		//}
		for (int m = 1; m <time; m++) {
			portion = 1.0;
			fix = 0.0;
			for (int j = p[time - m]-1; j >= p[time - 1-m]; j--) {
				for (int i = 0; i < m; i++) {
					portion *= (1 - a[2 *(m-i) + 1]);
					fix += a[2 * (m-i)] * exp(-r*double(p[time-i-1] - j)*dt);
				}
				//if (j <= 2) vv.push_back(vector<double>());
				for (int i = 0; i <= j; i++) {
					ss[i] = ss[i] * u;
					//cout << m << j << i << endl;
					v[i] = max(k - ss[i]/portion -fix, disc*(v[i + 1] * Prn + v[i] * (1 - Prn)));
				}
				if (j <= 2) vv.push_back(v);
			}
		}
		for (int j =p[0]-1; j >= 0; j--) {
			portion = 1.0;
			fix = 0.0;
			//if (j <= 2) vv.push_back(vector<double>());
			for (int i = 0; i <= j; i++) {
				for (int n = 0; n < time; n++) {
					portion *= (1 - a[2 * n + 1]);
					fix += a[2 * n] * exp(-r*double(p[n]-j)*dt);
				}
				ss[i] = ss[i] * u;
				v[i] = max(k - ss[i]/portion-fix, disc*(v[i + 1] * Prn + v[i] * (1 - Prn)));
			}
			if (j <= 2) vv.push_back(v);
		}
	}
	//cout << "vv sieze" <<vv.size() << endl;
	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 6; j++) {
			cout << "vv:" << vv[i][j] << endl;
		}
	}*/
	return vv;
}




vector<double> Eurowithd::value(int N, string type, double tol) {
	vector<double> values;
	double value_old=Treeprice(N, type)[2][0];
	N *= 2;
	vector<vector<double>> vv;
	double value_new = Treeprice(N, type)[2][0];
	double error = abs(value_new - value_old);
	while (error > tol) {
		N *= 2;
		value_old = value_new;
		vv=Treeprice(N,type);
		//value_new = Treeprice(N, type)[N-1][0];
		value_new = vv[2][0];
		error = abs(value_new - value_old);
	}
	values.push_back(value_new);
	cout << N << endl;
	//vector<vector<double>> vv = Treeprice(N, type);
	double dt = T / double(N);
	double u = exp(sigma*sqrt(dt));
	double d = 1.0 / u;
	double delta = (vv[1][1] - vv[1][0]) / (s*u - s*d);
	double gamma =2* ((vv[0][2] - vv[0][1]) / (s*u*u - s) - (vv[0][1] - vv[0][0]) / (s - s*d*d)) / (s*u*u - s*d*d);
	double theta = (vv[0][1] - vv[2][0]) / (2 * dt);
	/*for (int i = N-1; i >N-4; i--) {
		for (int j = 0; j < N; j++) {
			cout << vv[i][j] << endl;
		}
	}*/
	values.push_back(delta);
	values.push_back(gamma);
	values.push_back(theta);
	return values;
}

Eurowithd::~Eurowithd()
{
}