#include "stdafx.h"
#include "Eurowithd.hpp"
#include "iostream"
#include "vector"
#include <iomanip>

using namespace std;

int main() {
	double K = 55.55;
	int N[2] = { 6,7 };
	double tol = 1e-4;
	double s = 50.0;
		//double s[3] = {50.0,50.0-0.5*exp(-0.02*2.0/12.0),50.0-0.5*exp(-0.02*2.0/12.0)-0.75*exp(-0.02*0.5)};
	vector<vector<double>> a = { {0.0,0.01,0.0,0.01,0.0,0.01},{0.5,0.0,0.5,0.0,0.5,0.0},{0.5,0.0,0.0,0.01,0.75,0.0} };
	//a.push_back(vector<double>());
	double T[2] = { 0.25,7.0 / 12.0 };
	/*Eurowithd euro3(s, K, T[1], 0.02, 0.3, a[2]);
	int Nn = 100;
	for (int i = 0; i < 5; i++) {
		Nn = (i + 1)*100;
		cout << "N:" << Nn << endl;
		cout << setprecision(9) << euro3.Treeprice(Nn, "AP")[2][0] << endl;
	}*/
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			Eurowithd euro(s, K, T[j], 0.02, 0.3, a[i]);
			vector<double> valuese;
			valuese=euro.value(N[j],"EP",tol);
			for (int p = 0; p < valuese.size(); p++) {
				cout << setprecision(9) << valuese[p] << endl;
			}
			vector<double> valuesa;
			valuesa = euro.value(N[j], "AP", tol);
			for (int p = 0; p < valuesa.size(); p++) {
				cout << setprecision(9) << valuesa[p] << endl;

			}
		}
	}
	cout << "2nd finished" << endl;
	Eurowithd euro3(s, K, T[1], 0.02, 0.3, a[2]);
	vector<double> valuese3;
	valuese3=euro3.value(N[1],"EP",tol);
	cout << "3rd" << endl;
	for (int p = 0; p < valuese3.size(); p++) {
		cout << setprecision(9) << valuese3[p] << endl;
	}
	vector<double> valuesa3;
	valuesa3 = euro3.value(N[1], "AP", tol);
	for (int p = 0; p < valuesa3.size(); p++) {
		cout << setprecision(9) << valuesa3[p] << endl;
	}

	return 0;
}