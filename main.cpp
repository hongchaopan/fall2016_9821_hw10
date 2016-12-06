#include <iostream>
#include <fstream>
#include "Heat_Discrete_PDE_DIV.hpp"

// Output results to file
const IOFormat MatrixFormat(FullPrecision,0,",","\n");
using namespace std;

// Global constant variable
double S=52, K=50, vol=0.3, r=0.03, T=1.0, t_div=5.0/12.0, q_div=0.02;

void HW10_P4_1(){
    // Domain Discretization with alpha=0.4
    ofstream file("HW10_P4_1.csv");
    file.precision(12);
    file<<"Domain Discretization with alpha=0.4"<<endl;
    double alpha1=0.4;
    for(int M1:{4,16,64,256}){
        EU_Call_Div_PDE eu_div_pde(S,K,T,vol,r,q_div,t_div,M1,alpha1);

        file<<M1<<","<<eu_div_pde.M2<<","<<eu_div_pde.alpha2<<","<<eu_div_pde.N<<","<<eu_div_pde.x_left<<","<<eu_div_pde.x_right<<",";
        file<<eu_div_pde.x_left_new<<","<<eu_div_pde.x_right_new<<","<<eu_div_pde.tau_div<<",";
        file<<eu_div_pde.delta_tau1<<","<<eu_div_pde.delta_tau2<<","<<eu_div_pde.delta_x<<endl;
    }
    file<<endl;

    // Domain Discretization with alpha=4
    file<<"Domain Discretization with alpha=4"<<endl;
    alpha1=4;
    for(int M1:{4,16,64,256}){
        EU_Call_Div_PDE eu_div_pde(S,K,T,vol,r,q_div,t_div,M1,alpha1);

        file<<M1<<","<<eu_div_pde.M2<<","<<eu_div_pde.alpha2<<","<<eu_div_pde.N<<","<<eu_div_pde.x_left<<","<<eu_div_pde.x_right<<",";
        file<<eu_div_pde.x_left_new<<","<<eu_div_pde.x_right_new<<","<<eu_div_pde.tau_div<<",";
        file<<eu_div_pde.delta_tau1<<","<<eu_div_pde.delta_tau2<<","<<eu_div_pde.delta_x<<endl;
    }
    file.close();
}

// Finite difference methods
void HW10_P4_2(){
    // Forward Euler with alpha=0.4
    ofstream file("HW10_P4_2.csv");
    file.precision(12);
    double alpha1=0.4;
    file<<"Forward Euler with alpha=0.4"<<endl;
    for(int M1:{4,16,64,256}){
        EU_Call_Div_PDE eu_div_pde(S,K,T,vol,r,q_div,t_div,M1,alpha1);
        MatrixXd res1, res2;
        tie(res1, res2)=eu_div_pde.Forward_Euler();
        file<<eu_div_pde.u_value(res2)<<","<<eu_div_pde.Option_Value(res2)<<",";
        double delta, gamma, theta;
        tie(delta, gamma, theta)=eu_div_pde.Greeks(res2);
        file<<delta<<","<<gamma<<","<<theta<<endl;
    }
    file<<endl;

    file<<"Crank-Nicolson with LU of tridiagonal and alpha=0.4"<<endl;
    for(int M1:{4,16,64,256}){
        EU_Call_Div_PDE eu_div_pde(S,K,T,vol,r,q_div,t_div,M1,alpha1);
        MatrixXd res1, res2;
        tie(res1, res2)=eu_div_pde.Crank_Nicolson_LU();
        file<<eu_div_pde.u_value(res2)<<","<<eu_div_pde.Option_Value(res2)<<",";
        double delta, gamma, theta;
        tie(delta, gamma, theta)=eu_div_pde.Greeks(res2);
        file<<delta<<","<<gamma<<","<<theta<<endl;
    }
    file.close();
}

void HW10_P4_3(){
    // Get the u_approx
    // Forward Euler with alpha=0.4 and M=4
    ofstream file("HW10_P4_3.csv");
    file.precision(12);
    double alpha1=0.4;
    int M1=4;
    file<<"Forward Euler with alpha=0.4 and M=4"<<endl;
    EU_Call_Div_PDE eu_div_pde(S,K,T,vol,r,q_div,t_div,M1,alpha1);

    MatrixXd res1, res2;
    tie(res1, res2)=eu_div_pde.Forward_Euler();
    file<<"First Domain (before tau_div_+)"<<endl;
    file<<res1.format(MatrixFormat)<<endl;
    file<<"Second Domain (after tau_div_-)"<<endl;
    file<<res2.format(MatrixFormat)<<endl;

    file.close();
}




int main() {
    cout<<"Starting..."<<endl;
    HW10_P4_1();
    HW10_P4_2();
    HW10_P4_3();

    int stop;
    cout<<"Enter 0 to exit: "<<endl;
    cin>>stop;

    return 0;
}