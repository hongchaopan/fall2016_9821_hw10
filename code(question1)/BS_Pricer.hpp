#ifndef BS_PRICER_HPP
#define BS_Pricer_HPP
#include<math.h>
#include<string>
#include <boost/math/distributions/normal.hpp>

using namespace std; 


double normalDist(double d)
{
	boost::math::normal_distribution<double> normalVariable;
	return boost::math::cdf(normalVariable, d);
}
double BS_pricer(const double& t, double const& S, const double& K, const double& T, const double& sigma, const double& r, const double& q, string type)
{
	double d1 = (log(S/K)+(r-q+(sigma*sigma/2))*(T-t)) / (sigma*sqrt(T-t));
	double d2 = d1 - sigma*sqrt(T - t); 
	if (type == "call")
	{
		return S*exp(-q*(T - t))*normalDist(d1) - K*exp(-r*(T - t))*normalDist(d2); 
	}
	else
	{
		return K*exp(-r*(T - t)) * normalDist(-d2) - S*exp(-q*(T - t))*normalDist(-d1); 
	}
}




#endif
