//
// Created by Master on 2023/2/5.
//
#include "nelder_mead.h"
#include <Eigen/Dense>

using Eigen::VectorXd;

inline double pow2(double x)
{
    return pow(x, 2);
}

double test_func1(VectorXd v)
{
    // CROSS-IN-TRAY FUNCTION
    // argmin: https://www.sfu.ca/~ssurjano/crossit.html
    double x1 = v[0];
    double x2 = v[1];
    double exp_term = abs(100 - sqrt(pow2(x1) + pow2(x2)) / EIGEN_PI);
    double term = sin(x1) * sin(x2) * exp(exp_term);
    term = abs(term) + 1;
    return -0.0001 * pow(term, 0.1);
}

double test_func2(VectorXd v)
{
    // Gomez and Levy function
    // 4*x^2 - 2.1*x^4 + x^6/3 + x*y - 4*y^2 + 4*y^4
    // argmin at (-0.898 , 0.1726) and (0.898 , -0.1726)
    double x = v[0];
    double y = v[1];
    double term1 = 4 * pow(x, 2) - 2.1 * pow(x, 4) + 0.333333 * pow(x, 6);
    double term2 = x * y - 4 * pow(y, 2) + 4 * pow(y, 4);
    return term1 + term2;
}

double test_func3(VectorXd v)
{
    // f=0 at (1,3)
    double x = v[0];
    double y = v[1];
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

double test_func4(VectorXd v)
{
    double x1 = v[0];
    double x2 = v[1];
    double frac1 = 1 + cos(12 * sqrt(pow2(x1) + pow2(x2)));
    double frac2 = 0.5 * (pow2(x1) + pow2(x2)) + 2;
    return -frac1 / frac2;
}

void print_vec(VectorXd vec)
{
    int len = vec.size();
    printf("(");
    for (int i = 0; i < len - 1; i++)
    {
        printf("%lf, ", vec[i]);
    }
    printf("%lf)\n", vec[len - 1]);
}

int main()
{
    auto fn = test_func1;
    auto x = Tsolver::nelder_mead(fn, 2);
    printf("argmin found at point: ");
    print_vec(x);
    printf("f min found is %lf", fn(x));
    return 0;
}