#include "nelder_mead.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

using Eigen::VectorXd;

inline double pow2(const double x)
{
    return pow(x, 2);
}

double test_func1(const double* v)
{
    // CROSS-IN-TRAY FUNCTION
    // argmin: https://www.sfu.ca/~ssurjano/crossit.html
    const double x1 = v[0];
    const double x2 = v[1];
    const double exp_term = abs(100 - sqrt(pow2(x1) + pow2(x2)) / EIGEN_PI);
    double term = sin(x1) * sin(x2) * exp(exp_term);
    term = abs(term) + 1;
    return -0.0001 * pow(term, 0.1);
}

double test_func2(const double* v)
{
    // Gomez and Levy function
    // 4*x^2 - 2.1*x^4 + x^6/3 + x*y - 4*y^2 + 4*y^4
    // argmin at (-0.898 , 0.1726) and (0.898 , -0.1726)
    const double x = v[0];
    const double y = v[1];
    const double term1 = 4 * pow(x, 2) - 2.1 * pow(x, 4) + 0.333333 * pow(x, 6);
    const double term2 = x * y - 4 * pow(y, 2) + 4 * pow(y, 4);
    return term1 + term2;
}

double test_func3(const double* v)
{
    // f=0 at (1,3)
    const double x = v[0];
    const double y = v[1];
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

double test_func4(const double* v)
{
    const double x1 = v[0];
    const double x2 = v[1];
    const double frac1 = 1 + cos(12 * sqrt(pow2(x1) + pow2(x2)));
    const double frac2 = 0.5 * (pow2(x1) + pow2(x2)) + 2;
    return -frac1 / frac2;
}

void print_vec(const std::vector<double>& vec)
{
    const auto len = vec.size();
    printf("(");
    for (int i = 0; i < len - 1; i++)
    {
        printf("%lf, ", vec[i]);
    }
    printf("%lf)\n", vec[len - 1]);
}

int main()
{
    auto fn = test_func3;
    auto x = Tsolver::nelder_mead(fn, 2);
    printf("argmin found at point: ");
    print_vec(x);
    printf("f min found is %lf", fn(x.data()));
    return 0;
}
