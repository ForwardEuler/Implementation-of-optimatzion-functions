// Copyright (c) 2023, Chuan Tian
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

using Eigen::VectorXd;
using std::vector;

class Simplex
{
  public:
    int d;
    double alpha, gamma, rho, sigma;
    vector<VectorXd> points;
    VectorXd x0;
    static double (*fn)(VectorXd);

  public:
    Simplex(int d, double alpha = 1, double gamma = 2, double rho = 0.5, double sigma = 0.5)
        : d(d), alpha(alpha), gamma(gamma), rho(rho), sigma(sigma)
    {
        points.resize(d + 1, VectorXd(d));
        for (auto &i : points)
        {
            i.setRandom();
            i = i * 0.3;
        }
    }
    void order()
    {
        std::sort(points.begin(), points.end(), [](const VectorXd &a, const VectorXd &b) { return fn(a) < fn(b); });
        x0 = VectorXd(d).setZero();
        x0 = ((double)1 / d) * std::accumulate(points.begin(), points.begin() + d, x0);
    }
    inline VectorXd reflection() const
    {
        return x0 + rho * (x0 - points.back());
    }
    inline VectorXd expansion(VectorXd &xr) const
    {
        return x0 + gamma * (xr - x0);
    }
    VectorXd contraction(VectorXd &xr, int id) const
    {
        if (id == 0)
            return x0 + alpha * (xr - x0);
        else
            return x0 + alpha * (points.back() - x0);
    }
    void shrink()
    {
        auto s = points.size();
        for (int i = 1; i < s; i++)
        {
            points[i] = points[1] + sigma * (points[i] - points[1]);
        }
    }
};

double (*Simplex::fn)(VectorXd);
VectorXd nelder_mead(int d)
{
    auto f = Simplex::fn;
    Simplex simplex(d);
    bool converge = false;
    for (int i = 0; i < 1000000; i++)
    {
        simplex.order();
        VectorXd xr = simplex.reflection();
        VectorXd x1 = simplex.points[1];
        VectorXd xn = simplex.points[d - 1];
        VectorXd x_last = simplex.points.back();
        double f_xr = f(xr);
        double f_x1 = f(x1);
        if (f_x1 <= f_xr && f_x1 <= f(xn))
            simplex.points.back() = xr;
        else if (f_xr < f_x1)
        {
            VectorXd xe = simplex.expansion(xr);
            if (f(xe) < f_xr)
                simplex.points.back() = xe;
            else
                simplex.points.back() = xr;
        }
        else if (f_xr < f(x_last))
        {
            VectorXd xc = simplex.contraction(xr, 0);
            if (f(xc) < f_xr)
                simplex.points.back() = xc;
            else
                simplex.shrink();
        }
        else
        {
            VectorXd xc = simplex.contraction(xr, 1);
            if (f(xc) < f(x_last))
                simplex.points.back() = xc;
            else
                simplex.shrink();
        }
        VectorXd ds = simplex.points.back() - simplex.points.front();
        if (ds.norm() < 1e-9)
        {
            printf("Terminal condition met at iteration %d : l2 norm < 1e-9\n", i);
            converge = true;
            break;
        }
    }
    if (!converge)
        printf("WARNING: algorithm failed to coverage!\n");
    return simplex.points[0];
}

//================================================================
//==================++++++TEST FUNCTIONS+++++=====================
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
    Simplex::fn = test_func1;
    auto x = nelder_mead(2);
    printf("argmin found at point: ");
    print_vec(x);
    printf("f min found is %lf", Simplex::fn(x));
    return 0;
}