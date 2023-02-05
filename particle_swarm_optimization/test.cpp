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

#include <cstdio>
#include <Eigen/Dense>
#include "pso_parallel_cpu.h"

using Eigen::ArrayXd;

inline double pow2(double x)
{
    return pow(x,2);
}

double test_func1(ArrayXd v)
{
    //CROSS-IN-TRAY FUNCTION
    //argmin: https://www.sfu.ca/~ssurjano/crossit.html
    double x1 = v[0];
    double x2 = v[1];
    double exp_term = abs(100 - sqrt(pow2(x1) + pow2(x2))/ EIGEN_PI);
    double term = sin(x1) * sin(x2) * exp(exp_term);
    term = abs(term) + 1;
    return -0.0001 * pow(term, 0.1);
}

double test_func2(ArrayXd v)
{
    //Gomez and Levy function
    //4*x^2 - 2.1*x^4 + x^6/3 + x*y - 4*y^2 + 4*y^4
    //argmin at (-0.898 , 0.1726) and (0.898 , -0.1726)
    double x = v[0];
    double y = v[1];
    double term1 = 4 * pow(x, 2) - 2.1 * pow(x, 4) + 0.333333 * pow(x, 6);
    double term2 = x * y - 4 * pow(y, 2) + 4 * pow(y, 4);
    return term1 + term2;
}

double test_func3(ArrayXd v)
{
    // f=0 at (1,3)
    double x = v[0];
    double y = v[1];
    return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2);
}

double test_func4(ArrayXd v)
{
    double x1 = v[0];
    double x2 = v[1];
    double frac1 = 1 + cos(12*sqrt(pow2(x1)+ pow2(x2)));
    double frac2 = 0.5*(pow2(x1)+ pow2(x2)) + 2;
    return -frac1/frac2;
}

int main()
{
    auto fn = test_func4;
    ArrayXd v = Tsolver::pso(fn, 1000, 2, 1000);
    // std::cout << v << std::endl;
    printf("argmin at (%lf, %lf), f(x) = %lf\n", v[0], v[1], fn(v));
}
