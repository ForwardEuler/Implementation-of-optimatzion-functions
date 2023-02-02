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
#include <cmath>
#include <cstdio>
#include <iostream>

using Eigen::VectorXd;
class simplex
{
  public:
    double alpha, gamma, pho, delta;
    VectorXd tuple;

  public:
    simplex(double alpha = 1, double gamma = 2, double pho = 0.5, double delta = 0.5)
        : alpha(alpha), gamma(gamma), pho(pho), delta(delta)
    {
        tuple = VectorXd(5);
        tuple.setRandom();
    }
    void order()
    {
        std::sort(tuple.data(), tuple.data() + tuple.size());
    }
    void reflection(double (*fn)(VectorXd))
    {
    }
    void expansion(double (*fn)(VectorXd))
    {
    }
    void contraction(double (*fn)(VectorXd))
    {
    }
    void shrink(double (*fn)(VectorXd))
    {
    }
};

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
    simplex a;
    print_vec(a.tuple);
    a.order();
    print_vec(a.tuple);
    return 0;
}