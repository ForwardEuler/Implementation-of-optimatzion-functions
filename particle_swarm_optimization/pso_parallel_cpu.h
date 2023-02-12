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

#ifndef TSOLVER_PSO_H_
#define TSOLVER_PSO_H_
#include "Trand.h"
#include <Eigen/Dense>
#include <cmath>
#include <omp.h>
#include <vector>

namespace Tsolver
{
using Eigen::ArrayXd, std::vector, Trand::rand;
static Trand::seed rng(omp_get_max_threads());

struct Particle
{
    static double w, c1, c2, vmax;
    static int init_range;
    ArrayXd x;
    ArrayXd v;
    ArrayXd pbest;
    double best_fitness;

    Particle(const int d, double (*fn)(const double *))
    {
        x = ArrayXd(d).setRandom();
        v = ArrayXd(d).setRandom();
        x = x * init_range;
        pbest = ArrayXd(d).setZero();
        best_fitness = fn(x.data());
    }

    void update_velocity(const ArrayXd& gbest, const int tid)
    {
        v = w * v + c1 * rand(rng, tid) * (pbest - x) + c2 * rand(rng, tid) * (gbest - x);
        for (auto& i : v)
        {
            if (i > vmax)
                i = vmax;
            if (i < -vmax)
                i = -vmax;
        }
    }

    void update_postion()
    {
        x = x + v;
    }

    void update_pbest(const double curr_value)
    {
        if (curr_value < best_fitness)
        {
            best_fitness = curr_value;
            pbest = x;
        }
    }
};

template <typename T, typename A> int arg_min(const std::vector<T, A>& vec)
{
    return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}

inline vector<double> cast_to_vector(const ArrayXd& vxd)
{
    auto ret = vector<double>(vxd.data(), vxd.data() + vxd.rows());
    return ret;
}

double Particle::w;
double Particle::c1;
double Particle::c2;
double Particle::vmax;
int Particle::init_range;

inline vector<double> pso(double (*fn)(const double *), int n, int d, int max_itr, double vmax = 1, int scale = 10,
                          int w = 1, int c1 = 2, int c2 = 2)
{
    Particle::w = w;
    Particle::c1 = c1;
    Particle::c2 = c2;
    Particle::vmax = vmax;
    Particle::init_range = scale;

    std::vector<Particle> particles;
    ArrayXd gbest;
    double gbest_value;

    particles.resize(n, Particle(d, fn));
    gbest = ArrayXd(d).setZero();
    gbest_value = fn(gbest.data());
    for (int i = 0; i < max_itr; i++)
    {
        std::vector<double> fitness(n);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            #pragma omp for
            for (int j = 0; j < n; j++)
            {
                particles[j].update_velocity(gbest, tid);
                particles[j].update_postion();
                fitness[j] = fn(particles[j].x.data());
                particles[j].update_pbest(fitness[j]);
            }
        }
        const int min_index = arg_min(fitness);
        if (fitness[min_index] < gbest_value)
        {
            gbest_value = fitness[min_index];
            gbest = particles[min_index].x;
        }
    }
    return cast_to_vector(gbest);
}
} // namespace Tsolver
#endif
