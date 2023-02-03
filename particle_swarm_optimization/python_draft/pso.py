#  Copyright (c) 2023, Chuan Tian
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#
#
#

#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#
#
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#
#
import numpy as np
from numpy.random import rand

max_itr = 50
n = 100
w = 0.9
c1 = 2
c2 = 2
pbest = np.zeros(2)
gbest = np.zeros(2)


class particle:
    v_max = 2
    v_min = -2

    def __init__(self):
        self.x = np.random.random(2)
        self.v = np.random.random(2)

    def update_velocity(self):
        global gbest, pbest
        self.v = w * self.v + c1 * rand() * (pbest - self.x) + c2 * rand() * (gbest - self.x)

    def update_position(self):
        self.x = self.x + self.v

    def eval_fitness(self, fitness):
        global gbest, pbest
        if fitness(self.x) < fitness(gbest):
            gbest = self.x
        if fitness(self.x) < fitness(pbest):
            pbest = self.x


def pso(fnc):
    l = []
    for i in range(n):
        l.append(particle())
    for i in range(max_itr):
        for p in l:
            p.eval_fitness(fnc)
            p.update_velocity()
            p.update_position()
    print(gbest)


if __name__ == '__main__':
    def fn(x: np.ndarray):
        x1, x2 = x[0], x[1]
        return pow(x1, 2) - 2 * x1 * x2 + 4 * pow(x2, 2) + x1 - 3 * x2

    pso(fn)
