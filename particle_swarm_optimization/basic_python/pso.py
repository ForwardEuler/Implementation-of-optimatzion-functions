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
            p.eval_fitness(fn)
            p.update_velocity()
            p.update_position()
    print(gbest)


if __name__ == '__main__':
    def fn(x: np.ndarray):
        x1, x2 = x[0], x[1]
        return pow(x1, 2) - 2 * x1 * x2 + 4 * pow(x2, 2) + x1 - 3 * x2

    pso(fn)
