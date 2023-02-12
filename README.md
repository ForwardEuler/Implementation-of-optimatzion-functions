# C++ Implementation of optimatzion algorithms
### Note:
Implementation requires header only C++ linear algebra library Eigen (https://eigen.tuxfamily.org/)
and C++ standard 20+.

Currently Derivative-free method particle swarm optimization (PSO) and Nelder-Mead Method is implemented.

Algorithm will try to find the global minimum of C function f with signature:

`double f(double*)`

where `double*` is a double array / pointer to f64 array

the argument `int d` in the algorithm is the dimension of input vector,
the behavior of program is undefined if `d` is provided with wrong value.

argmin of objective function will be returned as `std::vector<double>`

====================================================

编译需要C++线性代数库Eigen(https://eigen.tuxfamily.org/)
和C++标准20+。

目前仅简单实现了粒子群优化与Nelder-Mead单纯形算法。

算法将试图找到具有以下签名的C语言函数 f 的全局最小值。

`double f(double*)`。

其中`double*`是一个C数组/指向 f64 输入向量数组的指针

算法中的参数`int d`是输入向量的维度。
如果`d`提供了错误的值，程序的行为将无法定义。

目标函数的argmin将以`std::vector<double>`的形式返回。

## TODO
- [X] C++实现particleswarm
- [X] C++实现matlab优化函数fminsearch （before Feb 5）; https://ww2.mathworks.cn/help/matlab/ref/fminsearch.html
- [ ] 添加文档，算法可视化; Add docs and visualization
- [ ] implement some quasi-Newton methods: Limited-memory BFGS, Newton-cg
