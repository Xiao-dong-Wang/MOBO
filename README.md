# MOBO

## About
Multi-Objective Bayesian Optimization

An efficient multi-objective optimization algorithm for expensive black box functions via **Tchebysheff formulation**.

The comparison between optimization results and true Pareto front within only 200 evaluations is shown below. The testcase used here is from the ZDT[1] benchmark problems.

![image](https://github.com/Xiao-dong-Wang/MOBO/blob/master/figures/Pareto_front.png) 

[1]. E. Zitzler, K. Deb, and L. Thiele. 2000. Comparison of Multiobjective Evolutionary Algorithms: Empirical Results. Evolutionary Computation 8, 2 (2000), 173–195.

## Usage
See run.py for multi-objective Bayesian optimization.

See plot_fig.py to obtain the Pareto front.

## Dependencies:

Autograd: https://github.com/HIPS/autograd

Scipy: https://github.com/scipy/scipy

