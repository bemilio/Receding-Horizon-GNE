## Receding-horizon solution of dynamic games

This repository contains the code associated to the article [Linear-Quadratic Dynamic Games as Receding-Horizon Variational
Inequalities](https://arxiv.org/submit/5818661). The code contains utilities for the computation of Open-Loop Nash equilibria and Closed-Loop Nash equilibria, both in the infinite-horizon unconstrained case and the finite-horizon, constrained case. 

Dependencies:
- [MPT3 toolbox](https://www.mpt3.org/), for the computation of forward-invariant polyhedra. Tested on v. 3.2.1
- [bemilio/gfne:barebone](https://github.com/bemilio/gfne/tree/barebone), for the computation of finite-horizon, constrained Closed-Loop Nash equilibria. The code is a barebone fork of [forrestlaine/gfne](github.com/forrestlaine/gfne), see also [Laine et al, 2023]. Add the cloned repository to the MATLAB path.

Minimal example script:

```
matlab/examples/basic_game
```

The simulation results of the article can be reproduced by running the following scripts:

```
matlab/examples/4_zones_power_systems/main_4_zones_power_system.m
```


```
matlab/examples/vehicle_platooning/main_vehicle_platooning.m
```

The `python` code is currently under development and it should be considered as untested.


[Laine et a, 2023] Laine, Forrest, et al. *"The computation of approximate generalized feedback nash equilibria."* SIAM Journal on Optimization (2023): 294-318.
