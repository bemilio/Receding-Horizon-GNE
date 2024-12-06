## Receding-horizon solution of dynamic games

This repository contains the code associated with the article:

**Linear-Quadratic Dynamic Games as Receding-Horizon Variational
Inequalities** 
*Emilio Benenati, Sergio Grammatico* 
[arXiv preprint](https://arxiv.org/submit/5818661)
DOI: [arXiv:2408.15703](https://doi.org/10.48550/arXiv.2408.15703)  

The code contains utilities for the computation of Open-Loop Nash equilibria and Closed-Loop Nash equilibria, both in the infinite-horizon unconstrained case and the finite-horizon, constrained case. 

###  Dependencies
- [MPT3 toolbox](https://www.mpt3.org/), for the computation of forward-invariant polyhedra. Tested on v. 3.2.1
- [bemilio/gfne:barebone](https://github.com/bemilio/gfne/tree/barebone), for the computation of finite-horizon, constrained Closed-Loop Nash equilibria. The code is a barebone fork of [forrestlaine/gfne](github.com/forrestlaine/gfne), see also [Laine et al, 2023]. Add the cloned repository to the MATLAB path.

### Running the code

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

### Publication data

The simulation data used for the paper is available with DOI:
```
10.4121/ea21437d-6fb7-4b37-b640-e7cb53a56a45
```

To recreate the plots, load the `.mat` files on MATLAB and run, respectively:
```
matlab/examples/4_zones_power_systems/plot_4_zones_power_system.m
```

```
matlab/examples/4_zones_power_systems/plot_vehicle_platooning.m
```

[Laine et a, 2023] Laine, Forrest, et al. *"The computation of approximate generalized feedback nash equilibria."* SIAM Journal on Optimization (2023): 294-318.
