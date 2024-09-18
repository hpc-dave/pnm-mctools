# pnm-mctools
This repository is intended as an extension of the OpenPNM framework for convenient implementation of multicomponent models. It consists of the following parts:
- A tool set based on an OpenPNM network or compatible object
- A Numerical differentiation algorithm
- Common reaction and adsorption implementations
- Additional IO functionality

## Basic Principle
Here we follow the Finite Volume approach for discretization, where the rates are balanced over the throats. 
For example transient diffusion-advection transport of a scalar species with first order reaction with reaction rate $`k_r`$ is expressed as:
```math
 \frac{\partial}{\partial t} \phi + \sum_i \vec{n}_i Q_{conv, i} + \sum_i \vec{n}_i Q_{diff, i} = k_r \phi
```
where the rates $`Q_i`$ are considered directional with direction $`\vec{n}_i`$. A key concept for the discretization of the equations is the orientation of $`\vec{n}_i`$, as it allows convient formulation of a $`\sum`$ (sum) and $`\Delta`$ (delta) operator. As an example, consider the following common computation of steady-state hydrodynamics:
```math
    \sum_j g \Delta P_{ij} = 0
```
at pore $`i`$, its neighbors $`j`$ and the conductance $`g`$. This toolset now allows us to use following implementation:
```python
    # Assume that an instance of mctools 'mctool' has been setup before:
    delta = mctool.Delta()
    sum = mctool.Sum()
    # then the Jacobian can be formulated as:
    J = sum(g, delta)
```
Here, `J` is a quadratic matrix of size $`N_p^2`$, where $`N_p`$ is the number of pores in the system.
For computing the rates in the throats explicitly, we can now easily write:
```python
   Q = g * (delta * P)
```
Finally, we can find the solution to this discretized system via Newton-Raphson iterations:
```python
   G = J * P  # compute initial defect
   for n in range(num_iterations):
       dP = scipy.sparse.linalg.spsolve(J, -G)  # solve the system
       P += dP                                  # update field values
       # update the Jacobian in case on nonlinear dependencies
       # ....
       G = J * P  # compute new defect for convergence test
       if not np.any(np.abs(G) > tol):          # check for convergence
           break
```

## Installation
This package can be installed using pip, e.g. by typing:
```bash
    python -m pip install git+https://github.com/hpc-dave/pnm-mctools.git
```

## The toolset
The very first step to access the functionality consists by importing the `MulticomponentTools` and creating an instance:
```python
from pnm_mctools import MulticomponentTools
# define an OpenPNM network 'pn' with a number of coupled components 'Nc'
mt = MulticomponentTools(network=pn, num_components=Nc)
```
This is the minimal setup, then no explicit boundary conditions will be applied to the system. In practice that means that no manipulation of the matrices or defects is conducted, leading to a no-flux boundary condition for boundary pores. In case we want to apply boundary conditions, we have to do the following:
```python
from pnm_mctools import MulticomponentTools
# define an OpenPNM network 'pn' with two coupled components
bc_0 = {'left': {'prescribed': 1}, 'right': {'prescribed: 0} }    # boundary condition for component 0 at pores with the label 'left' and 'right'
bc_1 = {'left': {'prescribed': 0}, 'right': {'prescribed: 1} }    # boundary condition for component 1 at pores with the label 'left' and 'right'
mt = MulticomponentTools(network=pn, num_components=2, bc=[bc_0, bc_1])
```
For more details have a look at the dedicated section.






