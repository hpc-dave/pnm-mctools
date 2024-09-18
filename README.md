# pnm-mctools
This repository is intended as an extension of the OpenPNM framework for convenient implementation of multicomponent models. It consists of the following parts:
- A tool set based on an OpenPNM network or compatible object
- A Numerical differentiation algorithm
- Common reaction and adsorption implementations
- Additional IO functionality

## Basic Principle
The models are considered to follow classic conservation laws, where:
```math
 \mathrm{Accumulation} = \mathrm{Inflow} - \mathrm{Outflow} + \mathrm{Source}
```
For example transient diffusion-advection transport of a scalar species with first order reaction with reaction rate $`k_r`$ is expressed as:
```math
 \frac{\partial}{\partial t} \phi + \sum_i \vec{n}_i J_{conv, i} + \sum_i \vec{n}_i J_{diff, i} = k_r \phi
```
where the fluxes $`J_i`$ are considered directional with direction $`\vec{n}_i`$. A key concept for the discretization of the equations is the orientation of $`\vec{n}_i`$, as it allows convient formulation of a $`\sum`$ and $`\Delta`$ operator. As an example, consider the following common computation of steady-state hydrodynamics:
```math
    \sum_j g \Delta P_{ij} = 0
```
at pore $`i`$, its neighbors $`j`$ and the conductance `$g$`. This toolset now allows for following implementation:
```python
    # Assume that an instance of mctools 'mctool' has been setup before:
    delta = mctool.Delta()
    sum = mctool.Sum()
    # then the Jacobian can be formulated as:
   J_conv = sum(g, delta)
```
