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
Where the OpenPNM network has $`N_p`$ pores and $`N_t`$ throats. This is the minimal setup, then no explicit boundary conditions will be applied to the system. In practice that means that no manipulation of the matrices or defects is conducted, leading to a no-flux boundary condition for boundary pores. In case we want to apply boundary conditions, we have to do the following:
```python
from pnm_mctools import MulticomponentTools
# define an OpenPNM network 'pn' with two coupled components
bc_0 = {'left': {'prescribed': 1}, 'right': {'prescribed: 0} }    # boundary condition for component 0 at pores with the label 'left' and 'right'
bc_1 = {'left': {'prescribed': 0}, 'right': {'prescribed: 1} }    # boundary condition for component 1 at pores with the label 'left' and 'right'
mt = MulticomponentTools(network=pn, num_components=2, bc=[bc_0, bc_1])
```
For more details have a look at the dedicated section.
Now we have everything set up and can start actually using the tools. Currently supported functionality is:
- Delta: A $`N_t`$ x $`N_p`$ sparse matrix for evaluating the differences at the throats
- Sum: A function object, which returns a $`N_p`$x$`N_p`$ sparse matrix, requiring at least a $`N_t`$ x $`N_p`$ matrix as last input parameter
- DDT: A $`N_p`$x$`N_p`$ sparse matrix, representing the discretized temporal derivative
- Upwind: A $`N_t`$ x $`N_p`$ sparse matrix with interpolation based on the upwind scheme, defined by a provided list of fluxes
- CentralDifference: A $`N_t`$ x $`N_p`$ sparse matrix with interpolation based on the central difference scheme
- SetBC: Allows adding or updating a boundary condition
- ApplyBC: Adapts the Jacobian or Defect according to the defined boundary conditions, if necessary
- NumericalDifferentiation: Returns a $`N_p`$x$`N_p`$ sparse matrix based on numerical differentiation of a provided defect

Every function takes the optional arguments `include` or `exclude`, which allow us to optimize the matrix and explicitly control, for wich components the matrices are computed. As an example:
```python
# assume component 0 is transported between pores while component 1 is just accumulating, so there should be transport whatsoever
c_up =  mt.Upwind(fluxes=Q_h, include=0) # only component 0 is subject to upwind fluxes, therefore all entries for component 1 will be 0
sum = mt.sum(exclude=1)                  # here we explicitly exclude component 1 from the sum operator
J_conv = sum(Q_h, c_up)                  # now the Jacobian for the convective transport has only 0 entries for component 1
                                         # Note, that defining the include and/or exclude in both 'sum' and 'c_up' is redundant, once would be enough
```

### Matrix layout and directional assembly
As mentioned above, the directional assembly is a key feature of this toolset. The direction of the flow is defined by the underlying OpenPNM network, specifically the `throat.conn` array which defines the connectivity of each throat. There, the flow is directed from column 0 to column 1. As an example, the `throat.conn` array may look like this:
```python
[
  [0, 1],
  [1, 2],
  [2, 3]
]
```
Then the flow is directed from pore 0 to pore 1, from pore 1 to pore and so on. The described network looks as follows:
```python
(0) -> (1) -> (2) -> (3)
```
Now let's assume that diffusive rates are of the form $`Q = -D \Delta c`$ 
```python
Q = -D * (delta * c) = [+1, -1, +1]
```
Then transport effectively occurs from pore 0 to pore 1, from pore 2 to pore 1 and from pore 2 to pore 3.

The resulting matrices are assembled in block wise fashion, where each block component within a pore is located in adjacent rows of the matrix. Those blocks are then sorted as the pores numbering. So to find the respective matrix row $`r`$ of a component $`n`$ in a pore $`i`$ with a total number of components $`N_c`$ we can compute:
```python
r = i * N_c + n
```
Note, that this only works out, since counting is based on 0 here!

## Reactions
Coming soon...
## Adsorption
Coming soon...
## IO
Coming soon...
