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
\int_V \frac{\partial}{\partial t} \phi \mathrm{d}V + \sum_i \vec{n}_i Q_{conv, i} + \sum_i \vec{n}_i Q_{diff, i} = \int_V k_r \phi \mathrm{d}V
```
where the rates $`Q_i`$ are considered directional with direction $`\vec{n}_i`$. A key concept for the discretization of the equations is the orientation of $`\vec{n}_i`$, as it allows convient formulation of a $`\sum`$ (sum) and $`\Delta`$ (delta) operator. As an example, consider the following common computation of steady-state hydrodynamics:
```math
\sum_j g \Delta P_{ij} = 0
```
at pore $`i`$, its neighbors $`j`$ and the conductance $`g`$. This toolset now allows us to use following implementation:
```python
from pnm_mctools import Operators as ops
# Assume that an instance of mctools 'mctool' has been setup before:
delta = ops.delta(mctool)
sum = ops.sum(mctool)
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
Where the OpenPNM network has $`N_p`$ pores and $`N_t`$ throats. This is the minimal setup, then no boundary conditions will be applied to the system. In practice that means that no manipulation of the matrices or defects is conducted, leading to a no-flux boundary condition for boundary pores. In case we want to apply boundary conditions, we have to do the following:
```python
from pnm_mctools import MulticomponentTools
from pnm_mctools import BoundaryConditions as bc
# define an OpenPNM network 'pn' with two coupled components
mt = MulticomponentTools(network=pn, num_components=2)
bc.set(mt, id=0, label='left', bc={'prescribed': 1.}      # boundary condition for component 0 at pores with the label 'left'
bc.set(mt, id=0, label='right', bc={'prescribed': 0.}     # boundary condition for component 0 at pores with the label 'right'
bc.set(mt, id=1, label='left', bc={'prescribed': 0.}      # boundary condition for component 1 at pores with the label 'left'
bc.set(mt, id=1, label='right', bc={'prescribed': 1.}     # boundary condition for component 1 at pores with the label 'right'
```
For more details have a look at the dedicated section.
Now we have everything set up and can start actually using the tools. Currently supported functionality is:
- `delta`: A $`N_t`$ x $`N_p`$ sparse matrix for evaluating the differences at the throats in `Operators`
- `sum`: A function object, which works as a sparse matrix, but also allows mulitple input values with appropriate scaling; returns a $`N_p`$x$`N_p`$ sparse matrix, requiring either a $`N_t`$ x $`N_p`$ matrix or a vector of length $`N_p`$ in `Operators`
- `ddt`: A $`N_p`$x$`N_p`$ sparse matrix, representing the discretized temporal derivative in `Operators`
- `upwind`: A $`N_t`$ x $`N_p`$ sparse matrix with interpolation based on the upwind scheme, defined by a provided list of fluxes in `Interpolation`
- `central_difference`: A $`N_t`$ x $`N_p`$ sparse matrix with interpolation based on the central difference scheme in `Interpolation`
- `set`: Allows adding or updating a boundary condition in `BoundaryConditions`
- `apply`: Adapts the Jacobian or Defect according to the defined boundary conditions, if necessary, in `BoundaryConditions`
- `conduct_numerical_differentiation`: Returns a $`N_p`$x$`N_p`$ sparse matrix based on numerical differentiation of a provided defect in `NumericalDifferentiation`

This whole toolset assumes that you want to assemble a Jacobian for solving your system. For linear models, the Jacobian and the regular discretized matrices are identical, so you can perform point-iterations. However, in the case that you don't want to assemble Jacobians, make sure that your computations are still correct!


Every function takes the optional arguments `include` or `exclude`, which allow us to optimize the matrix and explicitly control, for wich components the matrices are computed. As an example:
```python
# assume component 0 is transported between pores while component 1 is just accumulating, so there should be transport whatsoever
c_up =  ip.upwind(mctools, fluxes=Q_h, include=0)   # only component 0 is subject to upwind fluxes, therefore all entries for component 1 will be 0
sum = ops.sum(mctools, exclude=1)                   # here we explicitly exclude component 1 from the sum operator
J_conv = sum(Q_h, c_up)                             # now the Jacobian for the convective transport has only 0 entries for component 1
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

### Time derivative
Following the finite volume approach, the time derivative is by default discretized via an Euler-backward scheme as:
```math
\int_V \frac{\partial \phi}{\partial t} \mathrm{d}V \approx \frac{\phi^{n+1}-\phi^{n}}{\Delta t} \Delta V
```
where $`\Delta V`$ refers to the discretized volume, usually the `pore.volume` property in the OpenPNM network and $`n`$ refers to the discrete timestep.
When solving the discretized equations, the previous time timestep is an explicit contribution which we can realize by:
```python
ddt = ops.ddt(mctools, dt=dt)   # use whatever timestep you want, by default the `pore.volume` property is included
J_other = ...         # define whatever other contributions you want to
J = ddt + J_other     # assemble the full Jacobian
# enter the time loop
for n in range(num_tsteps):
    x_old = x.copy()                  # copy the current variables to a array with the previous timestep
    G = J * x - ddt * x_old           # define the initial defect
    for n in range(num_iterations):
        dx = scipy.sparse.linalg.spsolve(J, -G)  # solve the system
        x += dx                                  # update field values
        # update the Jacobian in case on nonlinear dependencies
        # ....
        G = J * x - ddt * x_old                  # compute new defect for convergence test
        if not np.any(np.abs(G) > tol):          # check for convergence
            break

```

### Boundary conditions
The boundary pores of a network are identified by a dedicated label, e.g. 'left' or 'inlet'. For each set of pores associated with a label, boundary conditions need to be applied for each component. Each boundary condition is provided to the MulticomponentTools instance:
```python
from pnm_mctools import MulticomponentTools
from pnm_mctools import BoundaryConditions as bc
...
mt = MulticomponentTools(....)
bc.set(mt, label='label1', bc = {'type1': value1})
bc.set(mt, label='label2', bc = {'type2': value2})
```
And in the case of multiple components, we can provide a component ID:
```python
from pnm_mctools import MulticomponentTools
from pnm_mctools import BoundaryConditions as bc
...
mt = MulticomponentTools(....)
bc.set(mt, id=0, label='label1', bc = {'type1': value1})
bc.set(mt, id=1, label='label1', bc = {'type2': value2})
```
Currently, following types of boundaries are supported:
- `noflow`:      There is no additional flow in or out of the pore, technically this is the default if no boundary condition is provided
- `prescribed`:  prescribes a value in the specified pore
- `value`:       alias for `prescribed`
- `rate`:        adds a rate value to the pore as explicit component
- `outflow`:     labels pore as outflow and interpolates value from connected pores, here the additional value term may be ommitted

A special case is the `outflow` boundary condition. There, we assume that transport from the adjacent pores is strictly directed into the labeled pore. No integration is conducted there and the values there are not conservative. To accomodate dependencies on those pores, e.g. in the case of diffusive contributions, we impose a value on the pore which is computed from a weighted average of the connected pores. **Be careful!** If a inverse flow occurs in this pore, the behavior is undefined!

## Numerical Differentiation
As part of the toolset or as standalone functionality, we can obtain an approximate Jacobian by numerically differentiating the defect. For a defect $`G(x)`$, we can express the numerical differentiation as:
```math
J = \frac{\partial G}{\partial x} \approx \frac{G(x+\Delta x) - G(x)}{\Delta x}
```
The big advantage here is, that we can avoid the cumbersome derivation of the (non-linear) defect $`G`$. However, this comes with increased runtime-costs. The here provided functionality exploits a few tricks to reduce the runtime penalty. In the simplest case you can call it by:
```python
def Defect(c):
    # define a defect function here
    # Note, that may have the same shape as the input array c 

mt = MulticomponentTools(...)
c = ...      # an array of shape (Np, Nc) with the solution variables
J, G = mt.NumericalDifferentiation(c, defect_func=Defect)

# continue solving the system
```
Especially for large systems (>5000 rows as a rough indicator), memory limitations may become problematic with the default tool. For this case, you may specify a memory wise optimization, which comes at a slight runtime penalty:
```python
J, G = mt.conduct_numerical_differentiation(c, defect_func=Defect, type='low_mem')
```
A special case of optimization you can achieve, if you know that the Jacobian is only dependent on the components and not on the connected pores, e.g. in the case of reaction in the pore:
```python
J, G = mt.conduct_numerical_differentiation(c, defect_func=Defect, type='constrained')
# or
J, G = mt.conduct_numerical_differentiation(c, defect_func=Defect, axis=1)
```
The option `axis=1` exist for compatibility reason with the [pymrm](https://pypi.org/project/pymrm/) package.
## Reactions
Typically, we introduce a non-linearity to our model when introducing reactions. So most of the times, exploiting numerical differentiation is the easiest way for determining the Jacobians. When defining your defect, take care that you also handle the equations in accordance with the above described discretization. Usually that means that your values need to be multiplied with the pore volume. Let's consider the following reaction:
```math
A + B \longrightarrow C
```
where the reaction rate is defined as
```math
r = k * c_A^2 * c_B
```
The defect of the reaction term could then be defined like this:
```python
def Reaction(c):
   k = 1e2
   r = k * c[:, 0]**2 * c[:, 1]
   G = np.zeros_like(c)
   G[:, 0] = -r
   G[:, 1] = -r
   G[:, 2] = r
   G *= Vp                         # multiplication with pore volume
   return G
```
However, for the specific case of a single educt with a linear reaction rate, e.g.
```math
A \longrightarrow B + C
```
This toolset provides a prepared solution:
```python
from pnm_mctools.Reactions import LinearReaction

# set up a network ...
k = 1e2
J_r = LinearReaction(network=..., num_components=3, k=1e2, educt=0, product=[1,2])
```
## Adsorption
Adsorption may be implemented in a variety of ways and none being fully superior. This repository provides some special implementations for single component adsorption. There, isotherms are employed for the adsorption process. Following types are currently provided:
- `Linear`:     $`y = y_{max} K \cdot c`$
- `Langmuir`:   $`y = y_{max} \frac{K \cdot c}{1+K \cdot c}`$
- `Freundlich`: $`y = y_{max} K c^{\frac{1}{n}}`$
with the surface load $`y`$ in $`\mathrm{mol/m^2}`$. Additionally, a function is provided for computing the Jacobian, where adsorption is approximated in terms of a pseudo-reaction term, allowing for the separate storage of the surface load.
There the pseudo-reaction rate is defined by:
```math
r_{ad} = k_r * (y_{eq} - y)~\mathrm{in mol/(m^2 s)}
```
with a pseudo reaction rate constant $`k_r`$. 

## IO
The IO functionality is independent of the Toolset and contained in the IO.py file and can be imported as:
```python
from pnm_mctools import IO
```
The most important functions here are:
- network_to_vtk: similar function as `project_to_vtk` of OpenPNM allowing for convenient addition of data values to the output
- WritePoresToVTK: Writes the pores as spheres to a VTK file, good for visualization
- WriteThroatsToVTK: Writes the throats as cylinders to a VTK file, good for visualization
