import numpy as np
from typing import List


def _compute_flux_matrix(Nt: int, Nc: int, *args):
    r"""
    computes matrix of size [Np*Nc, Nt*Nc], where all arguments are multiplied with the last argument

    Parameters
    ----------
        Nt: int
            number of throats
        Nc: int
            number of components
        args
            Factors to multiply with the final argument, where the final argument is a [Nt*Nc, Np*Nc] matrix

    Returns
    -------
    [Np*Nc, Nt*Nc] sized matrix

    Notes
    -----
    The naming is somewhat misleading, since it also works, if the last object is a vector. But if you read
    this I guess you are either way deep enough into the rabbit hole to understand what this function is
    intended for.
    """
    fluxes = args[-1].copy()
    for i in range(len(args)-1):
        arg = args[i]
        if isinstance(arg, list) and len(arg) == Nc:
            fluxes = fluxes.multiply(np.tile(np.asarray(arg), Nt))
        elif isinstance(arg, np.ndarray):
            _arg = np.tile(arg.reshape(-1, 1), reps=(1, Nc)) if arg.size == Nt else arg
            fluxes = fluxes.multiply(_arg.reshape(-1, 1))
        else:
            fluxes = fluxes.multiply(args[i])
    return fluxes


def get_include(Nc: int, include: int | List[int] | None, exclude: int | List[int] | None = None):
    r"""
    computes the include list from a list of excluded components

    Parameters
    ----------
    include: list
        list of included components
    exclude: list
        list of components to exclude

    Returns
    -------
    list of included parameters
    """
    if include is None and exclude is None:
        return include

    if include is not None:
        if isinstance(include, int):
            include = [include]
        elif not isinstance(include, list):
            raise ValueError('provided include is neither integer nor list!')
        return include
    else:
        if isinstance(exclude, int):
            exclude = [exclude]
        elif not isinstance(exclude, list):
            raise ValueError('provided exclude is neither integer nor list!')
        return [n for n in range(Nc) if n not in exclude]


class SumObject:
    r"""
    A helper object, which acts as a matrix, but also provides convenient overloads
    """
    def __init__(self, matrix, Nc: int, Nt: int):
        r"""
        Initializes the object

        Parameters
        ----------
        matrix:
            A matrix for computing a sum/divergence matrix
        Nc: int
            number of components
        Nt:
            number of throats
        """
        self.matrix = matrix
        self.Nt = Nt
        self.Nc = Nc

    def __mul__(self, other):
        r"""
        multiplication operator

        Parameters
        ----------
        other:
            vector, matrix or scalar to multiply with the internal matrix

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix * other

    def __matmul__(self, other):
        r"""
        matrix multiplication operator

        Parameters
        ----------
        other:
            vector, matrix or scalar to multiply with the internal matrix

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix @ other

    def __call__(self, *args):
        r"""
        calling operator, for convenience to provide high readability of the code

        Parameters
        ----------
        args:
            multiple arguments, which are multiplied with the last instance

        Returns
        -------
        quadratic matrix or vector, depending on the input type
        """
        return self.matrix * _compute_flux_matrix(self.Nt, self.Nc, *args)

    def multiply(self, *args, **kwargs):
        r"""
        Wrapper around the `multiply` function of the underlying matrix
        """
        return self.matrix.multiply(*args, **kwargs)


class MulticomponentTools:
    r"""
    Container around the data required for developing multicomponent transport models with OpenPNM pore networks.

    Parameters
    ----------
    network: OpenPNM network
        network with the geometrical information
    num_components: int
        number of components that shall be modelled with this toolset
    bc: list
        list of boundary conditions for each component. May be 'None', then no
        boundary conditions will be applied

    Notes
    -----

    A multicomponent model is expected to adhere to the conservation law:
    \[f
        \frac{\partial}{\partial t}\phi + \nabla \cdot J - S = 0
    \]
    with the extensive variable $\phi$, time $t$, flux $J$ and source $S$. This toolset provides convenient
    functions to compute gradients, divergences, time derivatives and fluxes. In the case of multiple coupled
    components, the matrix is organized block-wise so the rows of coupled components computed at a single node
    (or pore respectively) are located adjacent to each other.
    """
    def __init__(self, network, num_components: int = 1):
        self.bc = [{} for _ in range(num_components)]
        self.network = network
        self.num_components = num_components

    def get_num_components(self) -> int:
        r"""
        Getter for the number of components

        Returns
        -------
        integer with the number of components
        """
        return self.num_components

    def get_network(self):
        r"""
        Getter for the network

        Returns
        -------
        handle to the underlying network
        """
        return self.network

    def get_bc(self):
        r"""
        Getter for boundary conditions

        Returns
        -------
            handle to stored boundary conditions
        """
        return self.bc
