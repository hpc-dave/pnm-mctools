import numpy
from typing import Any
try:
    import scipy
    supports_scipy = True
except ImportError:
    supports_scipy = False

try:
    import torch
    supports_torch = True
    try:
        from .torch_utils import utils
        from .torch_utils import torch_sparse_linalg as torch_linalg
    except ImportError:
        from torch_utils import utils
        from torch_utils import torch_sparse_linalg as torch_linalg
    
except ImportError:
    supports_torch = False



def _get_scipy_iterative_solver(algorithm: str):
    match algorithm.lower():
        case 'cg':
            return scipy.sparse.linalg.cg
        case 'bicg':
            return scipy.sparse.linalg.bicg
        case 'bicgstab':
            return scipy.sparse.linalg.bicgstab
        case 'gmres':
            return scipy.sparse.linalg.gmres
        case _:
            raise ValueError(f'no string support found for iterative algorithm: {algorithm}')

def _get_torch_iterative_solver(algorithm: str):
    match algorithm.lower():
        case 'cg':
            return torch_linalg.cg
        case 'bicgstab':
            return torch_linalg.bicgstab
        case 'gmres':
            return torch_linalg.gmres
        case _:
            raise ValueError(f'no string support found for iterative algorithm: {algorithm}')


class Solver:
    def __init__(self, backend: str = 'scipy', device: str = None):
        if isinstance(backend, str):
            match backend.lower():
                case 'scipy':
                    assert supports_scipy, 'scipy was not installed in the environment'
                case 'torch':
                    assert supports_torch, 'pytorch was not installed in the environment'
                    if device is None:
                        device = torch.get_default_device()
                    else:
                        assert device.startswith('cpu', 'cuda'), f'The chosen device ({device}) is not supported'
                case _:
                    raise ValueError(f'Backend "{backend}" is not supported')
        self.backend = backend
        self.device = device

    def solve(self, A, b, x0=None, M=None, algorithm='bicgstab',
              rtol: float = 1e-10, maxiter: int | None = None, convert_if_required:bool = True):
        r"""
        Solve the linear system Ax = b.
        Parameters
        ----------
        A : sparse matrix type
            The system matrix.
        b : array-like
            The right-hand side vector.
        x0 : array-like, optional
            Initial guess for the solution.
        M : preconditioner, optional
            Preconditioner for the system.
        algorithm : str or callable, optional
            The algorithm to use for solving the system. If a string is provided, it must be one of the supported algorithms.
        rtol : float, optional
            The relative tolerance for the solver.
        maxiter : int, optional
            The maximum number of iterations for the solver.
        convert_if_required : bool, optional
            Whether to convert inputs to appropriate types for the backend.
        Returns
        -------
        x : array-like
            The solution vector.
        info : Any
            Additional information about the solver's performance.
        """
        match self.backend:
            case 'scipy':
                if isinstance(algorithm, str) and algorithm == 'direct':
                    x = scipy.sparse.linalg.spsolve(A, b)
                    info = None
                else:
                    if isinstance(algorithm, str):
                        solver = _get_scipy_iterative_solver(algorithm=algorithm)
                    else:
                        solver = algorithm
                    x, info = solver(A=A, b=b, x0=x0, M=M, rtol=rtol, maxiter=maxiter)
            case 'torch':
                if isinstance(algorithm, str):
                    solver = _get_torch_iterative_solver(algorithm=algorithm)
                else:
                    solver = algorithm
                if convert_if_required:
                    A = utils.to_sparse_torch_tensor(A, device=self.device)
                    if isinstance(b, numpy.ndarray):
                        B = torch.from_numpy(b).to(self.device)
                    elif torch.is_tensor(b):
                        B = b.to(self.device)
                    else:
                        B = b
                    if x0 is not None and isinstance(x0, numpy.ndarray):
                        x0 = torch.from_numpy(x0).to(self.device)
                x, info = solver(A=A, b=B, x0=x0, tol=rtol, maxiter=maxiter)
                if isinstance(b, numpy.ndarray) or scipy.sparse.issparse(b):
                    x = x.cpu().numpy()

            case _:
                raise ValueError(f'backend unknown: {self.backend}')
        return x, info
