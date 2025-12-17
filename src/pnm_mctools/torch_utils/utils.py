import numpy as np
from scipy import sparse as sp
import torch

def to_sparse_torch_tensor(A, spformat='csr', device: str | None = None):
    if device is None:
        device = torch.get_default_device()

    if isinstance(A, np.ndarray):
        assert A.ndim < 3, 'cannot handle multidimensional arrays for now'
        graph = np.nonzero(A.reshape(-1, 1))
        B = torch.sparse_coo_tensor(np.array(graph), A[graph], size=A.shape, device=device)
    elif sp.issparse(A):
        if sp.isspmatrix_csr(A):
            B = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, size=A.shape, device=device)
        elif sp.isspmatrix_csc(A):
            B = torch.sparse_csc_tensor(A.indptr, A.indices, A.data, size=A.shape, device=device)
        elif sp.isspmatrix_coo(A):
            B = torch.sparse_coo_tensor(np.array((A.row, A.col)), A.data, device=device)
        else:
            B = to_sparse_torch_tensor(sp.coo_matrix(A), spformat='coo', device=device)
    elif not (torch.is_sparse(A) or torch.is_sparse_csr(A)):
        raise TypeError(f'cannot convert type {type(A)}')
    else:
        B = A
    match spformat:
        case 'coo':
            B = B.to_sparse_coo()
        case 'csr':
            B = B.to_sparse_csr()
        case 'csc':
            B = B.to_sparse_csc()
        case 'bsr':
            B = B.to_sparse_bsr()
        case 'bsc':
            B = B.to_sparse_bsc()
        case _:
            raise TypeError(f'cannot convert to {spformat}')
    return B