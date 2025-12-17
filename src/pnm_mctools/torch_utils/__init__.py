try:
    from torch_sparse_linalg import bicgstab, gmres, cg
    from utils import to_sparse_torch_tensor
except ImportError:
    from .torch_sparse_linalg import bicgstab, gmres, cg
    from .utils import to_sparse_torch_tensor
