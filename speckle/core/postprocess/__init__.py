from .strain import StrainResult, compute_strain_from_icgn
from .strain_pls import (
    PLSStrainResult,
    compute_strain_pls,
    compute_strain_pls_from_icgn,
    get_vsg_size,
)

# Numba PLS 워밍업 (선택적)
try:
    from .strain_pls_numba import warmup_pls_numba
    _NUMBA_PLS_AVAILABLE = True
except ImportError:
    _NUMBA_PLS_AVAILABLE = False
    def warmup_pls_numba():
        pass

__all__ = [
    'StrainResult',
    'compute_strain_from_icgn',
    'PLSStrainResult',
    'compute_strain_pls',
    'compute_strain_pls_from_icgn',
    'get_vsg_size',
    'warmup_pls_numba',
    '_NUMBA_PLS_AVAILABLE',
]
