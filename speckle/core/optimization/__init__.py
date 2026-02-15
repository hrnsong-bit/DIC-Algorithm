# speckle/core/optimization/__init__.py

from .icgn import compute_icgn
from .results import ICGNResult
from .interpolation import create_interpolator, ImageInterpolator
from .shape_function import (
    warp,
    compute_steepest_descent,
    update_warp_inverse_compositional,
    get_initial_params,
    get_num_params
)

# Numba 가속 모듈 (선택적 — 없으면 무시)
try:
    from .icgn_core_numba import warmup_icgn_core
    from .interpolation_numba import warmup_numba_interp
    from .shape_function_numba import warmup_numba_shape
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

__all__ = [
    'compute_icgn',
    'ICGNResult',
    'create_interpolator',
    'ImageInterpolator',
    'warp',
    'compute_steepest_descent',
    'update_warp_inverse_compositional',
    'get_initial_params',
    'get_num_params',
    # Numba
    'warmup_icgn_core',
    'warmup_numba_interp',
    'warmup_numba_shape',
    '_NUMBA_AVAILABLE',
]
