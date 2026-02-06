"""후처리 모듈"""

from .strain import (
    StrainResult,
    compute_strain_from_icgn,
    compute_strain_from_displacement_field
)

__all__ = [
    'StrainResult',
    'compute_strain_from_icgn',
    'compute_strain_from_displacement_field'
]