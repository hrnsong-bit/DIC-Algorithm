"""스페클 품질 분석 패키지"""

from .models import BadPoint, SSSIGResult, QualityReport, BatchReport
from .core import (
    SpeckleQualityAssessor,
    compute_mig,
    compute_sssig_map,
    compute_gradient,
    create_specimen_mask,
    apply_mask_to_roi,
    get_mask_statistics,
    visualize_mask,
    visualize_mask_boundary,
    estimate_noise_variance,
    estimate_noise_from_pair,
    calculate_sssig_threshold,
    predict_displacement_accuracy,
)
from .io import load_image, load_folder, save_image, get_image_files, ResultExporter
from .visualization import draw_poi_overlay, create_result_visualization, draw_roi
from .batch import BatchProcessor

__version__ = "3.3.0"

__all__ = [
    # Models
    'BadPoint',
    'SSSIGResult',
    'QualityReport',
    'BatchReport',

    # Core
    'SpeckleQualityAssessor',
    'compute_mig',
    'compute_sssig_map',
    'compute_gradient',

    # Noise
    'estimate_noise_variance',
    'estimate_noise_from_pair',
    'calculate_sssig_threshold',
    'predict_displacement_accuracy',

    # Masking
    'create_specimen_mask',
    'apply_mask_to_roi',
    'get_mask_statistics',
    'visualize_mask',
    'visualize_mask_boundary',

    # IO
    'load_image',
    'load_folder',
    'save_image',
    'get_image_files',
    'ResultExporter',

    # Visualization
    'draw_poi_overlay',
    'create_result_visualization',
    'draw_roi',

    # Batch
    'BatchProcessor',
]
