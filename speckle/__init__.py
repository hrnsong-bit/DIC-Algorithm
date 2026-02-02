"""스페클 품질 분석 패키지"""

from .models import BadPoint, SSSIGResult, QualityReport, BatchReport
from .core import (
    SpeckleQualityAssessor, 
    compute_mig, 
    compute_sssig_map,
    create_specimen_mask,
    apply_mask_to_roi,
    get_mask_statistics,
    visualize_mask,
    visualize_mask_boundary
)
from .io import load_image, load_folder, save_image, get_image_files, ResultExporter
from .visualization import draw_poi_overlay, create_result_visualization, draw_roi
from .batch import BatchProcessor

__version__ = "3.2.0"

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
    'BatchProcessor'
]
