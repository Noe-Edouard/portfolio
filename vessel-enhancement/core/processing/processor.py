import numpy as np
import dask.array as da
from numpy import ndarray
from math import ceil
from dask.diagnostics import ProgressBar
from typing import Callable, Optional, Tuple

from core.utils.helpers import normalize_data
from core.utils.gpu import is_gpu_available_available
from core.config.experiment import HessianConfig, EnhancementConfig, ProcessingConfig, MethodsConfig, SegmentationConfig
from core.processing.derivator import Derivator
from core.processing.enhancer import Enhancer 
from core.processing.segmenter import Segmenter

class Processor:
    
    def __init__(self, config: ProcessingConfig):
        self.use_gpu = config.use_gpu and is_gpu_available_available()
        self.normalize = config.normalize
        self.parallelize = config.parallelize
        self.show_progress = config.show_progress
        self.overlap_size = config.overlap_size
        self.chunk_size = config.chunk_size
        
        self.derivator = Derivator(use_gpu=self.use_gpu)
        self.enhancer = Enhancer(use_gpu=self.use_gpu)
        self.segmenter = Segmenter()

    def enhance_data(self,
            data: ndarray,
            enhancement_function: Callable,
            enhancement_params: dict,
        ):
        
        # Parallel (3D)
        if self.parallelize and data.ndim == 3:
            chunk_size = self.chunk_size or tuple(ceil(s//2) for s in data.shape)
            overlap_size = self.overlap_size or max(enhancement_params.get('scales', [10]))
            
            # Parallelize 
            data_dask = da.from_array(data, chunks=chunk_size)

            processed_chunks = da.map_overlap(
                enhancement_function,
                data_dask,
                depth=overlap_size,
                boundary='reflect',
                dtype=np.float32,
                **enhancement_params,
            )

            if self.show_progress:
                with ProgressBar():
                    data_enhanced = processed_chunks.compute()
            else:
                data_enhanced = processed_chunks.compute()
                
        else: # Sequential (2D)
            data_enhanced = enhancement_function(data, **enhancement_params)
        
        # Normalization
        if self.normalize:
            data_enhanced = normalize_data(data_enhanced)
            
        return data_enhanced
        
    def process_data(self,
            data: ndarray,
            hessian_config: HessianConfig,
            enhancement_config: EnhancementConfig,
            segmentation_config: SegmentationConfig,
            methods: MethodsConfig,
            ground_truth: Optional[ndarray] = None,
        ) -> Tuple[ndarray, ndarray, float]:
            
            # Get functions
            hessian_function = self.derivator.select_hessian_function(methods.derivator)
            enhancement_function = self.enhancer.select_enhancement_function(methods.enhancer)
            segmentation_function = self.segmenter.select_segmentation_function(methods.segmenter)
            
            # Update configs
            hessian_params = hessian_config.to_dict()
            enhancement_config.hessian_function = hessian_function
            enhancement_config.hessian_params = hessian_params
            enhancement_params = enhancement_config.to_dict()
            segmentation_params = segmentation_config.to_dict()
            
            # Enhancement
            data_enhanced = self.enhance_data(
                data=data, 
                enhancement_function=enhancement_function, 
                enhancement_params=enhancement_params, 
                )
            
            # Segmentation
            data_segmented, threshold = segmentation_function(
                data=data_enhanced,
                ground_truth=ground_truth,
                **segmentation_params
            )

            return data_enhanced, data_segmented, threshold
