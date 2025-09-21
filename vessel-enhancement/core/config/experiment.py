from numpy import ndarray
from typing import Literal, Callable, Sequence, Tuple, Optional
from dataclasses import dataclass

from core.config.base import ConfigBase
from core.config.metrics import Metrics


### CONFIGS

@dataclass
class LoadingConfig(ConfigBase):
    normalize: bool
    crop: bool
    target_shape: Sequence[int]
    raw_file: str
    gt_file: str

@dataclass
class MethodsConfig(ConfigBase):
    derivator: Literal['default', 'gaussian', 'farid', 'cubic', 'trigonometric', 'catmull', 'bspline', 'bezier']
    enhancer: Literal['frangi']
    segmenter: Literal['thresholding']

@dataclass
class HessianConfig(ConfigBase):
    mode: Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']
    cval: float
   
@dataclass
class ProcessingConfig(ConfigBase):
    use_gpu: bool = True,
    normalize: bool = True,
    parallelize: bool = False,
    show_progress: bool = False,
    overlap_size: Optional[int] = 10,
    chunk_size: Optional[Sequence[int]] = None,
    
@dataclass
class EnhancementConfig(ConfigBase):
    black_ridges: bool
    scales: Sequence[int]
    alpha: float
    beta: float
    gamma: Optional[float] = None
    skimage: Optional[bool] = False
    hessian_function: Optional[Callable[..., list[ndarray]]] = None
    hessian_params: Optional[dict] = None

@dataclass
class SegmentationConfig(ConfigBase):
    threshold: float
   
   
@dataclass
class ExperimentConfig(ConfigBase):
    loading: LoadingConfig
    methods: MethodsConfig
    processing: ProcessingConfig
    hessian: HessianConfig
    enhancement: EnhancementConfig
    segmentation: SegmentationConfig


### EXPERIMENT
 
@dataclass
class Experiment(ConfigBase):
    data_enhanced: ndarray
    data_segmented: ndarray
    config: ExperimentConfig
    metrics: Optional[Metrics] = None
    id: Optional[str] = None
    