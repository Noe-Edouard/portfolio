from typing import Any, Literal, Optional
from numpy import ndarray
from dataclasses import dataclass

from core.config.base import ConfigBase
from core.config.experiment import Experiment
from core.config.setup import SetupConfig


### BENCHMARK 

BenchmarkResults = dict[str, dict[str, Experiment]] # {param: {value: Experiment}}
RunnerResultsParsed = dict[str, dict[str, dict[Any, list[float]]]] # {param: {value: {metric: [float for images]}}}

@dataclass
class BenchmarkConfig(ConfigBase): 
    mode: Literal['hessian', 'enhancement']
    results_dir: str
    params: dict[str, list] 
    params_grid: Optional[dict[str, Any]]
   
@dataclass
class BenchmarkData(ConfigBase):
    data_raw: ndarray
    data_gt: ndarray
    results: BenchmarkResults

@dataclass
class RunnerConfig(ConfigBase):
    setup: SetupConfig
    images_dir: str
    labels_dir: str
 
   
