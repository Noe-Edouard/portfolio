import numpy as np
from numpy import ndarray
from typing import Any

from core.processing.processor import Processor
from core.experiments.benchmarks.base import BenchmarkBase
from core.experiments.analytics.enhancement import AnalyticsEnhancement
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import Logger
from core.config.figure import FigureData
from core.config.benchmark import BenchmarkData
from core.config.experiment import ExperimentConfig, Experiment

class BenchmarkEnhancement(BenchmarkBase):
    
    def __init__(self, 
            save_mode: bool, 
            plot_mode: bool,
            logger: Logger, 
            loader: Loader, 
            saver: Saver,
        ):
        super().__init__(save_mode, plot_mode, logger, loader, saver)
        self.analytics = AnalyticsEnhancement()
    
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        if param == 'scales_min':
            scales = np.arange(value, 20, 1)
            setattr(config.enhancement, 'scales', scales)
        elif param == 'scales_max':
            scales = np.arange(1, value, 1)
            setattr(config.enhancement, 'scales', scales)
        else:    
            setattr(config.enhancement, param, value)

        return config
    
    
    def _run_experiment(self,
            data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        
        # Process data
        processor = Processor(experiment_config.processing)
        data_enhanced, data_segmented, threshold = processor.process_data(
            data=data_raw,
            hessian_config=experiment_config.hessian,
            enhancement_config=experiment_config.enhancement,
            segmentation_config=experiment_config.segmentation,
            methods=experiment_config.methods,
            ground_truth=data_gt,
        )
        experiment_config.segmentation.threshold = threshold
        
        # Return experiment
        experiment = Experiment(
            config=experiment_config,
            data_enhanced=data_enhanced,
            data_segmented=data_segmented,
        )
        
        return experiment
    
    
    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        
        figures = []
        
        # Alpha/Beta/Gamma
        figures.append(self.analytics.get_params_curves(
            benchmark_results=benchmark_data.results
        ))
        
        # Scales
        figures.append(self.analytics.get_scales_curves(
            benchmark_results=benchmark_data.results
        ))
        
        return figures
    
   
