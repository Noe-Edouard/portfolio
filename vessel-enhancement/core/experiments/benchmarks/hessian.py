from numpy import ndarray
from typing import Any


from core.processing.processor import Processor
from core.experiments.benchmarks.base import BenchmarkBase
from core.experiments.analytics.hessian import AnalyticsHessian
from core.experiments.metrics import mcc
from core.config.experiment import ExperimentConfig, Experiment, ProcessingConfig, HessianConfig, EnhancementConfig, SegmentationConfig, MethodsConfig
from core.config.benchmark import BenchmarkData
from core.config.figure import FigureData
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import Logger
from core.io.saver import Saver
from core.utils.searcher import GridSearcher
from core.utils.helpers import crop_data



class BenchmarkHessian(BenchmarkBase):
    
    def __init__(self, 
            save_mode: bool, 
            plot_mode: bool,
            logger: Logger, 
            loader: Loader, 
            saver: Saver,
            params_grid: dict,
        ):
        
        super().__init__(save_mode, plot_mode, logger, loader, saver)
        self.analytics = AnalyticsHessian()
        self.grid_searcher = GridSearcher(
            params_grid=params_grid, 
            update_function=self.gs_update_function, 
            eval_function=self.gs_eval_func, 
            show_progress=False,
        )
     
    
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        setattr(config.methods, param, value)
        
        return config
    
    def gs_update_function(self, 
            combination: dict, 
            data_raw: ndarray, 
            data_gt: ndarray, 
            experiment_config: ExperimentConfig
        ):
    
        for param, value in combination.items():
            setattr(experiment_config.enhancement, param, value)
        
        # Parse config
        params = {
            "data_raw": data_raw,
            "data_gt": data_gt,
            "processing_config": experiment_config.processing,
            "hessian_config": experiment_config.hessian,
            "enhancement_config": experiment_config.enhancement,
            "segmentation_config": experiment_config.segmentation,
            "methods": experiment_config.methods,
        }
        return params

    def gs_eval_func(self, 
            data_raw: ndarray, 
            data_gt: ndarray, 
            processing_config: ProcessingConfig,
            hessian_config: HessianConfig,
            enhancement_config: EnhancementConfig,
            segmentation_config: SegmentationConfig,
            methods: MethodsConfig,
        ):
        processing_config = ProcessingConfig(
            use_gpu=processing_config.use_gpu,
            normalize=processing_config.normalize,
            parallelize=False,
        )
        processor = Processor(processing_config)
        data_enhanced, data_segmented, threshold = processor.process_data(
            data=data_raw, 
            ground_truth=data_gt, 
            hessian_config=hessian_config,
            enhancement_config=enhancement_config,
            segmentation_config=segmentation_config,
            methods=methods,
        )
        score = mcc(data_segmented, data_gt)
    
        return score
    
    def _run_experiment(self,
            data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        
        # Grid Search
        best_params, best_score = self.grid_searcher.fit(
            params={
                "data_raw": crop_data(data=data_raw, target_shape=(128, 128, 128)) 
                            if data_raw.ndim == 3 else data_raw,
                "data_gt": crop_data(data=data_gt, target_shape=(128, 128, 128)) 
                            if data_gt.ndim == 3 else data_gt,
                "experiment_config": experiment_config,
            }
        )
        
        # Update Config
        for key, value in best_params.items():
            setattr(experiment_config.enhancement, key, value)
        
        
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
    
    
    def _create_figures(self, benchmark_data: BenchmarkData) ->list[FigureData]:
        
        # Parse experiments
        experiments = []
        data_raw = benchmark_data.data_raw
        data_gt = benchmark_data.data_gt
        for params, values in benchmark_data.results.items():
            for value, experiment in values.items():
                experiments.append(experiment)
        
        figures: list[FigureData] = []
        
        # Histogram
        figures.append(self.analytics.get_histograms(
            experiments=experiments,
            data_raw=data_raw,
            data_gt=data_gt,
        ))
        
        # Config
        figures.append(self.analytics.get_configs(
            experiments=experiments,
        ))
                
        # Metrics
        figures.append(self.analytics.get_metrics(
            experiments=experiments,
        ))
        
        # Curves
        figures.append(self.analytics.get_curves(
            experiments=experiments,
            ground_truth=data_gt,
        ))
        
        # Views
        figures.extend(self.analytics.get_views(
            experiments=experiments,
            data_gt=data_gt,
            data_raw=data_raw,
        ))
        
        return figures
    
