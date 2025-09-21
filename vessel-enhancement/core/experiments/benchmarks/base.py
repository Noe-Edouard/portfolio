import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from core.io.loader import Loader
from core.io.logger import Logger
from core.io.saver import Saver
from core.processing.processor import Processor
from core.config.figure import FigureData
from core.config.experiment import ExperimentConfig, Experiment, LoadingConfig
from core.config.benchmark import BenchmarkConfig, BenchmarkResults, BenchmarkData
from core.config.metrics import Metrics
from core.experiments.analytics.base import AnalyticsBase
from core.experiments.metrics import dice, mcc, roc, pr
from core.utils.decorator import log_section

class BenchmarkBase(ABC):

    def __init__(self, 
            save_mode: bool, 
            plot_mode: bool,
            logger: Logger, 
            loader: Loader,
            saver: Saver,
            analytics: Optional[AnalyticsBase] = None,
        ):
        self.save_mode = save_mode
        self.plot_mode = plot_mode
        self.loader = loader
        self.logger = logger
        self.saver = saver if save_mode else None
        self.analytics = analytics
        

    @abstractmethod
    def _update_config(self, config: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
        """Met à jour dans la config de l'expérience le paramètre étudié dans le benchmark"""
        pass
    
    
    @abstractmethod
    def _run_experiment(self, data_raw: ndarray,
            data_gt: ndarray,
            experiment_config: ExperimentConfig,
        ) -> Experiment:
        """Lance l'expérience avec les paramètre de la config"""
        pass
    
    
    @abstractmethod
    def _create_figures(self, benchmark_data: BenchmarkData) -> list[FigureData]:
        """Construit les figures utiles pour l'analyse du benchmark."""
        pass
    
    def load_data(self, loading_config: LoadingConfig):
        data_raw = self.loader.load_data(
            filename=loading_config.raw_file,
            normalize=loading_config.normalize,
            crop=loading_config.crop,
            target_shape=loading_config.target_shape,
        )
        data_gt = self.loader.load_data(
            filename=loading_config.gt_file,
            normalize=loading_config.normalize,
            crop=loading_config.crop,
            target_shape=loading_config.target_shape,
        )
        return data_raw, data_gt
    
    def _show_figures(self, figures: list[FigureData]):
        for figure in figures:
            if figure.mode == 'text':
                self.logger.info(f'{figure.figure}')
            else:
                plt.show()

    def _compute_metrics(self, data_segmented: ndarray, data_gt: ndarray) -> Metrics:
        
        metrics = Metrics(
            dice=dice(data_segmented, data_gt),
            mcc=mcc(data_segmented, data_gt),
            roc=roc(data_segmented, data_gt),
            pr=pr(data_segmented, data_gt),
        )
        
        return metrics
    
    @log_section('Benchmark execution')
    def run(self, 
            benchmark_config: BenchmarkConfig, 
            experiment_config: ExperimentConfig, 
        ) -> BenchmarkResults:
        
        
        # Load data
        data_raw, data_gt = self.load_data(experiment_config.loading)
        
        i = 0
        image_name = Path(experiment_config.loading.raw_file).stem
        results: BenchmarkResults = {param: {value: None for value in values} for param, values in benchmark_config.params.items()}
        for param, values in benchmark_config.params.items():
            for value in tqdm(values, desc=f"Processing {image_name} - {param:<12}"):
                
                # Update config
                exp_config = self._update_config(
                    config=deepcopy(experiment_config), 
                    param=param, 
                    value=value
                )
                
                # Run experiment 
                experiment = self._run_experiment(
                    data_raw=data_raw, 
                    data_gt=data_gt,
                    experiment_config=exp_config,
                )
                
                # Compute metrics
                metrics = self._compute_metrics(
                    data_segmented=experiment.data_segmented,
                    data_gt=data_gt,
                )
                
                # Store experiment
                experiment.metrics = metrics 
                experiment.id = f"{image_name}_{i}"
                results[param][value] = experiment
                
                # Udpate id
                i+=1

                
        benchmark_data = BenchmarkData(
            data_raw=data_raw,
            data_gt=data_gt,
            results=results,
        )
        
        if self.plot_mode or self.save_mode:
            figures = self._create_figures(benchmark_data)
        if self.plot_mode:
            self._show_figures()
        if self.save_mode:
            self.saver.save_results(results, image_name, 'results')
            for figure in figures:
                self.saver.save_figure(figure, image_name)

    

