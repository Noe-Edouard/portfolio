import os
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from core.experiments.benchmarks.hessian import BenchmarkHessian
from core.experiments.benchmarks.enhancement import BenchmarkEnhancement
from core.experiments.benchmarks.base import BenchmarkBase
from core.experiments.analytics.runner import AnalyticsRunner
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import setup_logger
from core.config.benchmark import BenchmarkConfig
from core.config.experiment import ExperimentConfig
from core.config.setup import SetupConfig
from core.config.figure import FigureData
from core.utils.decorator import log_time, log_section, log_init
from configs.args import INPUT_DIR

class BenchmarkRunner():
    
    @log_init()
    def __init__(self, setup: SetupConfig):
        self.setup = setup
        self.plot_mode = self.setup.plot_mode
        self.save_mode = self.setup.save_mode
        self.logger = setup_logger(log_file=self.setup.log_file, debug_mode=self.setup.debug_mode)
        self.loader = Loader(self.setup.input_dir, self.logger)
        # self.parallelizer = Parallelizer(max_workers=4, use_processes=True)
        self.saver = Saver(experiment_name=self.setup.name, output_dir=self.setup.output_dir, logger=self.logger) if self.save_mode else None
        
        
        
        self.analytics = AnalyticsRunner()


    def _get_benchmark(self, benchmark_config: BenchmarkConfig) -> BenchmarkBase:
      
        if benchmark_config.mode == 'hessian':
            benchmark = BenchmarkHessian(
                save_mode=self.save_mode,
                plot_mode=self.plot_mode,
                logger=self.logger,
                loader=self.loader,
                saver=self.saver,
                params_grid=benchmark_config.params_grid
            )
        
        elif benchmark_config.mode == 'enhancement':
            benchmark = BenchmarkEnhancement(
                save_mode=self.save_mode,
                plot_mode=self.plot_mode,
                logger=self.logger,
                loader=self.loader,
                saver=self.saver,
            )
        
        else:
            raise ValueError(f'Benchmark mode unknown : {benchmark_config.mode}')
        
        return benchmark
            
    def _get_files(self, images_dir: str = "images", labels_dir: str = "labels"):
        images_dir = Path(f"{INPUT_DIR}/{self.setup.input_dir}") / images_dir
        labels_dir = Path(f"{INPUT_DIR}/{self.setup.input_dir}") / labels_dir
        images_files = []
        labels_files = []
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"The directories 'images' or 'labels' do not exist: {images_dir}, {labels_dir}")
        for image_name in os.listdir(images_dir):
            idx = image_name[len("image_"):]
            images_files.append(f"images/{image_name}")
            label_name = f"label_{idx}"
            labels_files.append(f"labels/{label_name}")
        return images_files, labels_files
    


    def _save_figures(self, figures: list[FigureData]):
        for i, figure in enumerate(figures):
            if figure.name is None:
                setattr(figure, 'name', f"figure_{i}")
                
        for figure in figures:
            if self.plot_mode:
                if figure.mode == 'text':
                    self.logger.info(figure.figure)
                plt.show()
            if self.save_mode:
                self.saver.save_figure(figure, 'overview')
                
                
    @log_time()
    @log_section("Runner execution")
    def run(self, 
            images_dir: str,
            labels_dir: str,
            benchmark_config: BenchmarkConfig,
            experiment_config: ExperimentConfig,
        ) -> str:
        
        # Get files
        raw_files, gt_files = self._get_files(
            images_dir=images_dir, 
            labels_dir=labels_dir,
        )
        
        # Select Benchmark
        benchmark = self._get_benchmark(benchmark_config)
        
        # results = self.parallelizer.run(
        #     func=benchmark.run,
        #     iterable=(
        #         (file_raw, file_gt, benchmark_config, experiment_config) 
        #         for file_raw, file_gt in zip(files_raw, files_gt)
        #     ),
        #     show_progress=False,
        #     unpack_args=True,
        # )
        
        # Sauvegarde les results avec pickle
        
        for raw_file, gt_file in zip(raw_files, gt_files):
            # Update_config
            exp_config = deepcopy(experiment_config)
            exp_config.loading.raw_file = raw_file
            exp_config.loading.gt_file = gt_file
            # Run benchmark
            benchmark.run(benchmark_config, exp_config)
        if self.save_mode:
            results_dir = self.saver.output_dir / 'results'
            return results_dir

        else:
            return
    
    
    @log_section("Runner analysis")
    def analyse(self, benchmark_config: BenchmarkConfig, results_dir: str):
        
        # Load results
        results = []
        for results_file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, results_file)
            results.append(self.loader.load_results(file_path))        
        
        # Get analysis
        match benchmark_config.mode:
            case 'hessian': 
                figures = self.analytics.get_hessian_figures(results, benchmark_config.params)
            case "enhancement": 
                figures = self.analytics.get_enhancement_figures(results, benchmark_config.params)

        self._save_figures(figures)
        
 