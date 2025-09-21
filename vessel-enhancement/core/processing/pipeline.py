from numpy import ndarray


from core.config.setup import SetupConfig
from core.config.experiment import Experiment, ExperimentConfig, ProcessingConfig, LoadingConfig
from core.processing.processor import Processor
from core.io.saver import Saver
from core.io.loader import Loader
from core.io.logger import setup_logger
from core.utils.viewer import Viewer
from core.utils.decorator import log_section, log_time, log_init

class Pipeline:
    
    @log_init()
    def __init__(self, setup: SetupConfig, processing_config: ProcessingConfig):
        
        # Modes
        self.save_mode = setup.save_mode
        self.plot_mode = setup.plot_mode

        # Logger
        self.logger = setup_logger(log_file=setup.log_file, debug_mode=setup.debug_mode)

        # Loader
        self.loader = Loader(input_dir=setup.input_dir, logger=self.logger) 

        # Viewer
        self.viewer = Viewer()
        
        # Saver
        self.saver = Saver(experiment_name=setup.name, output_dir=setup.output_dir, logger=self.logger) if self.save_mode else None
       
        
    def load_data(self, loading_config: LoadingConfig):
            data_raw = self.loader.load_data(
                filename=loading_config.raw_file,
                normalize=loading_config.normalize,
                crop=loading_config.crop,
                target_shape=loading_config.target_shape,
            )
            if loading_config.gt_file is not None:
                data_gt = self.loader.load_data(
                    filename=loading_config.gt_file,
                    normalize=loading_config.normalize,
                    crop=loading_config.crop,
                    target_shape=loading_config.target_shape,
                )
            else :
                data_gt = None
                
            return data_raw, data_gt
        
        
    def plot_results(self, data_raw: ndarray, data_enhanced: ndarray, data_segmented: ndarray, threshold: float = None):
        if data_raw.ndim == 2:
            figure = self.viewer.display_images([data_raw, data_enhanced, data_segmented], ["RAW", "ENHANCED", "SEGMENTED"])
            if self.save_mode:
                self.saver.save_plot(figure, filename='results')
        else:
            histogram = self.viewer.display_histograms([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
            slices = self.viewer.display_slices([data_raw, data_enhanced, data_segmented], ['RAW', 'ENHANCED', 'SEGMENTED'])
            volume = self.viewer.display_volume(volume=data_enhanced, threshold=threshold)
            if self.save_mode:
                self.saver.save_plot(histogram, 'histogram')
                self.saver.save_anim(slices, 'slices')
                self.saver.save_plot(volume, 'volume')
           
    def save_results(self, data_enhanced: ndarray, data_segmented: ndarray, experiment_config: ExperimentConfig):     
        if self.save_mode:
            self.saver.save_data(data_enhanced, f'data_enhanced')
            self.saver.save_data(data_segmented, f'data_segmented')
            self.saver.save_config(experiment_config, f'experiment_config')
    
    @log_section("Pipeline execution")
    @log_time()
    def run(self, 
            experiment_config: ExperimentConfig, 
        ) -> Experiment:
        
        # Parse config
        loading_config = experiment_config.loading
        processing_config = experiment_config.processing
        hessian_config = experiment_config.hessian
        enhancement_config = experiment_config.enhancement
        segmentation_config = experiment_config.segmentation
        methods = experiment_config.methods
        
        # Load Data
        data_raw = self.load_data(loading_config)
            
        # Process_data
        processor = Processor(processing_config)
        data_enhanced, data_segmented, threshold = processor.process_data(
            data=data_raw,
            hessian_config=hessian_config,
            enhancement_config=enhancement_config,
            segmentation_config=segmentation_config,
            methods=methods
        )
        
        # Save results
        if self.save_mode:
            self.save_results(data_enhanced, data_segmented, experiment_config)

        # Plot results
        if self.plot_mode:
            self.plot_results(data_raw, data_enhanced, data_segmented, threshold)
        
        # Return results
        experiment_data = Experiment(
            config=experiment_config,
            data_enhanced=data_enhanced,
            data_segmented=data_segmented,
        )
            
        return experiment_data