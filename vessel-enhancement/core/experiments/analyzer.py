import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from math import ceil

from core.utils.helpers import compute_time
from core.utils.decorator import log_time, log_section, log_init
from core.utils.viewer import Viewer 
from core.io.logger import setup_logger
from core.io.loader import Loader
from core.io.saver import Saver
from core.config.setup import SetupConfig
from core.config.experiment import ProcessingConfig, EnhancementConfig
from core.processing.enhancer import Enhancer
from core.processing.derivator import Derivator
from core.processing.processor import Processor

logger = setup_logger(log_file='benchmark', debug_mode=True)

class Analyzer:
    
    @log_init()
    def __init__(self, setup: SetupConfig):

        self.save_mode = setup.save_mode
        self.plot_mode = setup.plot_mode
        
        self.logger = setup_logger(log_file=setup.log_file, debug_mode=setup.debug_mode)
        self.loader = Loader(setup.input_dir, self.logger)
        self.viewer = Viewer()
        self.saver = Saver(experiment_name=setup.name, output_dir=setup.output_dir, logger=self.logger)
        

    @log_time()
    @log_section("Chunk analysis")
    def chunk_analysis(self, 
            volume_sizes: list[int] = [64, 128, 256, 512], 
            chunk_ratios: list[int] = 100**np.linspace(0.62, 0.88, 40) # log large scale btw 15 and 50
        ):
        
        # Processor
        enhancer = Enhancer(use_gpu=False)
        derivator = Derivator(use_gpu=False)
        processor = Processor(
            ProcessingConfig(
                use_gpu=False,
                normalize=True,
                parallelize=True,
                show_progress=True,
                overlap_size=10,
                chunk_size=None,
            )      
        )
        
        # Enhancement config
        enhancement_params = EnhancementConfig(
            scales=[2, 4, 6, 8, 10],
            alpha=0.5,
            beta=0.5,
            skimage=False,
            hessian_function=derivator.farid,
            hessian_params={'mode': 'reflect', 'cval': 0.0}
        ).to_dict()
        
        
        all_times = []
        
        for volume_size in volume_sizes:
            volume = np.ones((volume_size, volume_size, volume_size), np.float32)
            times = []
            for ratio in chunk_ratios:
                size = int(volume_size*ratio/100)
                processor.chunk_size = (size, size, size)
                
                # Parallel processing
                times.append(compute_time(
                    processor.enhance_data,
                    data=volume, 
                    enhancement_function=enhancer.frangi,
                    enhancement_params=enhancement_params,
                ))
                
            all_times.append(times)
        
        # Plot times
        fig = plt.figure(figsize=(10, 8))
        colors = [
            # "#022c7aff",
            # "#0743b1ff",
            "#175ddfff",
            "#407ff5ff",
            "#6097fcff",
        ]
        for i, volume_size in enumerate(volume_sizes):
            plt.plot(chunk_ratios, all_times[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
        X = 100**np.linspace(0.611, 0.89, 1000)
        def f(x):
            from math import floor
            if x <= 0:
                return 0  # On définit f(x)=0 pour x<=0
            # Trouver le n correspondant au segment
            n = floor(100 / x)
            x_start = 100 / (n + 1)
            x_end = 100 / n
            # Fonction linéaire 0->1 sur le segment
            y = (x - x_start) / (x_end - x_start) * 10
            return y
        plt.plot(X, [f(x) for x in X], color='black', linestyle='--', label=r'$f(x) = \frac{x - 100/(n+1)}{100/n - 100/(n+1)}$')

        plt.xlim([15, 62])
        plt.title(f"Influence de la taille des chunks sur le temps d'éxécution")
        plt.xlabel("Chunk ratio (% du volume)")
        plt.ylabel("Temps d'exécution (s)")
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()

        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, 'chunk_analysis_1')

        # Plot times
        fig = plt.figure(figsize=(10, 8))
        colors = [
            # "#022c7aff",
            # "#0743b1ff",
            "#175ddfff",
            "#407ff5ff",
            "#6097fcff",
        ]
        for i, volume_size in enumerate(volume_sizes):
            plt.plot(chunk_ratios, all_times[i], '-+', color=colors[i], label=f"volume size: {volume_size}")
        
        plt.xlim([15, 62])
        plt.title(f"Influence de la taille des chunks sur le temps d'éxécution")
        plt.xlabel("Chunk ratio (% du volume)")
        plt.ylabel("Temps d'exécution (s)")
        plt.legend()
        plt.grid(True)
        
        
        plt.tight_layout()

        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, 'chunk_analysis_2')
        


    @log_time()
    @log_section("Parallelization check")
    def para_check(self, input_file: str):
        
        volume = self.loader.load_data(input_file)
        size = volume.shape   
        chunk_size = (ceil(size[0]/2), ceil(size[1]/2), ceil(size[2]/2))

        # Processor
        enhancer = Enhancer(use_gpu=False)
        derivator = Derivator(use_gpu=False)
        processor = Processor(
            ProcessingConfig(
                use_gpu=False,
                normalize=True,
                parallelize=True,
                show_progress=True,
                overlap_size=10,
                chunk_size=chunk_size,
            )      
        )
        
        # Enhancement config
        enhancement_params = EnhancementConfig(
            scales=[2, 4, 6, 8, 10],
            alpha=0.5,
            beta=0.5,
            gamma=20,
            skimage=False,
            hessian_function=derivator.farid,
            hessian_params={'mode': 'reflect', 'cval': 0.0}
        ).to_dict()

        # Sequential processing
        processor.parallelize = False
        sequential = processor.enhance_data(
            data=volume, 
            enhancement_function=enhancer.frangi,
            enhancement_params=enhancement_params,
        )
        
        # Parallel processing
        processor.parallelize = True
        parallel = processor.enhance_data(
            data=volume, 
            enhancement_function=enhancer.frangi,
            enhancement_params=enhancement_params,
        )
            
        fig = self.viewer.display_slices([sequential, parallel, sequential-parallel], ['Sequantial', 'Parallel', 'Difference'])
        self.saver.save_anim(fig, 'para_check')
        plt.close()

        max_absolute_error = np.abs(sequential - parallel).max()
        mean_absolute_error = np.abs(sequential - parallel).mean()
        logger.info(f"Mean Absolute Error (sequential vs parallel): {mean_absolute_error:.4e}")
        logger.info(f"Mean Absolute Error (sequential vs parallel): {max_absolute_error:.4e}")



    @log_time()
    @log_section("PARA vs SEQ")
    def para_vs_seq(self, volume_sizes: list[int] = [32, 64, 128, 256, 512]):
        
        # Processor
        enhancer = Enhancer(use_gpu=False)
        derivator = Derivator(use_gpu=False)
        processor = Processor(
            ProcessingConfig(
                use_gpu=False,
                normalize=True,
                parallelize=False,
                show_progress=True,
                overlap_size=10,
                chunk_size=(64, 64, 64),
            )      
        )
        
        # Enhancement config
        enhancement_params = EnhancementConfig(
            scales=[2, 4, 6, 8, 10],
            alpha=0.5,
            beta=0.5,
            skimage=False,
            hessian_function=derivator.farid,
            hessian_params={'mode': 'reflect', 'cval': 0.0}
        ).to_dict()
        
        times_sequential = []
        times_parallel = []
        
        for size in volume_sizes:
            
            volume = np.ones((size, size, size), np.float32)
            chunk_size = (ceil(size/2), ceil(size/2), ceil(size/2))
            processor.chunk_size = chunk_size
            
            # Sequential processing
            processor.parallelize = False
            times_sequential.append(compute_time(
                processor.enhance_data,
                data=volume, 
                enhancement_function=enhancer.frangi,
                enhancement_params=enhancement_params,
            ))
            
            # Parallel processing
            processor.parallelize = True
            times_parallel.append(compute_time(
                processor.enhance_data,
                data=volume, 
                enhancement_function=enhancer.frangi,
                enhancement_params=enhancement_params,
            ))
        
        # Resume
        logger.info('='*30+' RESUME '+'='*30+'\n')
        logger.info(f'Volume sizes:     {volume_sizes}')
        logger.info(f'Times sequential: {times_sequential}')
        logger.info(f'Times parallel:   {times_parallel}')
        
        headers = ['Volume size (px)', 'Time sequential (s)', 'Time parallel (s)']
        rows = list(zip(volume_sizes, times_sequential, times_parallel))
        logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
        print('='*68+'\n')
        
        # Linear regression (log scale)
        a, b = np.polyfit(np.log(volume_sizes), np.log(times_sequential), 1)
        c, d = np.polyfit(np.log(volume_sizes), np.log(times_parallel), 1)
        
        # Plot 
        fig = plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(volume_sizes, times_sequential, '+-', label="Séquentiel", color='red')
        plt.plot(volume_sizes, times_parallel, '+-',  label="Parallèle", color='dodgerblue')
        plt.xlabel("Taille du volume (voxels)")
        plt.ylabel("Temps (secondes)")
        plt.title("Temps de traitement PARA vs SEQ")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(np.log(volume_sizes), np.log(times_sequential), '+', label='real', color='limegreen')
        plt.plot(np.log(volume_sizes), a * np.log(volume_sizes) + b, '--', label=f'fit: y = {a:.2f} x + {b:.2f}', color='red')
        plt.xlabel('log(Taille du volume)')
        plt.ylabel('log(Temps d\'exécution)')
        plt.title('Traitement séquentiel')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(np.log(volume_sizes), np.log(times_parallel), '+', label='real', color='limegreen')
        plt.plot(np.log(volume_sizes), c * np.log(volume_sizes) + d, '--', label=f'fit: y = {a:.2f} x + {d:.2f}', color='dodgerblue')
        plt.xlabel('log(Taille du volume)')
        plt.ylabel('log(Temps d\'exécution)')
        plt.title('Traitement parallèle')
        plt.legend()
        plt.grid(True)

        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, 'para_vs_seq')
                    
        return volume_sizes, times_sequential, times_parallel
    
    
    @log_time()
    @log_section("CPU vs GPU")
    def cpu_vs_gpu(self, volume_sizes: list[int] = [16, 32, 64, 128, 256]):
        
        # Processor
        processor = Processor(
            ProcessingConfig(
                use_gpu=False,
                normalize=True,
                parallelize=False,
                show_progress=True,
                overlap_size=10,
                chunk_size=(64, 64, 64),
            )      
        )
        
        # Enhancement config
        
        enhancement_params = EnhancementConfig(
            scales=[2, 4, 6, 8, 10],
            alpha=0.5,
            beta=0.5,
            skimage=False,
            hessian_params={'mode': 'reflect', 'cval': 0.0}
        ).to_dict()
        
        times_cpu = []
        times_gpu = []
        
        for size in volume_sizes:
            
            volume = np.ones((size, size, size), np.float32)
            chunk_size = (ceil(size/2), ceil(size/2), ceil(size/2))
            processor.chunk_size = chunk_size
            
            # CPU processing
            enhancer = Enhancer(use_gpu=False)
            derivator = Derivator(use_gpu=False)
            enhancement_params['hessian_function'] = derivator.farid
            processor.use_gpu = False
            times_cpu.append(compute_time(
                processor.enhance_data,
                volume, 
                enhancer.frangi,
                enhancement_params,
            ))
            
            # GPU processing
            enhancer = Enhancer(use_gpu=True)
            derivator = Derivator(use_gpu=True)
            enhancement_params['hessian_function'] = derivator.farid
            processor.use_gpu = True
            times_gpu.append(compute_time(
                processor.enhance_data,
                volume, 
                enhancer.frangi,
                enhancement_params,
            ))
        
        # Resume
        logger.info('='*30+' RESUME '+'='*30+'\n')
        logger.info(f'Volume sizes:     {volume_sizes}')
        logger.info(f'Times cpu: {times_cpu}')
        logger.info(f'Times gpu: {times_gpu}')
        
        headers = ['Volume size (px)', 'Time CPU (s)', 'Time GPU (s)']
        rows = list(zip(volume_sizes, times_cpu, times_gpu))
        logger.info('\n' + tabulate(rows, headers=headers, tablefmt='github', floatfmt='>.3f', intfmt='^'))
        print('='*68+'\n')
        
        # Plot 
        fig = plt.figure(figsize=(16, 4))
        plt.plot(volume_sizes, times_cpu, '+-', label="CPU", color='red')
        plt.plot(volume_sizes, times_gpu, '+-',  label="GPU", color='dodgerblue')
        plt.xlabel("Taille du volume (voxels)")
        plt.ylabel("Temps (secondes)")
        plt.title("Comparaison des temps de traitement CPU vs GPU")
        plt.legend()
        plt.grid(True)
        
       
        if self.plot_mode:
            plt.show()
        if self.save_mode:
            self.saver.save_plot(fig, 'cpu_vs_gpu')

    
    
    @log_time()
    @log_section("Analyzer run")
    def run(self):

        # chunk_analysis_params = {
        #     'volume_sizes': [256],
        #     'chunk_ratios': [60],
        #     # 'chunk_ratios': 100**np.linspace(0.62, 0.88, 50),
        # }
        # para_check_params = {'input_file': 'test.nii'}
        # para_vs_seq_params = {'volume_sizes': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]}
        # cpu_vs_gpu_params = {'volume_sizes': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]}
        cpu_vs_gpu_params = {'volume_sizes': [50, 60]}

        # self.chunk_analysis(**chunk_analysis_params)
        # self.para_check(**para_check_params)
        # self.para_vs_seq(**para_vs_seq_params)
        self.cpu_vs_gpu(**cpu_vs_gpu_params)
        
        