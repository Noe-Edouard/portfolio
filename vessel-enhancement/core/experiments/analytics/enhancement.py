import matplotlib.pyplot as plt

from core.experiments.analytics.base import AnalyticsBase
from core.config.benchmark import BenchmarkResults
from core.config.figure import FigureData


class AnalyticsEnhancement(AnalyticsBase):
    
    def get_params_curves(self, benchmark_results: BenchmarkResults) -> FigureData:
        
        fig = plt.figure(figsize=(16, 6))
        colors = ['red', 'dodgerblue', 'limegreen']
        params = ['alpha', 'beta', 'gamma']
        for i, (color, param) in enumerate(zip(colors, params)):
            values = []
            scores = []
            for value, experiment in benchmark_results[param].items():
                values.append(value)
                scores.append(experiment.metrics.mcc)
            plt.subplot(1, len(params), i + 1)
            plt.plot(values, scores, '+-', color=color)
            plt.xscale('log')
            plt.xlabel(param)
            plt.ylabel("MCC Score")
            plt.title(f"Influence du paramètre {param}")
            plt.grid(True)

        plt.tight_layout()
        figure = self._create_figure(fig, 'params', 'plot')
        plt.close(fig)
        
        return figure
    
    def get_scales_curves(self, benchmark_results: BenchmarkResults) -> FigureData:
        
        fig = plt.figure(figsize=(16, 6))
        colors = ['orangered', 'magenta']
        params = ['scales_min', 'scales_max']
        
        for i, (color, param) in enumerate(zip(colors, params)):
            values = []
            scores = []
            for value, experiment in benchmark_results[param].items():
                values.append(value)
                scores.append(experiment.metrics.mcc)
                
            plt.subplot(1, len(params), i + 1)
            plt.plot(values, scores, '+-', color=color)
            plt.xlabel(param)
            plt.ylabel("MCC Score")
            plt.grid(True)
            if param == 'scales_min':
                plt.title(f"Influence de l'échelle min (max_scale=20)")
            elif param == 'scales_max':
                plt.title(f"Influence de l'échelle max (min_scale=1)")

        plt.tight_layout()
        figure = self._create_figure(fig, 'scales', 'plot')
        plt.close(fig)
        
        return figure
    
 
