import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import fields
from tabulate import tabulate

from core.experiments.analytics.base import AnalyticsBase
from core.config.benchmark import BenchmarkResults, RunnerResultsParsed
from core.config.figure import FigureData
from core.config.metrics import Metrics


class AnalyticsRunner(AnalyticsBase):
    
    def _parse_results(self, results: list[BenchmarkResults], params: dict[str, list[float]]) -> RunnerResultsParsed:
        metrics = [field.name for field in fields(Metrics)]
        
        # Parse [param][value][metric] = list[scores]
        benchmark_data = {
            param: {
                value: {metric: [] for metric in metrics}
                for value in values
            }
            for param, values in params.items()
        }

        for image_result in results:
            for param, values in image_result.items():
                for value, experiment in values.items():
                    for metric in metrics:
                        benchmark_data[param][value][metric].append(getattr(experiment.metrics, metric))

        return benchmark_data
    
    
    def get_hessian_figures(self, results_raw: list[BenchmarkResults], params: dict[str, list]) -> list[FigureData]:
        
        # Parse results for analysis
        results = self._parse_results(results_raw, params)
        
        # Compute mean/std/all_scores/best_experiment
        metrics = [metric.name for metric in fields(Metrics)]
        methods = [method for method in results['derivator'].keys()]
        
        mean = {method: {metric: 0.0 for metric in metrics} for method in methods}
        std = {method: {metric: 0.0 for metric in metrics} for method in methods}
        all_scores = {metric: [] for metric in metrics}
        
        for method in methods:
            for metric in metrics:
                scores = results['derivator'][method][metric]
                mean[method][metric] = np.mean(scores)
                std[method][metric] = np.std(scores)
                all_scores[metric].extend(scores)

          
        figures = []  
              
        # Table (mean+std)
        rows = []
        for method in methods:
            rows.append({
                'method': method,
                **{metric: f'{mean[method][metric]:.3f} ± {std[method][metric]:.3f}' for metric in metrics}
            })
            
        best_row = {'method': 'best'}
        for metric in metrics:
            best_method = max(methods, key=lambda m: mean[m][metric])
            best_row[metric] = best_method
        rows.append(best_row)

        table = tabulate(rows, headers="keys", tablefmt='github')
        content = f"\n BENCHMARK HESSIAN - METRICS TABLE (mean)\n" + table
        figures.append(self._create_figure(content, 'table', 'text'))


        # Box plot
        for metric in metrics:
            data = [results['derivator'][method][metric] for method in methods]
            colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(methods)))
            # colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(methods)))
            
            fig = plt.figure(figsize=(10, 8))
            bp = plt.boxplot(data, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors): # Box color
                patch.set_facecolor(color)
            
            for median in bp['medians']: # Median color
                median.set_color('black')
                median.set_linewidth(2)

            plt.xticks(ticks=range(1, len(methods) + 1), labels=methods)
            plt.title(f'Distribution de {metric}')
            plt.ylabel(metric)
            plt.grid(True, axis='y')

            figures.append(self._create_figure(fig, f'box_{metric}', 'plot'))
            plt.close(fig)
            



        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(methods)))

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)

        for method, color in zip(methods, colors):
            values = [mean[method][metric] for metric in metrics]
            values += values[:1]
            ax.plot(angles, values, label=method, color=color)
            ax.fill(angles, values, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.title('Radar métriques moyennes')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        figures.append(self._create_figure(fig, 'radar', 'plot'))
        plt.close(fig)
        
        
        # Best/Worst (mcc)

        images_scores = []
        images_files = []
        for image_result in results_raw:
            mcc_scores = []
            for method, experiment in image_result['derivator'].items():
                mcc_scores.append(getattr(experiment.metrics, 'mcc'))
            images_scores.append(np.mean(mcc_scores))
            images_files.append(Path(experiment.config.loading.raw_file).stem)

        best_idx = int(np.argmax(images_scores))
        worst_idx = int(np.argmin(images_scores))

        best_img_name = images_files[best_idx]
        worst_img_name = images_files[worst_idx]

        content = f"BEST RESULTS\n\n"
        content += f"Best image: {best_img_name} (mean={images_scores[best_idx]:.4f})\n"
        content += f"Worst image: {worst_img_name} (mean={images_scores[worst_idx]:.4f})\n"
        figures.append(self._create_figure(content, 'best_worst', 'text'))
        

        return figures
    
    
    
        
    def get_enhancement_figures(self, results_raw: list[BenchmarkResults], params: dict[str, list]) -> list[FigureData]:
        
        # Parse results for analysis
        results = self._parse_results(results_raw, params)
        
        # Compute scores
        figures = []
        params = ['alpha', 'beta', 'gamma', 'scales_min', 'scales_max']
        colors = ['red', 'dodgerblue', 'limegreen', 'orangered', 'magenta']
        
        for color, param in zip(colors, params):
            values = []
            mean = []
            std = []
            
            # Scores per image
            param_scores = results[param]  # dict[value] -> {'mcc': list[float]}
            scores_per_value = {v: metrics['mcc'] for v, metrics in param_scores.items()}
            values_sorted = sorted(scores_per_value.keys())
            num_images = len(next(iter(scores_per_value.values())))  # nombre d'images
        
            for value, metrics in results[param].items():
                scores = metrics['mcc']
                mean.append(np.mean(scores))
                std.append(np.std(scores))
                values.append(value)
            
            best_counts = {v: 0 for v in values_sorted}

            for i in range(num_images):
                best_value = max(values_sorted, key=lambda v: scores_per_value[v][i])
                best_counts[best_value] += 1
            
            mean = np.array(mean)
            std = np.array(std)
            
            fig = plt.figure(figsize=(14, 6))
            
            # MEAN + STD
            plt.subplot(1, 2, 1)
            plt.plot(values, mean, '+-', label=f"mean", color=color)
            plt.fill_between(values, mean - std, mean + std, color=color, alpha=0.2, label='± std')
            plt.xlabel(param)
            plt.ylabel('MCC Score')
            plt.title(f'Influence (moyenne) du paramètre {param}')
            plt.legend()
            plt.grid(True)
            if param in ['alpha', 'beta', 'gamma']:
                plt.xscale('log')

            # Best values histogram
            plt.subplot(1, 2, 2)
            # Convert X values to strings to avoid bar overlap and show exact param values
            x_labels = [str(v) for v in best_counts.keys()]
            y_values = list(best_counts.values())
            plt.bar(x_labels, y_values, color=color)
            plt.xlabel(param)
            plt.ylabel("Fréquence")
            plt.title(f"Histogramme des meilleures valeurs de {param}")
            plt.grid(True, axis='y')

            figures.append(self._create_figure(fig, param, 'plot'))

        return figures

   
 
