
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pathlib import Path
from pprint import pformat
from tabulate import tabulate
from dataclasses import fields


from core.config.experiment import Experiment, ExperimentConfig
from core.config.metrics import Metrics
from core.config.figure import FigureData
from core.experiments.analytics.base import AnalyticsBase
from core.experiments.metrics import roc_curve, precision_recall_curve
from core.utils.helpers import create_error_map

from pprint import pformat

class AnalyticsHessian(AnalyticsBase):
    

    def get_configs(self, experiments: list[Experiment]) -> FigureData:
        from dataclasses import asdict
        
        configs: dict[str, ExperimentConfig] = {}
        for experiment in experiments:
            configs[experiment.config.methods.derivator] = experiment.config
        
        content = f"EXPERIMENT CONFIG FOR < {Path(experiment.config.loading.raw_file).stem} >\n"
        for method, config in configs.items():
            config_dict = asdict(config)
            config_str = pformat(config_dict, indent=4, sort_dicts=False)
            content += f"\n{method.upper()}:\n{config_str}\n"

        figure = self._create_figure(content, 'configs', 'text')
        
        return figure



    def get_metrics(self, experiments: list[Experiment]) -> FigureData:
        metrics = [metric.name for metric in fields(Metrics)]
        rows = []

        for experiment in experiments:
            row = {
                'method': experiment.config.methods.derivator,
                **{metric: f'{experiment.metrics[metric]:.4f}' for metric in metrics}
            }
            rows.append(row)

        # Best methods
        best_methods = {'method': 'best'}
        for metric in metrics:
            best_row = max(rows, key=lambda r: r[metric])
            best_methods[metric] = best_row['method']
        rows.append(best_methods)

        # Format table
        table = tabulate(rows, headers="keys", tablefmt='github', floatfmt='.4f')
        content = f'\nHESSIAN BENCHMARK METRICS\n{table}'

        figure = self._create_figure(content, 'metrics', 'text')
        return figure



    def get_histograms(self,
            experiments: list[Experiment],
            data_raw: np.ndarray,
            data_gt: np.ndarray,
            bins: int = 50,
            density: bool = False,
            color: str = 'dodgerblue',
        ) -> FigureData:
        
        ncols = len(experiments) + 1
        nrows = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3 * nrows))

        # Add vertical labels
        fig.text(0.015, 0.752, 'Enhanced', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')
        fig.text(0.015, 0.278, 'Segmented', va='center', ha='center', rotation='vertical', fontsize=11, fontweight='bold')

        # Data and titles for each row
        row_data = [
            [data_raw] + [exp.data_enhanced for exp in experiments],
            [data_gt] + [exp.data_segmented for exp in experiments],
        ]
        row_titles = [
            ['raw data'] + [exp.config.methods.derivator for exp in experiments],
            ['ground truth'] + [exp.config.methods.derivator for exp in experiments],
        ]

        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].hist(row_data[row][col].ravel(), bins=bins, density=density, color=color)
                axs[row, col].set_title(row_titles[row][col], fontsize=9)
                if row == 0:
                    axs[row, col].set_ylabel('Density' if density else 'Frequency')
                else:
                    axs[row, col].set_xlabel('Intensity')
                    axs[row, col].set_ylabel('Density' if density else 'Frequency')
                axs[row, col].grid(True)
                axs[row, col].tick_params('y', labelsize=8, labelrotation=90)

        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.15, hspace=0.2)
        
        figure = self._create_figure(fig, 'histograms', 'plot')
        plt.close(fig)
        
        return figure


    def get_curves(self, 
            experiments: list[Experiment], 
            ground_truth: ndarray
        ) -> FigureData:
        
        y_true = ground_truth.ravel()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(experiments)))
        for i, experiment in enumerate(experiments):
            y_scores = experiment.data_enhanced.ravel()

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax1.plot(fpr, tpr, label=experiment.config.methods.derivator, color=colors[i])

            # PR
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ax2.plot(recall, precision, label=experiment.config.methods.derivator, color=colors[i])

        # ROC subplot
        ax1.plot([0, 1], [0, 1], 'k--', label='random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # PR subplot
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        # Final display
        fig.tight_layout()
        figure = self._create_figure(fig, 'curves', 'plot')
        plt.close(fig)
        
        return figure

    def get_views(self,
            experiments: list[Experiment],
            data_gt: ndarray,
            data_raw: ndarray
        ) -> list[FigureData]:
        dim = data_gt.ndim
        data_enhanced = [data_raw] + [exp.data_enhanced for exp in experiments]
        data_segmented = [data_gt] + [exp.data_segmented for exp in experiments]
        error_maps = [data_gt] + [create_error_map(data_gt, exp.data_segmented) for exp in experiments]
        methods = [exp.config.methods.derivator for exp in experiments]

        titles_enhanced = ['raw data'] + methods
        titles_segmented = ['ground truth'] + methods
        titles_error_maps = ['ground truth'] + methods

        if dim == 2:
            plot_enhanced   = self.viewer.display_images(data_enhanced, titles=titles_enhanced)
            plot_segmented  = self.viewer.display_images(data_segmented, titles=titles_segmented, binary_mode=True)
            plot_error_maps = self.viewer.display_images(error_maps, titles=titles_error_maps, error_mode=True)
            
            fig_enhanced    = self._create_figure(plot_enhanced, 'images_enhanced', 'plot')
            fig_segmented   = self._create_figure(plot_segmented, 'images_segmented', 'plot')
            fig_error_maps  = self._create_figure(plot_error_maps, 'error_maps', 'plot')
            
            figures = [fig_enhanced, fig_segmented, fig_error_maps]

        
        else: # dim == 3
            mip_enhanced      = self.viewer.display_mip(data_enhanced, titles=titles_enhanced)
            slices_enhanced   = self.viewer.display_slices(data_enhanced, titles=titles_enhanced, interval=100)
            slices_segmented  = self.viewer.display_slices(data_segmented, titles=titles_segmented, interval=100, binary_mode=True)
            slices_error_maps = self.viewer.display_slices(error_maps, titles=titles_error_maps, interval=100, error_mode=True)
            
            fig_mip_enhanced      = self._create_figure(mip_enhanced, 'mip_enhanced', 'plot')
            fig_slices_enhanced   = self._create_figure(slices_enhanced, 'slices_enhanced', 'anim')
            fig_slices_segmented  = self._create_figure(slices_segmented, 'slices_segmented', 'anim')
            fig_slices_error_maps = self._create_figure(slices_error_maps, 'slices_error_maps', 'anim')
            
            figures = [fig_mip_enhanced, fig_slices_enhanced, fig_slices_segmented, fig_slices_error_maps]

        plt.close('all')
        
        return figures
            
            
            
            
    
         
            


