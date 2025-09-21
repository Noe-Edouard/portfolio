from core.experiments.analyzer import Analyzer
from core.experiments.benchmarks.runner import BenchmarkRunner 
from core.processing.pipeline import Pipeline
from core.config.builder import ConfigBuilder
from core.config.setup import SetupConfig
from core.config.benchmark import RunnerConfig, BenchmarkConfig
from core.config.experiment import ExperimentConfig
from configs.args import get_parser


parser = get_parser()
args = parser.parse_args()

run_pipeline = args.run_pipeline
run_analyzer = args.run_analyzer
run_benchmark = args.run_benchmark
benchmark_type = args.benchmark_type
test = args.test


ROOT = "tests/" if test else "" 

SRC_PIPELINE_SETUP = ROOT + 'configs/pipeline/setup.yaml'
SRC_PIPELINE_EXPERIMENT = ROOT + 'configs/pipeline/experiment.yaml'

SRC_ANALYZER_SETUP = ROOT + 'configs/analyzer/setup.yaml'

SRC_BENCHMARK_RUNNER = ROOT + 'configs/benchmark/runner.yaml'
SRC_BENCHMARK_HESSIAN = ROOT + 'configs/benchmark/hessian.yaml'
SRC_BENCHMARK_ENHANCEMENT = ROOT + 'configs/benchmark/enhancement.yaml'
SRC_BENCHMARK_EXPERIMENT = ROOT + 'configs/benchmark/experiment.yaml'



def main():
    

    if run_pipeline:
        setup_config: SetupConfig = ConfigBuilder(SRC_PIPELINE_SETUP, SetupConfig)
        experiment_config: SetupConfig = ConfigBuilder(SRC_PIPELINE_EXPERIMENT, ExperimentConfig)
        
        pipeline = Pipeline(setup_config)
        pipeline.run(experiment_config)


    if run_analyzer:
        setup_config = ConfigBuilder(SRC_ANALYZER_SETUP, SetupConfig)
        
        analyzer = Analyzer(setup_config)
        analyzer.run()

    if run_benchmark:
        runner_config: RunnerConfig = ConfigBuilder(SRC_BENCHMARK_RUNNER, RunnerConfig)
        experiment_config = ConfigBuilder(SRC_BENCHMARK_EXPERIMENT, ExperimentConfig)
        
        if benchmark_type == "hessian":
            benchmark_config: BenchmarkConfig = ConfigBuilder(SRC_BENCHMARK_HESSIAN, BenchmarkConfig)
        elif benchmark_type == "enhancement":
            benchmark_config: BenchmarkConfig = ConfigBuilder(SRC_BENCHMARK_ENHANCEMENT, BenchmarkConfig)
        else:
            raise ValueError(f'Benchmark type unknown: {benchmark_type}')
        
        runner = BenchmarkRunner(runner_config.setup)
        dirname = runner.run(
            images_dir=runner_config.images_dir, 
            labels_dir=runner_config.labels_dir,
            benchmark_config=benchmark_config,
            experiment_config=experiment_config,
        )     
        
        runner.analyse(benchmark_config=benchmark_config, results_dir=dirname)  
   
        
if __name__ == "__main__":
    main()
    
  

    
        
