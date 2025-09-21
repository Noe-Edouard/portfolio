import argparse

LOG_DIR = "logs"
INPUT_DIR = "data"
OUTPUT_DIR = "outputs"

def get_parser():
    parser = argparse.ArgumentParser(description="Runner pipeline")

    parser.add_argument('--run_pipeline', action='store_true', help='Run the pipeline')
    parser.add_argument('--run_analyzer', action='store_true', help='Run the analyzer')
    parser.add_argument('--run_benchmark', action='store_true', help='Run the benchmark')
    parser.add_argument('--benchmark_type', choices=['hessian', 'enhancement'], default='hessian', help='Type of benchmark')
    parser.add_argument('--test', action='store_true', help='Use test config')
    
    return parser



