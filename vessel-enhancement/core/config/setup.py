from dataclasses import dataclass
from core.config.base import ConfigBase


@dataclass
class SetupConfig(ConfigBase):
    name: str 
    input_dir: str
    output_dir: str
    log_file: str
    debug_mode: bool
    plot_mode: bool
    save_mode: bool