from dataclasses import dataclass
from core.config.base import ConfigBase

@dataclass
class Metrics(ConfigBase):
    dice: float
    mcc: float
    roc: float
    pr: float