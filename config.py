
import dataclasses

@dataclasses.dataclass
class Config:
    data_size: int = 1000
    drift_threshold: float = 0.1
