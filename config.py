from dataclasses import dataclass


@dataclass
class DriftTrackerConfig:
    """Configuration thresholds for batch ML drift and data quality monitoring."""

    psi_warn: float = 0.1
    psi_alert: float = 0.25
    js_warn: float = 0.05
    js_alert: float = 0.15
    ks_alpha: float = 0.05
    null_delta_warn: float = 0.02
    null_delta_alert: float = 0.05
    out_of_range_warn: float = 0.01
    out_of_range_alert: float = 0.05
    n_bins: int = 10
    psi_eps: float = 1e-6
    js_eps: float = 1e-8
