from dataclasses import dataclass
import math


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

    def __post_init__(self) -> None:
        for name, value in (
            ("psi_warn", self.psi_warn),
            ("psi_alert", self.psi_alert),
            ("js_warn", self.js_warn),
            ("js_alert", self.js_alert),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.psi_warn >= self.psi_alert:
            raise ValueError("psi_warn must be less than psi_alert")
        if self.js_warn >= self.js_alert:
            raise ValueError("js_warn must be less than js_alert")

        if not math.isfinite(self.ks_alpha) or not 0.0 < self.ks_alpha < 1.0:
            raise ValueError("ks_alpha must be in (0, 1)")

        for name, value in (
            ("null_delta_warn", self.null_delta_warn),
            ("null_delta_alert", self.null_delta_alert),
            ("out_of_range_warn", self.out_of_range_warn),
            ("out_of_range_alert", self.out_of_range_alert),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.null_delta_warn >= self.null_delta_alert:
            raise ValueError("null_delta_warn must be less than null_delta_alert")
        if self.out_of_range_warn >= self.out_of_range_alert:
            raise ValueError("out_of_range_warn must be less than out_of_range_alert")
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        for name, value in (("psi_eps", self.psi_eps), ("js_eps", self.js_eps)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
