"""Chronologically separate model selection and empirical interval calibration."""

import math
import numpy as np


def split_development(records):
    if len(records) < 40:
        raise ValueError("At least 40 known pre-test outcomes required")
    if any(r["split"] != "validation" for r in records):
        raise ValueError("Only pre-test validation rows allowed")
    dates = [r["forecast_date"] for r in records]
    if dates != sorted(set(dates)):
        raise ValueError("Development dates must be unique and ordered")
    # Predeclared split: first 30 available dates select the model; remainder calibrates.
    return records[:30], records[30:]


def calibrate(records, model, coverage=0.8):
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    if len(records) < 10:
        raise ValueError("At least 10 calibration outcomes required")
    errors = sorted(abs(float(r["actual"]) - float(r[model])) for r in records)
    if not np.isfinite(errors).all():
        raise ValueError("Nonfinite calibration errors")
    rank = math.ceil((len(errors) + 1) * coverage)
    if rank > len(errors):
        raise ValueError("Insufficient calibration for requested coverage")
    return {
        "nominal_coverage": coverage,
        "radius": errors[rank - 1],
        "calibration_samples": len(errors),
        "rank": rank,
        "method": "finite-sample absolute-residual order statistic; chronological held-out calibration",
        "guaranteed_coverage": False,
        "caveat": "Time-series dependence and distribution shifts violate exchangeability; nominal coverage is a target, not a guarantee.",
    }


def bounds(point, radius):
    if not math.isfinite(point) or not math.isfinite(radius) or point < 0 or radius < 0:
        raise ValueError("Invalid interval inputs")
    # Outward integer rounding for integer count observations.
    return max(0, math.floor(point - radius)), math.ceil(point + radius)


def interval_metrics(records, model, calibration):
    b = [bounds(float(r[model]), calibration["radius"]) for r in records]
    coverage = np.mean(
        [lo <= float(r["actual"]) <= hi for r, (lo, hi) in zip(records, b)]
    )
    return {
        "empirical_coverage": float(coverage),
        "mean_width": float(np.mean([hi - lo for lo, hi in b])),
        "test_samples": len(records),
    }
