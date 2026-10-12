"""Three-day-gap forecasts using historically available repository snapshots.

Model training inputs and targets come only from each origin's eligible snapshot.
Later final labels are used exclusively to score predictions after selection.
"""

import csv
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from data_quality import daily_counts
from uncertainty import split_development, calibrate, bounds, interval_metrics
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Kyiv")
MODELS = [
    "last_observed",
    "weekly",
    "mean7",
    "mean28",
    "ridge_1",
    "ridge_10",
    "ridge_100",
]


def load_counts(path, cutoff, *, assume_continuous_reporting=False, quality_sink=None):
    days, y, quality = daily_counts(
        path, cutoff, assume_continuous_reporting=assume_continuous_reporting
    )
    if quality_sink is not None:
        quality_sink.append({"file": str(Path(path).relative_to(ROOT)), **quality})
    return days, y


def feature(history, target):
    if len(history) < 28:
        raise ValueError("Insufficient observed history")
    return [
        history[-1],
        history[-2],
        history[-7],
        history[-14],
        np.mean(history[-7:]),
        np.mean(history[-14:]),
        np.mean(history[-28:]),
        np.std(history[-7:]),
        np.sin(2 * np.pi * target.weekday() / 7),
        np.cos(2 * np.pi * target.weekday() / 7),
    ]


def training(days, y, gap):
    """For historical target index i use observations only through i-gap."""
    if gap < 1:
        raise ValueError("Gap must be positive")
    indices = list(range(max(28 + gap - 1, len(y) - 365), len(y)))
    X = np.array([feature(y[: i - gap + 1], days[i]) for i in indices])
    return X, y[indices], indices


def predict_all(days, y, target):
    if not days or len(days) != len(y):
        raise ValueError("Dates and counts must have matching nonempty lengths")
    if not np.isfinite(y).all() or (np.asarray(y) < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if any(days[i + 1] - days[i] != timedelta(days=1) for i in range(len(days) - 1)):
        raise ValueError(
            "Input calendar must be continuous; unavailable days cannot be skipped"
        )
    gap = (target - days[-1]).days
    if not 1 <= gap <= 7:
        raise ValueError(
            f"Unsupported data gap {gap}; supported 1..7, never fill missing days with zero"
        )
    X, labels, indices = training(days, y, gap)
    if len(labels) < 60:
        raise ValueError("Insufficient training labels")
    lookup = dict(zip(days, y))
    p = {
        "last_observed": float(y[-1]),
        "weekly": float(lookup[target - timedelta(days=7)]),
        "mean7": float(np.mean(y[-7:])),
        "mean28": float(np.mean(y[-28:])),
    }
    for alpha in [1, 10, 100]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X, labels)
        p[f"ridge_{alpha}"] = max(0.0, float(model.predict([feature(y, target)])[0]))
    return p, {
        "data_gap_days": gap,
        "training_target_last_day": str(days[indices[-1]]),
        "training_examples": len(labels),
        "feature_last_day": str(days[-1]),
    }


def eligible(row):
    if row["status"] != "VERIFIED_RUN_BOUND":
        raise ValueError("Unverified snapshot")
    origin = datetime.fromisoformat(row["forecast_origin_local"])
    if origin.tzinfo is None:
        raise ValueError("Origin timezone is required")
    local = origin.astimezone(TZ)
    if (
        local.hour
        or local.minute
        or local.second
        or local.microsecond
        or str(local.date()) != row["forecast_date"]
    ):
        raise ValueError("Origin must be midnight Europe/Kyiv matching target date")
    if datetime.fromisoformat(row["available_by_utc"].replace("Z", "+00:00")) >= origin:
        raise ValueError("Snapshot not available strictly before forecast")
    cutoff = date.fromisoformat(row["last_calendar_complete_day"])
    if cutoff >= origin.date():
        raise ValueError("Future or partial input date")
    folder = ROOT / "snapshots" / row["snapshot_sha"]
    meta = json.loads((folder / "metadata.json").read_text())
    if meta["sha"] != row["snapshot_sha"] or meta["raw_sha256"] != row["raw_sha256"]:
        raise ValueError("Snapshot provenance mismatch")
    if hashlib.sha256((folder / "kyiv.csv").read_bytes()).hexdigest() != meta.get(
        "kyiv_sha256"
    ):
        raise ValueError("Historical event file checksum mismatch")
    if str(cutoff) != meta["last_calendar_complete_day"]:
        raise ValueError("Cutoff differs from audited snapshot")
    if datetime.fromisoformat(
        meta["latest_global_start_local"]
    ) > datetime.fromisoformat(row["available_by_utc"].replace("Z", "+00:00")):
        raise ValueError("Source events exceed snapshot availability bound")
    from audit_time import INSPECTED_WORKFLOW_SHA256

    if (
        hashlib.sha256((folder / "workflow.yml").read_bytes()).hexdigest()
        != INSPECTED_WORKFLOW_SHA256
    ):
        raise ValueError("Workflow provenance mismatch")
    return folder / "kyiv.csv", cutoff


def score(records, models):
    actual = [r["actual"] for r in records]
    return {
        m: {
            "MAE": float(mean_absolute_error(actual, [r[m] for r in records])),
            "RMSE": float(np.sqrt(mean_squared_error(actual, [r[m] for r in records]))),
        }
        for m in models
    }


def select(records):
    # The interface accepts ONLY validation rows; test labels cannot select a model.
    if not records or any(r["split"] != "validation" for r in records):
        raise ValueError("Validation rows required")
    metrics = score(records, MODELS)
    return min(MODELS, key=lambda m: metrics[m]["MAE"]), metrics


def create_predictions(origins, assume_continuous_reporting, quality_reports):
    predictions = []
    for row in origins:
        path, cutoff = eligible(row)
        days, y = load_counts(
            path,
            cutoff,
            assume_continuous_reporting=assume_continuous_reporting,
            quality_sink=quality_reports,
        )
        p, details = predict_all(days, y, date.fromisoformat(row["forecast_date"]))
        predictions.append(
            {
                **{
                    k: row[k]
                    for k in [
                        "forecast_date",
                        "forecast_origin_local",
                        "split",
                        "snapshot_sha",
                        "available_by_utc",
                    ]
                },
                **details,
                **p,
            }
        )
    return predictions


def available_validation(
    predictions, origins, assume_continuous_reporting, quality_reports
):
    first_test = next(r for r in origins if r["split"] == "test")
    selection_path, selection_cutoff = eligible(first_test)
    vd, vy = load_counts(
        selection_path,
        selection_cutoff,
        assume_continuous_reporting=assume_continuous_reporting,
        quality_sink=quality_reports,
    )
    known = dict(zip(vd, vy))
    val = [
        dict(r, actual=float(known[date.fromisoformat(r["forecast_date"])]))
        for r in predictions
        if r["split"] == "validation"
        and date.fromisoformat(r["forecast_date"]) <= selection_cutoff
    ]
    return val, known, first_test, selection_cutoff


def load_scoring_labels(assume_continuous_reporting, quality_reports):
    # All input-dependent forecasts are computed before loading later scoring labels.
    truth = ROOT / "scoring_data/volunteer.csv"
    provenance = json.loads((ROOT / "scoring_data/provenance.json").read_text())
    if hashlib.sha256(truth.read_bytes()).hexdigest() != provenance["sha256"]:
        raise ValueError("Scoring snapshot checksum mismatch")
    with truth.open() as f:
        raw = list(csv.DictReader(f))
    latest = max(datetime.fromisoformat(r["started_at"]).astimezone(TZ) for r in raw)
    complete = latest.date() - timedelta(days=1)
    d, y = load_counts(
        truth,
        complete,
        assume_continuous_reporting=assume_continuous_reporting,
        quality_sink=quality_reports,
    )
    return dict(zip(d, y)), truth


def main(assume_continuous_reporting=False):
    if not assume_continuous_reporting:
        raise ValueError(
            "Use --assume-continuous-reporting to acknowledge unknown source coverage; otherwise no count forecast is produced"
        )
    with (ROOT / "availability.csv").open() as f:
        origins = list(csv.DictReader(f))
    quality_reports = []
    predictions = create_predictions(
        origins, assume_continuous_reporting, quality_reports
    )
    val, known, first_test, selection_cutoff = available_validation(
        predictions, origins, assume_continuous_reporting, quality_reports
    )
    selection, calibration_rows = split_development(val)
    chosen, vm = select(selection)
    interval = calibrate(calibration_rows, chosen)
    stability = []
    for block in np.array_split(np.array(selection, dtype=object), 3):
        records = list(block)
        winner, metrics = select(records)
        stability.append(
            {
                "first_date": records[0]["forecast_date"],
                "last_date": records[-1]["forecast_date"],
                "samples": len(records),
                "winner": winner,
                "metrics": metrics,
            }
        )
    labels, truth = load_scoring_labels(assume_continuous_reporting, quality_reports)
    test = [r for r in predictions if r["split"] == "test"]
    # Model selection must use labels available BEFORE the first test origin,
    # not revised validation outcomes from the latest scoring snapshot.
    for r in predictions:
        if r["split"] == "validation":
            r["actual"] = (
                float(known[date.fromisoformat(r["forecast_date"])])
                if date.fromisoformat(r["forecast_date"]) in known
                else ""
            )
            r["used_for_selection"] = r["forecast_date"] in {
                s["forecast_date"] for s in selection
            }
            r["used_for_calibration"] = r["forecast_date"] in {
                s["forecast_date"] for s in calibration_rows
            }
        else:
            r["actual"] = float(labels[date.fromisoformat(r["forecast_date"])])
            r["used_for_selection"] = False
            r["used_for_calibration"] = False
        if r["split"] == "test":
            r["interval_lower"], r["interval_upper"] = bounds(
                r[chosen], interval["radius"]
            )
        else:
            r["interval_lower"] = r["interval_upper"] = ""
    reported = list(
        dict.fromkeys(["last_observed", "weekly", "mean7", "mean28", chosen])
    )
    tm = score(test, reported)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "data_quality.json").write_text(json.dumps(quality_reports, indent=2))
    (out / "model_stability.json").write_text(json.dumps(stability, indent=2))
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0]))
        w.writeheader()
        w.writerows(predictions)
    report = {
        "target": "Kyiv City alert starts during local calendar day",
        "simulated_origin": "00:00 Europe/Kyiv; historical replay, not real-time issued predictions",
        "availability_rule": "Input head SHA of successful scheduled GitHub run completed before origin; conservative server updated_at bound",
        "observed_data_gaps_days": sorted({r["data_gap_days"] for r in predictions}),
        "supported_data_gaps_days": [1, 2, 3, 4, 5, 6, 7],
        "validation_dates": [val[0]["forecast_date"], val[-1]["forecast_date"]],
        "validation_origins_used": len(val),
        "validation_origins_excluded_due_to_unavailable_labels": sum(
            r["split"] == "validation" for r in predictions
        )
        - len(val),
        "model_selection_dates": [
            selection[0]["forecast_date"],
            selection[-1]["forecast_date"],
        ],
        "model_selection_samples": len(selection),
        "calibration_dates": [
            calibration_rows[0]["forecast_date"],
            calibration_rows[-1]["forecast_date"],
        ],
        "uncertainty": {**interval, **interval_metrics(test, chosen, interval)},
        "stability_block_winners": [r["winner"] for r in stability],
        "selection_labels_snapshot_sha": first_test["snapshot_sha"],
        "selection_labels_available_by_utc": first_test["available_by_utc"],
        "selection_label_cutoff": str(selection_cutoff),
        "test_dates": [test[0]["forecast_date"], test[-1]["forecast_date"]],
        "selected_model": chosen,
        "validation_metrics": vm,
        "test_metrics": tm,
        "continuous_reporting_assumption": True,
        "source_coverage_verified": False,
        "scoring_snapshot_sha256": hashlib.sha256(truth.read_bytes()).hexdigest(),
        "limitations": [
            "Calendar-complete days assume continuous volunteer reporting; source coverage is not verified.",
            "Availability proof relies on GitHub run metadata and inspected workflow; exact push time and original message publication times unknown.",
            "Within each eligible snapshot, historical training examples use three-day time cutoffs; older revisions known by that forecast origin may be present.",
            "Latest 30-day holdout was already inspected during prototype development. This is a revised backtest, not a pristine unseen test.",
            "Historical replay does not claim predictions were actually issued then.",
            "Empirical calibration does not guarantee coverage under dependent, shifting time-series distributions.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    dates = [date.fromisoformat(r["forecast_date"]) for r in test]
    ax.plot(dates, [r["actual"] for r in test], marker="o", label="Observed")
    ax.plot(dates, [r[chosen] for r in test], label=chosen)
    ax.fill_between(
        dates,
        [r["interval_lower"] for r in test],
        [r["interval_upper"] for r in test],
        alpha=0.15,
        label="Nominal 80% empirical interval",
    )
    ax.set(
        title="Kyiv City: availability-aware historical replay (3-day data gap)",
        ylabel="Alert starts / local day",
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "backtest.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assume-continuous-reporting",
        action="store_true",
        help="Acknowledge that quiet days and reporting outages cannot be distinguished",
    )
    args = parser.parse_args()
    main(args.assume_continuous_reporting)
