"""Issue an actually timestamped future-day forecast using an available snapshot.

Uses a previously selected historical model, not tuned on the test set.
Updates public run metadata, conservatively selects an input head snapshot,
and records what was available before the actual forecast issuance.
"""

import argparse, hashlib, json, urllib.request
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from pathlib import Path
import forecast
from audit_time import choose, snapshot
from uncertainty import bounds

TZ = ZoneInfo("Europe/Kyiv")
ROOT = Path(__file__).resolve().parent


def main(assume_continuous_reporting=False):
    if not assume_continuous_reporting:
        raise ValueError("Explicit continuous-reporting assumption required")
    report = json.loads((ROOT / "results/metrics.json").read_text())
    endpoint = "https://api.github.com/repos/Vadimkin/ukrainian-air-raid-sirens-dataset/actions/runs?per_page=30"
    with urllib.request.urlopen(
        urllib.request.Request(endpoint, headers={"User-Agent": "air-alert-research"}),
        timeout=30,
    ) as f:
        runs = json.load(f)
    observed = datetime.now(timezone.utc)
    run = choose(runs["workflow_runs"], observed)
    if run is None:
        raise ValueError("No successful eligible source run")
    if (
        observed - datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
    ).total_seconds() > 3 * 86400:
        raise ValueError("No recent source availability evidence")
    meta = snapshot(run)
    folder = ROOT / "snapshots" / run["head_sha"]
    cutoff = date.fromisoformat(meta["last_calendar_complete_day"])
    if (
        hashlib.sha256((folder / "kyiv.csv").read_bytes()).hexdigest()
        != meta["kyiv_sha256"]
    ):
        raise ValueError("Snapshot checksum mismatch")
    issued = datetime.now(TZ)
    target = issued.date() + timedelta(days=1)
    available = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
    if available >= issued:
        raise ValueError("Source availability must precede actual issuance")
    if datetime.fromisoformat(meta["latest_global_start_local"]) > available:
        raise ValueError("Source event dates exceed availability bound")
    selection_bound = datetime.fromisoformat(
        report["selection_labels_available_by_utc"].replace("Z", "+00:00")
    )
    if selection_bound >= issued:
        raise ValueError("Selection labels unavailable at issuance")
    quality = []
    days, y = forecast.load_counts(
        folder / "kyiv.csv",
        cutoff,
        assume_continuous_reporting=True,
        quality_sink=quality,
    )
    predictions, details = forecast.predict_all(days, y, target)
    generated = datetime.now(TZ)
    if generated.date() != issued.date():
        raise ValueError("Midnight crossed during issuance; rerun")
    model = report["selected_model"]
    point = predictions[model]
    lo, hi = bounds(point, report["uncertainty"]["radius"])
    result = {
        "created_at_local": generated.isoformat(),
        "created_at_utc": generated.astimezone(timezone.utc).isoformat(),
        "target_date_local": str(target),
        "target_timezone": "Europe/Kyiv",
        "target": "Count of recorded Kyiv City alert starts over the full next local calendar day",
        "selected_model": model,
        "expected_count": point,
        "interval_lower": lo,
        "interval_upper": hi,
        "nominal_interval_coverage": report["uncertainty"]["nominal_coverage"],
        "coverage_guaranteed": False,
        "snapshot_sha": run["head_sha"],
        "snapshot_available_by_utc": run["updated_at"],
        "snapshot_evidence_url": run["html_url"],
        "raw_sha256": meta["raw_sha256"],
        "selection_labels_available_by_utc": report[
            "selection_labels_available_by_utc"
        ],
        "historical_model_report_sha256": hashlib.sha256(
            (ROOT / "results/metrics.json").read_bytes()
        ).hexdigest(),
        **details,
        "data_quality": quality[0],
        "warning": "Research forecast only. Source coverage is assumed and empirical interval is not guaranteed. Follow official alerts.",
    }
    path = ROOT / "results/issued_forecasts"
    path.mkdir(exist_ok=True)
    name = generated.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    (path / name).write_text(json.dumps(result, indent=2))
    (path / (name + ".source_runs.json")).write_text(json.dumps(runs, indent=2))
    (ROOT / "results/next_day_forecast.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assume-continuous-reporting", action="store_true")
    args = parser.parse_args()
    main(args.assume_continuous_reporting)
