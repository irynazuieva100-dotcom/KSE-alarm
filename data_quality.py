"""Validate observed event starts; never infer verified coverage from silence."""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

TZ = ZoneInfo("Europe/Kyiv")


def read_starts(path, region="Kyiv City"):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Duplicate CSV column names")
        if not {"region", "started_at"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires region and started_at columns")
        selected = []
        total = 0
        for number, row in enumerate(reader, 2):
            total += 1
            if not row.get("region") or not row.get("started_at") or None in row:
                raise ValueError(f"Malformed CSV row {number}")
            try:
                t = datetime.fromisoformat(row["started_at"])
            except ValueError as e:
                raise ValueError(f"Invalid start timestamp at row {number}") from e
            if t.tzinfo is None or t.utcoffset() is None:
                raise ValueError(f"Timestamp must contain timezone at row {number}")
            if row["region"] == region:
                selected.append(t.astimezone(TZ))
    if not selected:
        raise ValueError(f"No events for {region}")
    unique = sorted(set(selected))
    return unique, {
        "source_rows": total,
        "region_rows": len(selected),
        "duplicate_starts_removed": len(selected) - len(unique),
        "coverage_verified": False,
        "coverage_status": "UNKNOWN_NO_SOURCE_COVERAGE_METADATA",
        "end_times_used": False,
        "publication_times_present": False,
    }


def daily_counts(path, cutoff, *, assume_continuous_reporting=False):
    starts, quality = read_starts(path)
    if not assume_continuous_reporting:
        raise ValueError(
            "Source coverage is unknown. Explicit continuous-reporting assumption required for count backtest; no verified zero counts are claimed."
        )
    eligible = [t for t in starts if t.date() <= cutoff]
    if not eligible:
        raise ValueError("No events before cutoff")
    first = eligible[0].date() + timedelta(days=1)
    if cutoff < first:
        raise ValueError("Cutoff before usable history")
    days = [first + timedelta(days=i) for i in range((cutoff - first).days + 1)]
    counts = dict.fromkeys(days, 0.0)
    for t in eligible:
        if t.date() in counts:
            counts[t.date()] += 1
    zero = [str(d) for d in days if counts[d] == 0]
    longest = run = 0
    for d in days:
        run = run + 1 if counts[d] == 0 else 0
        longest = max(longest, run)
    quality.update(
        continuous_reporting_assumption=True,
        last_input_day=str(cutoff),
        excluded_after_cutoff=len(starts) - len(eligible),
        days_without_recorded_starts=len(zero),
        longest_run_without_recorded_starts=longest,
        zero_day_interpretation="No recorded starts; treated as zero ONLY under explicit coverage assumption",
        warning="Reporting gaps cannot be distinguished from quiet days; metrics are conditional on assumed coverage.",
    )
    return days, np.array([counts[d] for d in days]), quality
