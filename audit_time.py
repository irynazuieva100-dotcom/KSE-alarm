"""Conservative historical availability audit, independent of forecast model."""

import csv, hashlib, io, json, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Kyiv")
REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
INSPECTED_WORKFLOW_SHA256 = (
    "da4684bd25347d9c9e13782db2cea6444ae0114173894a2969ef9c52cf83bf99"
)


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "air-alert-audit"}),
                timeout=30,
            ) as f:
                return f.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def stamp(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def snapshot(run):
    sha = run["head_sha"]
    folder = ROOT / "snapshots" / sha
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metadata.json").exists():
        if (
            hashlib.sha256((folder / "workflow.yml").read_bytes()).hexdigest()
            != INSPECTED_WORKFLOW_SHA256
        ):
            raise ValueError("Cached workflow differs from inspected configuration")
        cached = json.loads((folder / "metadata.json").read_text())
        if hashlib.sha256((folder / "kyiv.csv").read_bytes()).hexdigest() != cached.get(
            "kyiv_sha256"
        ):
            raise ValueError("Cached regional CSV checksum mismatch")
        return cached
    prefix = f"https://raw.githubusercontent.com/{REPO}/{sha}/"
    workflow = fetch(prefix + ".github/workflows/update_datasets.yml")
    text = workflow.decode()
    # Only the inspected scheduled workflow configuration is admissible.
    if hashlib.sha256(workflow).hexdigest() != INSPECTED_WORKFLOW_SHA256:
        raise ValueError("Unrecognized checkout configuration: " + sha)
    raw = fetch(prefix + "datasets/volunteer_data_en.csv")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    if (
        not rows
        or not {"region", "started_at", "finished_at", "naive"} <= rows[0].keys()
    ):
        raise ValueError("Unknown CSV schema")
    latest = max(datetime.fromisoformat(r["started_at"]).astimezone(TZ) for r in rows)
    city = [r for r in rows if r["region"] == "Kyiv City"]
    with (folder / "kyiv.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(city)
    (folder / "workflow.yml").write_bytes(workflow)
    meta = {
        "sha": sha,
        "source_url": prefix + "datasets/volunteer_data_en.csv",
        "kyiv_sha256": hashlib.sha256((folder / "kyiv.csv").read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "kyiv_rows": len(city),
        "latest_global_start_local": latest.isoformat(),
        "last_calendar_complete_day": str(latest.date() - timedelta(days=1)),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "availability_evidence": "Successful scheduled GitHub Actions run with this head_sha and inspected default event-SHA checkout; updated_at is used as conservative upper bound, not exact publication time.",
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def choose(runs, origin):
    valid = [
        r
        for r in runs
        if r["event"] == "schedule"
        and r["conclusion"] == "success"
        and r["status"] == "completed"
        and stamp(r["updated_at"]) < origin
    ]
    return max(valid, key=lambda r: stamp(r["updated_at"])) if valid else None


def main():
    runs = json.loads((ROOT / "runs.json").read_text())["workflow_runs"]
    first = datetime(2026, 7, 13, tzinfo=TZ)
    last = datetime(2026, 10, 10, tzinfo=TZ)
    requests = []
    unique = {}
    for i in range((last - first).days + 1):
        origin = first + timedelta(days=i)
        run = choose(runs, origin)
        requests.append((origin, run))
        if run:
            unique[run["head_sha"]] = run
    metadata = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(snapshot, r): sha for sha, r in unique.items()}
        for count, f in enumerate(as_completed(futures), 1):
            sha = futures[f]
            try:
                metadata[sha] = f.result()
            except Exception as e:
                errors[sha] = str(e)
            if count % 10 == 0 or count == len(futures):
                print(
                    f"Checked snapshots {count}/{len(futures)}; failures {len(errors)}",
                    flush=True,
                )
    table = []
    for origin, run in requests:
        row = {
            "forecast_origin_local": origin.isoformat(),
            "forecast_date": str(origin.date()),
            "split": (
                "validation" if origin.date() < datetime(2026, 9, 11).date() else "test"
            ),
        }
        if not run:
            row["status"] = "NO_PRIOR_RUN"
        elif run["head_sha"] in errors:
            row.update(status="UNVERIFIED_SNAPSHOT", error=errors[run["head_sha"]])
        else:
            meta = metadata[run["head_sha"]]
            end = datetime.fromisoformat(meta["last_calendar_complete_day"]).date()
            row.update(
                status="VERIFIED_RUN_BOUND",
                snapshot_sha=run["head_sha"],
                run_id=run["id"],
                evidence_url=run["html_url"],
                available_by_utc=run["updated_at"],
                available_by_local=stamp(run["updated_at"]).astimezone(TZ).isoformat(),
                latest_global_start_local=meta["latest_global_start_local"],
                last_calendar_complete_day=str(end),
                data_lag_days=(origin.date() - end).days,
                raw_sha256=meta["raw_sha256"],
            )
            assert stamp(row["available_by_utc"]) < origin
            assert end < origin.date()
        table.append(row)
    fields = list(dict.fromkeys(k for row in table for k in row))
    with (ROOT / "availability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    summary = {
        "origins": len(table),
        "verified": sum(r["status"] == "VERIFIED_RUN_BOUND" for r in table),
        "unique_snapshots": len(unique),
        "download_errors": errors,
        "validation_origins": sum(r["split"] == "validation" for r in table),
        "test_origins": sum(r["split"] == "test" for r in table),
        "lag_days": sorted({r["data_lag_days"] for r in table if "data_lag_days" in r}),
        "publication_scope": "Snapshot availability bound; individual original-message publication times unknown. Use only event data from the bound snapshot; never later revisions as inputs.",
        "coverage_scope": "Calendar-complete is NOT verified complete source coverage. No regional outage metadata available.",
        "historical_simulation": "Forecast origins are retrospective simulated origins, not claims of predictions actually issued then.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
