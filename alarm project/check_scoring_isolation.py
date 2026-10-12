"""End-to-end check: later scoring outcomes cannot affect model selection/input."""

import contextlib, csv, io, json, shutil, tempfile
from pathlib import Path
from unittest.mock import patch
import forecast


def main():
    base = forecast.ROOT
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shutil.copy(base / "availability.csv", root / "availability.csv")
        # Copy cache to avoid platform-dependent symlinks and source mutations.
        for name in ["snapshots", "scoring_data"]:
            shutil.copytree(base / name, root / name)
        original = forecast.load_counts
        with patch.object(forecast, "ROOT", root), contextlib.redirect_stdout(
            io.StringIO()
        ):
            forecast.main(assume_continuous_reporting=True)
            first = json.loads((root / "results/metrics.json").read_text())
            with (root / "results/predictions.csv").open() as f:
                a = list(csv.DictReader(f))

            def altered(path, cutoff, **kwargs):
                days, y = original(path, cutoff, **kwargs)
                if Path(path).parent.name == "scoring_data":
                    y = y + 1000
                return days, y

            with patch.object(forecast, "load_counts", altered):
                forecast.main(assume_continuous_reporting=True)
            second = json.loads((root / "results/metrics.json").read_text())
            with (root / "results/predictions.csv").open() as f:
                b = list(csv.DictReader(f))
        if (
            first["selected_model"] != second["selected_model"]
            or first["validation_metrics"] != second["validation_metrics"]
        ):
            raise AssertionError("Future scoring labels changed selection")
        if len(a) != 90 or len(b) != 90:
            raise AssertionError("Unexpected forecast count")
        for x, y in zip(a, b):
            for model in forecast.MODELS:
                if x[model] != y[model]:
                    raise AssertionError("Future scoring labels changed prediction")
        if first["test_metrics"] == second["test_metrics"]:
            raise AssertionError("Perturbation had no effect on scores")
    print(
        "PASS: all 90 forecasts and model selection unchanged when later scoring labels change."
    )


if __name__ == "__main__":
    main()
