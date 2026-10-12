import tempfile, unittest, csv, shutil
from pathlib import Path
from datetime import date
from unittest.mock import patch
import forecast
from data_quality import daily_counts, read_starts


class QualityTests(unittest.TestCase):
    def write(self, d, text):
        p = Path(d) / "events.csv"
        p.write_text(text)
        return p

    def test_missing_schema_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, "region,wrong\nKyiv City,2026-01-01\n")
            with self.assertRaises(ValueError):
                read_starts(p)

    def test_naive_and_invalid_timestamps_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for t in ["2026-01-01 00:00:00", "not-a-date", ""]:
                p = self.write(d, "region,started_at\nKyiv City," + t + "\n")
                with self.assertRaises(ValueError):
                    read_starts(p)

    def test_coverage_is_not_silently_assumed(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(
                d, "region,started_at\nKyiv City,2026-01-01T00:00:00+00:00\n"
            )
            with self.assertRaises(ValueError):
                daily_counts(p, date(2026, 1, 4))

    def test_duplicates_and_unverified_zero_days_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(
                d,
                "region,started_at\nKyiv City,2026-01-01T00:00:00+00:00\nKyiv City,2026-01-03T00:00:00+00:00\nKyiv City,2026-01-03T00:00:00+00:00\n",
            )
            days, y, q = daily_counts(
                p, date(2026, 1, 4), assume_continuous_reporting=True
            )
            self.assertEqual(y.tolist(), [0.0, 1.0, 0.0])
            self.assertFalse(q["coverage_verified"])
            self.assertEqual(q["duplicate_starts_removed"], 1)
            self.assertEqual(q["days_without_recorded_starts"], 2)

    def test_corrupted_historical_file_is_rejected(self):
        with (forecast.ROOT / "availability.csv").open() as f:
            row = next(csv.DictReader(f))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "snapshots" / row["snapshot_sha"]
            shutil.copytree(forecast.ROOT / "snapshots" / row["snapshot_sha"], folder)
            with (folder / "kyiv.csv").open("a") as f:
                f.write("\n")
            with patch.object(forecast, "ROOT", root), self.assertRaises(ValueError):
                forecast.eligible(row)

    def test_default_main_refuses_unknown_coverage(self):
        with self.assertRaises(ValueError):
            forecast.main()


if __name__ == "__main__":
    unittest.main()
