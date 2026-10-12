import unittest, csv, tempfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
from forecast import training, predict_all, load_counts, select, eligible, ROOT


class ForecastTests(unittest.TestCase):
    def series(self):
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(400)]
        return days, np.array([(i * 7) % 11 for i in range(400)], dtype=float)

    def test_training_features_exclude_gap_and_target(self):
        days, y = self.series()
        X, labels, indices = training(days, y, 3)
        for row, i in zip(X, indices):
            self.assertEqual(row[0], y[i - 3])
        i = indices[10]
        z = y.copy()
        z[i - 2 :] += 10000
        Z, _, _ = training(days, z, 3)
        np.testing.assert_array_equal(X[10], Z[10])

    def test_unsupported_gap_rejected(self):
        days, y = self.series()
        with self.assertRaises(ValueError):
            predict_all(days, y, days[-1] + timedelta(days=8))

    def test_all_supported_gaps_align_training_features(self):
        days, y = self.series()
        for gap in range(1, 8):
            X, labels, indices = training(days, y, gap)
            for row, i in zip(X, indices):
                self.assertEqual(row[0], y[i - gap])
            p, q = predict_all(days, y, days[-1] + timedelta(days=gap))
            self.assertEqual(q["data_gap_days"], gap)
            self.assertTrue(all(np.isfinite(v) and v >= 0 for v in p.values()))

    def test_nonfinite_and_discontinuous_inputs_rejected(self):
        days, y = self.series()
        target = days[-1] + timedelta(days=3)
        broken = y.copy()
        broken[10] = np.nan
        with self.assertRaises(ValueError):
            predict_all(days, broken, target)
        missing = days.copy()
        missing[10] += timedelta(days=1)
        with self.assertRaises(ValueError):
            predict_all(missing, y, target)

    def test_test_rows_cannot_select_model(self):
        with self.assertRaises(ValueError):
            select([{"split": "test"}])

    def test_future_events_and_end_times_do_not_change_counts(self):
        text = "region,started_at,finished_at,naive\nKyiv City,2026-01-01 22:30:00+00:00,2099-01-01 00:00:00+00:00,True\nKyiv City,2026-01-02 22:30:00+00:00,,False\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.csv"
            p.write_text(text)
            days, y = load_counts(p, date(2026, 1, 4), assume_continuous_reporting=True)
            p.write_text(text + "Kyiv City,2026-01-10 00:00:00+00:00,,False\n")
            other, z = load_counts(
                p, date(2026, 1, 4), assume_continuous_reporting=True
            )
            self.assertEqual(days, other)
            np.testing.assert_array_equal(y, z)

    def test_origin_and_cutoff_fail_closed(self):
        with (ROOT / "availability.csv").open() as f:
            r = next(csv.DictReader(f))
        eligible(r)
        invalid = dict(r, available_by_utc=r["forecast_origin_local"])
        with self.assertRaises(ValueError):
            eligible(invalid)
        invalid = dict(r, last_calendar_complete_day="2026-07-11")
        with self.assertRaises(ValueError):
            eligible(invalid)


if __name__ == "__main__":
    unittest.main()
