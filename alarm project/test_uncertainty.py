import unittest
from uncertainty import calibrate, bounds, split_development


class IntervalTests(unittest.TestCase):
    def test_integer_bounds_and_nonnegative_support(self):
        self.assertEqual(bounds(1.2, 2.4), (0, 4))
        with self.assertRaises(ValueError):
            bounds(1, -1)

    def test_order_statistic(self):
        rows = [{"actual": i, "model": 0} for i in range(20)]
        result = calibrate(rows, "model", 0.8)
        self.assertEqual(result["rank"], 17)
        self.assertEqual(result["radius"], 16)
        self.assertFalse(result["guaranteed_coverage"])

    def test_calibration_and_selection_disjoint(self):
        rows = [
            {"forecast_date": f"2026-01-{i:03}", "split": "validation"}
            for i in range(1, 59)
        ]
        a, b = split_development(rows)
        self.assertEqual(len(a), 30)
        self.assertEqual(len(b), 28)
        self.assertFalse(
            {r["forecast_date"] for r in a} & {r["forecast_date"] for r in b}
        )

    def test_test_data_cannot_calibrate_development_split(self):
        with self.assertRaises(ValueError):
            split_development([{"split": "test"}] * 58)


if __name__ == "__main__":
    unittest.main()
