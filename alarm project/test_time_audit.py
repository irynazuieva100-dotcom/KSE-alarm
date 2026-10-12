import unittest
from datetime import datetime, timezone
from audit_time import choose


class AvailabilityTests(unittest.TestCase):
    def test_strictly_before_origin(self):
        origin = datetime(2026, 10, 10, tzinfo=timezone.utc)
        runs = [
            dict(
                event="schedule",
                conclusion="success",
                status="completed",
                updated_at=t,
                id=i,
            )
            for i, t in enumerate(
                ["2026-10-09T23:59:59Z", "2026-10-10T00:00:00Z", "2026-10-10T00:01:00Z"]
            )
        ]
        self.assertEqual(choose(runs, origin)["id"], 0)

    def test_failed_and_unfinished_runs_are_excluded(self):
        origin = datetime(2026, 10, 10, tzinfo=timezone.utc)
        runs = [
            dict(
                event="schedule",
                conclusion="failure",
                status="completed",
                updated_at="2026-10-09T12:00:00Z",
            ),
            dict(
                event="schedule",
                conclusion="success",
                status="in_progress",
                updated_at="2026-10-09T12:00:00Z",
            ),
        ]
        self.assertIsNone(choose(runs, origin))

    def test_no_history_means_no_candidate(self):
        self.assertIsNone(choose([], datetime.now(timezone.utc)))


if __name__ == "__main__":
    unittest.main()
