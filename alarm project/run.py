"""Small public entry point: backtest, check, or next-day forecast."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Kyiv recorded air-alert start forecasting with historical input-availability checks"
    )
    parser.add_argument("command", choices=["backtest", "check", "next-day"])
    parser.add_argument(
        "--assume-continuous-reporting",
        action="store_true",
        help="Required for count forecasting because source coverage is not verified",
    )
    args = parser.parse_args()
    if args.command == "check":
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v"], cwd=ROOT, check=True
        )
        subprocess.run(
            [sys.executable, "check_scoring_isolation.py"], cwd=ROOT, check=True
        )
    else:
        if not args.assume_continuous_reporting:
            parser.error(
                "count forecasting requires --assume-continuous-reporting; source coverage is unknown"
            )
        script = "forecast.py" if args.command == "backtest" else "issue_forecast.py"
        subprocess.run(
            [sys.executable, script, "--assume-continuous-reporting"],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
