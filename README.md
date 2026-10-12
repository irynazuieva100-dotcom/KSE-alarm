# Kyiv Alert Forecast

A Python mini-project for KSE Stage 2: next-day forecasts of **recorded air-alert starts in Kyiv City**, with historical input-availability evidence, data checks and empirical uncertainty intervals. Research use only; follow official alerts for safety.

## Quick start

Python 3.11+ recommended. From the extracted project directory:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py backtest --assume-continuous-reporting
```

Optional: create and activate a virtual environment first. On macOS/Linux use `python3` if `python` is unavailable. The committed historical backtest works without network access or API tokens after dependencies are installed.

To issue a forecast for the next full local calendar day (network access required):

```bash
python run.py next-day --assume-continuous-reporting
```

The explicit flag acknowledges that source outages cannot be distinguished from quiet days. Without it, no count forecast is produced. It is an assumption, not verified source coverage.

## Data and time availability

Public source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset . The volunteer data are collected from eTryvoga. Upstream MIT license is in SOURCE_LICENSE. Kyiv City is distinct from Kyivska oblast.

For every historical origin, use the input head SHA of a successful scheduled GitHub Actions run whose server-reported updated_at is strictly earlier than the forecast origin. Inspect the exact workflow at that SHA and use its default event-SHA checkout. A successful run provides a conservative upper bound on snapshot availability, not an exact push timestamp. The head snapshot is selected rather than the new output commit from the run. Full execution logs were inaccessible (403); the evidence is run metadata plus inspected workflow and snapshot identity. Exact public publication times and original Telegram message publication times remain unknown.

The audit covers 90 simulated midnight origins, 13 July–10 October 2026. All have calendar-complete data ending three days before the forecast date. Time conversion uses Europe/Kyiv including DST. The first partial day and latest partial day are excluded. Training features use the same observation gap as their targets. Gaps 1–7 days are supported; larger gaps are rejected. Missing intervening days are not filled with zero.

Coverage is not independently verified. No recorded starts within the assumed observed period are counted as zero only under the explicit continuous-reporting assumption. Data-quality reports make this limitation visible. No inferred quiet day is labeled a verified zero. Metrics measure agreement with the volunteer feed, not independently verified complete real-world alert counts.

## Evaluation design

Candidates: last observed count, same weekday last week, trailing 7/28-day mean, Ridge alpha 1/10/100. Ridge uses lags, trailing statistics and known weekday features, with fitting/scaling limited to eligible past observations and the latest 365 available training targets.

Thirty chronological days (13 July–11 August) select the model. Three ten-day blocks provide stability diagnostics, with winners mean28, mean28, mean7. The following 28 known days (12 August–8 September) calibrate an empirical absolute-error interval. The outcomes for selection and calibration come from a historical snapshot available before the first test origin. Development dates 9–10 September are excluded because their complete outcomes were not available in that snapshot.

Final historical scoring: 11 September–10 October (30 days), using a later snapshot ONLY for outcomes. Changing all later scoring outcomes leaves every forecast, selected model and development score unchanged; check_scoring_isolation.py tests this end to end.

The selected model is mean28. Test MAE 1.2357 recorded starts/day, RMSE 1.5505. Comparator MAE: last observed 1.3667; weekly 1.8667; mean7 1.2429. This small difference does not establish significant superiority.

Nominal 80% intervals use the 24th order statistic of 28 held-out absolute errors, then round count bounds outward. Test coverage was 30/30, with mean width 9.27 recorded starts. Wide intervals are reported honestly. Dependence and distribution shifts prevent a coverage guarantee. Calibration is disjoint from model selection and precedes test scoring.

The test period was already examined during prototype development: this is a revised historical replay, not an untouched holdout. Historical origins are simulations; they are not claims that predictions were issued then. The separate next-day command stores actual creation times, source evidence, input hashes and model-report hash for forecasts issued before their target day.

## Code map

| File | Purpose |
|---|---|
| run.py | Public commands: check, backtest, next-day |
| forecast.py | Eligible inputs, gap-aligned forecasts, selection and scoring |
| data_quality.py | CSV/time validation, duplicates, unknown coverage |
| uncertainty.py | Disjoint chronological calibration and count intervals |
| audit_time.py | Rebuild snapshot-availability audit from cached run metadata |
| issue_forecast.py | Actually timestamped forecast for tomorrow |
| check_scoring_isolation.py | Full-pipeline future-outcome perturbation check |
| test_*.py | Unit tests for time boundaries, corruption, gaps and calibration |

## Evidence and outputs

availability.csv: 90 source-to-origin mappings. snapshots/: historical Kyiv extracts, historical workflow files, source identities and checksums. runs.json: recorded GitHub API run evidence. scoring_data/: later outcomes; never use it as historical forecast input.

results/metrics.json: model choice, periods, scores, calibration and limitations.
results/predictions.csv: candidate predictions, source identities and selection/calibration flags.
results/data_quality.json: quality reports and explicit unknown coverage.
results/model_stability.json: chronological-block comparisons.
results/backtest.png: observed counts, forecasts and empirical intervals.
results/next_day_forecast.json and issued_forecasts/: actual future-day issuance evidence.

Local regional hashes were registered from earlier SHA-pinned extracts during the data-control upgrade. They detect subsequent corruption; they are not independent historical publication evidence.

## Submission

Upload all project files with their folders to GitHub. Include data, requirements, README and tests so reviewers can run it. The GitHub Actions check is supplied but has not run on GitHub until a repository is created and pushed.

Provide the full real AI conversation export separately. AI_LOG_INSTRUCTIONS.txt is an instruction, not that export. REFLECTION_DRAFT.txt describes actual iterations; review it before submitting. Deadline: 12 October 2026, 23:59 Europe/Kyiv. Form: https://forms.gle/y7fdZsg19XdW7Xtc9 . This package has not been submitted and no GitHub repository has been created.
