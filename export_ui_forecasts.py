"""Export precomputed structural and ML forecasts for a static web UI."""

import argparse
import copy
import csv
import math
from datetime import date
from pathlib import Path

from model_persistence import load_forecast_model
from project_data import WEATHER_PATH, load_weather
from service_change_features import load_service_changes
from submission_utils import KEY_COLUMNS, ROOT, write_csv


DEFAULT_MODEL_PATH = ROOT / "subm" / "tram_ml_forecast_model.joblib"
DEFAULT_TEMPLATE_PATH = ROOT / "dataset" / "test_submission.csv"
DEFAULT_SERVICE_CHANGES_PATH = ROOT / "dataset" / "tram_service_changes_2025.csv"
DEFAULT_OUTPUT_PATH = ROOT / "subm" / "ui_forecasts.csv"
DEFAULT_SUBMISSION_PATH = ROOT / "subm" / "READY_TO_UPLOAD_DECOMPOSED_ML.csv"


def parse_args():
    """Parse paths for the saved model, source data, and exported UI table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--weather", type=Path, default=WEATHER_PATH)
    parser.add_argument(
        "--service-changes", type=Path, default=DEFAULT_SERVICE_CHANGES_PATH
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--submission-output", type=Path, default=DEFAULT_SUBMISSION_PATH
    )
    return parser.parse_args()


def load_template(path):
    """Load the required route/date/hour grid and its fallback predictions."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def scenario_model(model, days, season_enabled, weather_enabled, events_enabled):
    """Create a lightweight inference view with selected corrections disabled.

    Args:
        model: Loaded ``OriginResidualMLModel``.
        days: Forecast dates whose weather can be replaced by the monthly norm.
        season_enabled: Keep the learned monthly index when true.
        weather_enabled: Keep supplied forecast weather when true.
        events_enabled: Keep planned service-change features when true.

    Returns:
        A shallow model copy sharing the fitted estimator and tables.
    """
    scenario = copy.copy(model)
    scenario.baseline = copy.copy(model.baseline)
    scenario.service_events = model.service_events if events_enabled else []

    if not season_enabled:
        scenario.baseline.monthly_index = {
            month: 1.0 for month in model.baseline.monthly_index
        }
    if not weather_enabled:
        scenario.baseline.weather = dict(model.baseline.weather)
        for day in days:
            center = model.baseline.weather_centers.get(
                day.month, model.baseline.default_weather_center
            )
            scenario.baseline.weather[day] = list(map(float, center))
    return scenario


def export_ui_forecasts(model, template):
    """Build a static multiplicative decomposition for UI sliders.

    Args:
        model: Loaded ``OriginResidualMLModel`` with a structural baseline.
        template: Competition template rows. Its prediction is retained for a
            route that is absent from the trained model.

    Returns:
        Rows with a neutral prediction, three correction coefficients, and the
        final fitted-model prediction.
    """
    if not hasattr(model, "baseline") or not hasattr(model, "predict_days"):
        raise TypeError("UI export requires a fitted ML model with a baseline")

    parsed = [
        (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
        for row in template
    ]
    day_keys = sorted({(route, day) for route, day, _ in parsed})
    trained_keys = [key for key in day_keys if key[0] in model.tables]
    days = sorted({day for _, day in trained_keys})

    # Sequential scenarios make the decomposition reconstruct the final model:
    # neutral -> season -> weather -> planned events.
    neutral_predictions = scenario_model(
        model, days, False, False, False
    ).predict_days(trained_keys)
    season_predictions = scenario_model(
        model, days, True, False, False
    ).predict_days(trained_keys)
    weather_predictions = scenario_model(
        model, days, True, True, False
    ).predict_days(trained_keys)
    final_predictions = model.predict_days(trained_keys)

    coefficients = {}
    for key in trained_keys:
        neutral_total = neutral_predictions[key][1]
        season_total = season_predictions[key][1]
        weather_total = weather_predictions[key][1]
        final_total = final_predictions[key][1]
        if min(neutral_total, season_total, weather_total, final_total) <= 0:
            raise ValueError(f"Non-positive daily prediction for {key}")
        coefficients[key] = (
            season_total / neutral_total,
            weather_total / season_total,
            final_total / weather_total,
        )

    result = []
    for source, (route, day, hour) in zip(template, parsed):
        key = (route, day)
        if route in model.tables:
            base = neutral_predictions[key][0][hour]
            season, weather, event = coefficients[key]
            prediction = final_predictions[key][0][hour]
        else:
            base = prediction = float(source["prediction"])
            season = weather = event = 1.0

        values = (base, season, weather, event, prediction)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                f"Invalid prediction for route={route}, date={day}, hour={hour}"
            )
        result.append({
            "route": route,
            "date": day.isoformat(),
            "hour": hour,
            "base_prediction": round(base, 10),
            "season_coefficient": round(season, 10),
            "weather_coefficient": round(weather, 10),
            "event_coefficient": round(event, 10),
            "prediction": round(prediction, 10),
        })
    return result


def submission_rows(rows):
    """Convert UI rows to the four-column competition submission format."""
    return [
        {
            "route": row["route"],
            "date": row["date"],
            "hour": row["hour"],
            "prediction": round(row["prediction"]),
        }
        for row in rows
    ]


def main():
    """Write the static UI table and its matching competition submission."""
    args = parse_args()
    weather, _ = load_weather(args.weather)
    service_events = load_service_changes(args.service_changes)
    model = load_forecast_model(args.model, weather, service_events)
    rows = export_ui_forecasts(model, load_template(args.template))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = KEY_COLUMNS + [
        "base_prediction",
        "season_coefficient",
        "weather_coefficient",
        "event_coefficient",
        "prediction",
    ]
    write_csv(args.output, rows, columns)
    args.submission_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.submission_output,
        submission_rows(rows),
        KEY_COLUMNS + ["prediction"],
    )
    print(f"Wrote {len(rows)} UI rows to {args.output.resolve()}")
    print(
        f"Wrote {len(rows)} submission rows to "
        f"{args.submission_output.resolve()}"
    )


if __name__ == "__main__":
    main()
