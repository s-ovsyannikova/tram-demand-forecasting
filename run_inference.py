"""Load a trained model and forecast a configurable inclusive date horizon."""

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

from model_persistence import load_forecast_model
from project_data import load_weather
from service_change_features import load_service_changes
from submission_utils import KEY_COLUMNS, ROOT, write_csv


DEFAULT_MODEL_PATH = ROOT / "subm" / "tram_forecast_model.joblib"


def parse_args():
    """Parse model, weather, horizon, route, and output CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument(
        "--service-changes",
        type=Path,
        help="Optional CSV with planned route changes known for the horizon.",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--routes",
        help="Comma-separated route IDs. Defaults to every trained route.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to subm/inference_START_END.csv.",
    )
    return parser.parse_args()


def dates_between(start, end):
    """Return every date in the inclusive forecast horizon."""
    if end < start:
        raise ValueError("Forecast end must not precede start")
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def parse_routes(value, available_routes):
    """Parse requested routes and verify that the model contains each one."""
    if value is None:
        return sorted(available_routes)
    routes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    missing = sorted(set(routes) - set(available_routes))
    if missing:
        raise ValueError(f"Routes absent from trained model: {missing}")
    return routes


def validate_weather(model, days):
    """Ensure forecast weather covers the horizon when the model requires it."""
    weather_model = getattr(model, "baseline", model)
    if (not weather_model.use_weather
            or weather_model.weather_mode != "forecast_weather"):
        return
    missing = [day for day in days if day not in weather_model.weather]
    if missing:
        raise ValueError(
            "Missing forecast weather for: "
            + ", ".join(day.isoformat() for day in missing)
        )


def predict_horizon(model, routes, days):
    """Build submission-shaped hourly rows for all routes and dates.

    Args:
        model: Loaded ``TramForecastModel``.
        routes: Route identifiers available in the fitted model.
        days: Ordered forecast dates on or after the model cutoff.

    Returns:
        A list of dictionaries with route, date, hour, and rounded prediction.
    """
    keys = [(route, day) for route in routes for day in days]
    if hasattr(model, "predict_days"):
        predictions = model.predict_days(keys)
    else:
        predictions = {
            key: model.predict_day(*key)
            for key in keys
        }

    rows = []
    for route, day in keys:
        prediction, _, _ = predictions[route, day]
        for hour, value in enumerate(prediction):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid prediction for route={route}, day={day}, hour={hour}")
            rows.append({
                "route": route,
                "date": day.isoformat(),
                "hour": hour,
                "prediction": round(value),
            })
    return rows


def main():
    """Load model and weather, validate the horizon, and write predictions."""
    args = parse_args()
    weather, _ = load_weather(args.weather)
    service_events = (
        load_service_changes(args.service_changes)
        if args.service_changes else []
    )
    model = load_forecast_model(args.model, weather, service_events)
    days = dates_between(args.start, args.end)
    if args.start < model.cutoff:
        raise ValueError(
            f"Forecast starts before model cutoff {model.cutoff.isoformat()}")

    routes = parse_routes(args.routes, model.tables)
    validate_weather(model, days)
    rows = predict_horizon(model, routes, days)
    output = args.output or (
        ROOT / "subm" / f"inference_{args.start}_{args.end}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows, KEY_COLUMNS + ["prediction"])
    print(
        f"Wrote {len(rows)} rows for {len(routes)} routes and "
        f"{len(days)} days to {output.resolve()}"
    )


if __name__ == "__main__":
    main()
