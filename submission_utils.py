"""CSV writing and competition submission utilities."""

import csv
import math
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KEY_COLUMNS = ["route", "date", "hour"]
SUBMISSION_ROUTES = (1, 5, 7, 11, 12, 17, 25, 26, 28, 50)
SUBMISSION_START = date(2025, 11, 1)
SUBMISSION_END = date(2025, 12, 31)


def write_csv(path, rows, columns):
    """Write dictionaries to a semicolon-delimited UTF-8 CSV file.

    Args:
        path: Destination ``Path`` object.
        rows: Iterable of dictionaries containing values to write.
        columns: Ordered column names used for the header and each output row.
    """
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _expected_submission_keys():
    """Return every required ``(route, date, hour)`` submission key."""
    day_count = (SUBMISSION_END - SUBMISSION_START).days + 1
    days = [SUBMISSION_START + timedelta(days=i) for i in range(day_count)]
    return {
        (route, day, hour)
        for route in SUBMISSION_ROUTES
        for day in days
        for hour in range(24)
    }


def save_submission(model, path, template_path=None):
    """Generate and validate a competition submission from a fitted model.

    Args:
        model: Fitted model exposing ``tables`` and ``predict_day``.
        path: Destination ``Path`` for the submission CSV.
        template_path: Optional template CSV path. Defaults to the project's
            ``dataset/test_submission.csv`` file.

    Returns:
        Metadata with the row count and routes that retained template values
        because the fitted model had no corresponding route table.
    """
    template_path = template_path or ROOT / "dataset" / "test_submission.csv"
    with template_path.open(encoding="utf-8-sig") as stream:
        template = list(csv.DictReader(stream, delimiter=";"))

    keys = [
        (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
        for row in template
    ]
    expected = _expected_submission_keys()
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Invalid submission grid")

    cache = {}
    result = []
    fallback_routes = set()
    for row, (route, day, hour) in zip(template, keys):
        if route not in model.tables:
            value = float(row["prediction"])
            fallback_routes.add(route)
        else:
            if (route, day) not in cache:
                cache[route, day] = model.predict_day(route, day)[0]
            value = cache[route, day][hour]

        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid submission value")
        result.append({
            "route": route,
            "date": day.isoformat(),
            "hour": hour,
            "prediction": round(value),
        })

    write_csv(path, result, KEY_COLUMNS + ["prediction"])
    return {
        "rows": len(result),
        "template_fallback_routes": sorted(fallback_routes),
    }
