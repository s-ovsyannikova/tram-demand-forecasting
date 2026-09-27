"""Route-specific features from planned service changes."""

import csv
from datetime import date
from pathlib import Path

import numpy as np


SERVICE_FEATURE_NAMES = [
    "service_changed_fraction",
    "service_suspended",
    "service_shortened_fraction",
    "service_rerouted_fraction",
    "service_late_night_fraction",
    "service_severity_mean",
    "service_severity_max",
]
SUPPORTED_STATUSES = ("suspended", "shortened", "rerouted", "late_night")


def load_service_changes(path):
    """Load and validate the published route service-change calendar.

    Args:
        path: Semicolon-delimited CSV containing route, date range, hour range,
            day filter, status, and severity.

    Returns:
        A list of dictionaries with parsed dates and numeric fields.
    """
    events = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            event = dict(row)
            event["route"] = int(row["route"])
            event["start_date"] = date.fromisoformat(row["start_date"])
            event["end_date"] = date.fromisoformat(row["end_date"])
            event["start_hour"] = int(row["start_hour"])
            event["end_hour"] = int(row["end_hour"])
            event["severity"] = float(row["severity"])
            if event["end_date"] < event["start_date"]:
                raise ValueError(f"Service event ends before it starts: {row['event_id']}")
            if not 0 <= event["start_hour"] <= event["end_hour"] <= 23:
                raise ValueError(f"Invalid service hours: {row['event_id']}")
            if event["day_filter"] not in ("all", "weekday", "weekend"):
                raise ValueError(f"Invalid day filter: {row['event_id']}")
            if event["status"] not in SUPPORTED_STATUSES:
                raise ValueError(f"Invalid service status: {row['event_id']}")
            if not 0 <= event["severity"] <= 1:
                raise ValueError(f"Invalid service severity: {row['event_id']}")
            events.append(event)
    return events


def service_features(events, route, day):
    """Build daily features known from the plan for a route and date.

    Args:
        events: Parsed events returned by ``load_service_changes``.
        route: Route identifier.
        day: Target date for training or prediction.

    Returns:
        Seven numeric features describing affected-hour fractions, status, and
        severity. Overlapping events use the maximum hourly severity.
    """
    changed = np.zeros(24)
    statuses = {status: np.zeros(24) for status in SUPPORTED_STATUSES}
    severity = np.zeros(24)

    for event in events:
        if event["route"] != route:
            continue
        if not event["start_date"] <= day <= event["end_date"]:
            continue
        if event["day_filter"] == "weekday" and day.weekday() >= 5:
            continue
        if event["day_filter"] == "weekend" and day.weekday() < 5:
            continue
        hours = slice(event["start_hour"], event["end_hour"] + 1)
        changed[hours] = 1.0
        statuses[event["status"]][hours] = 1.0
        severity[hours] = np.maximum(severity[hours], event["severity"])

    return [
        float(changed.mean()),
        float(statuses["suspended"].max()),
        float(statuses["shortened"].mean()),
        float(statuses["rerouted"].mean()),
        float(statuses["late_night"].mean()),
        float(severity.mean()),
        float(severity.max()),
    ]
