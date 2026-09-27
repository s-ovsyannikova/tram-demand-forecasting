"""Project data loaders and deterministic calendar/weather features."""

import calendar
import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime
from statistics import mean, median

from core_model import ROOT, dates_between, load_history


WEATHER_PATH = ROOT / "open-meteo-55.78N37.58E151m.csv"
TRAFFIC_PATH = ROOT / "62521CSV" / "data-62521-15-09-2026.csv"
WEATHER_FEATURES = [
    "temperature_mean",
    "log1p_rain_mm",
    "log1p_snow_cm",
    "wind_mean",
]
SEASON_YEARS = (2019, 2022, 2023, 2024)
MONTHS = {
    name: number
    for number, name in enumerate([
        "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
        "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
    ], start=1)
}
HOLIDAY_PERIODS = [
    ("2025-02-22", "2025-02-23"),
    ("2025-03-08", "2025-03-09"),
    ("2025-05-01", "2025-05-04"),
    ("2025-05-08", "2025-05-11"),
    ("2025-06-12", "2025-06-15"),
    ("2025-11-02", "2025-11-04"),
    ("2025-12-31", "2025-12-31"),
]
HOLIDAYS = {
    day
    for start, end in HOLIDAY_PERIODS
    for day in dates_between(date.fromisoformat(start), date.fromisoformat(end))
}


def calendar_features(day):
    """Return calendar class, workday flags, and effective weekday for 2025."""
    if day.year != 2025:
        raise ValueError("Calendar is defined only for 2025")
    new_year = day.month == 1 and day.day <= 8
    holiday = day in HOLIDAYS
    working_saturday = day == date(2025, 11, 1)
    kind = "new_year" if new_year else "holiday" if holiday else "ordinary"
    effective_weekday = 4 if working_saturday else day.weekday()
    return {
        "kind": kind,
        "is_holiday": int(new_year or holiday),
        "is_workday": int(
            working_saturday or (day.weekday() < 5 and kind == "ordinary")),
        "is_working_saturday": int(working_saturday),
        "effective_weekday": effective_weekday,
    }


def load_weather(path=WEATHER_PATH):
    """Load hourly Open-Meteo CSV and aggregate four daily weather features."""
    by_day = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        metadata = next(csv.DictReader(stream))
        if (metadata["timezone"] != "Europe/Moscow"
                or int(metadata["utc_offset_seconds"]) != 10800):
            raise ValueError("Weather must use Moscow local time")
        stream.readline()
        reader = csv.DictReader(stream)
        seen = set()
        for row in reader:
            timestamp = datetime.fromisoformat(row["time"])
            if timestamp in seen:
                raise ValueError("Duplicate weather timestamp")
            seen.add(timestamp)
            values = {key: float(value) for key, value in row.items() if key != "time"}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Missing/nonfinite weather: {timestamp}")
            by_day[timestamp.date()].append(values)

    weather = {}
    for day, rows in by_day.items():
        if len(rows) != 24:
            raise ValueError(f"Incomplete weather day: {day}")
        temperature_column = next(
            key for key in rows[0] if key.startswith("temperature_2m"))
        rain = sum(row["rain (mm)"] for row in rows)
        snow = sum(row["snowfall (cm)"] for row in rows)
        if rain < 0 or snow < 0:
            raise ValueError("Negative precipitation")
        weather[day] = [
            mean(row[temperature_column] for row in rows),
            math.log1p(rain),
            math.log1p(snow),
            mean(row["wind_speed_10m (km/h)"] for row in rows),
        ]
    return weather, metadata


def seasonal_index(path=TRAFFIC_PATH):
    """Build a normalized monthly tram-demand index from historical statistics."""
    yearly = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            if not row["Year"].isdigit():
                continue
            if row["Type of transport"].strip() != "Трамвай":
                continue
            year = int(row["Year"])
            if year not in SEASON_YEARS:
                continue
            month = MONTHS[row["Month"].strip()]
            if month in yearly[year]:
                raise ValueError("Duplicate external month")
            value = float(row["Passenger traffic"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid external traffic")
            yearly[year][month] = value / calendar.monthrange(year, month)[1]

    if (set(yearly) != set(SEASON_YEARS)
            or any(set(values) != set(range(1, 13)) for values in yearly.values())):
        raise ValueError("Incomplete historical seasonal data")
    normalized = {
        year: {
            month: value / mean(values.values())
            for month, value in values.items()
        }
        for year, values in yearly.items()
    }
    index = {
        month: median(values[month] for values in normalized.values())
        for month in range(1, 13)
    }
    center = mean(index.values())
    return {month: value / center for month, value in index.items()}


def audited_history():
    """Load supplied labels or an independently rebuilt aggregate when available."""
    audit = ROOT / "subm" / "data_audit" / "report.json"
    if not audit.exists():
        return load_history(), "supplied labels (raw audit unavailable)"
    report = json.loads(audit.read_text(encoding="utf-8"))
    if report["combined_raw"]["mismatch_count"] == 0:
        return load_history(), "supplied labels, verified against combined raw events"

    rebuilt = {}
    with (audit.parent / "labels_rebuilt_raw.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            key = (
                int(row["route"]),
                date.fromisoformat(row["date"]),
                int(row["hour"]),
            )
            rebuilt[key] = float(row["boardings"])
    expected_routes = {1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
    history = {key: value for key, value in rebuilt.items() if key[0] in expected_routes}
    return history, "rebuilt from both raw event files"
