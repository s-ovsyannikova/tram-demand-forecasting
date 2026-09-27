import csv
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def dates_between(start, end):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def load_history():
    history = {}
    for filename in ("labels_day_train.csv", "labels_day_test.csv"):
        with (ROOT / "dataset" / "labels" / filename).open(encoding="utf-8-sig") as stream:

            for row in csv.DictReader(stream, delimiter=";"):
                key = (int(row["route"]), date.fromisoformat(row["date"]), int(row["hour"]))
                value = float(row["boardings"])

                if key in history or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Duplicate key or invalid target: {key}")
                
                history[key] = value
    return history


def daily_vectors(history):
    """
    Labels contain positive event counts only. Missing hours within a recorded
    route-day are treated as zero. Entire missing days remain unknown.
    """
    result = {}
    for (route, day, hour), value in history.items():
        result.setdefault((route, day), [0.0] * 24)[hour] = value
    return result


def weighted_median(values, weights):
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")
    
    pairs = sorted(zip(values, weights))
    halfway = sum(weights) / 2
    cumulative = 0.0

    for index, (value, weight) in enumerate(pairs):
        cumulative += weight
        if math.isclose(cumulative, halfway) and index + 1 < len(pairs):
            return (value + pairs[index + 1][0]) / 2
        if cumulative > halfway:
            return value
        
    raise ValueError("Empty weighted median")


def normalize(values):
    total = sum(values)
    return [v / total for v in values] if total > 0 else [1 / 24] * 24


class TwoStageModel:
    def __init__(self, half_life=56, profile_prior_weight=3.0, trend_strength=0.5):
        """Configure recency weighting, profile shrinkage, and trend strength.

        Args:
            half_life: Number of days after which an observation gets half its
                original weight. ``None`` gives every date equal weight.
            profile_prior_weight: Weight of the broader workday/weekend profile
                when it is blended with an exact weekday profile. Zero disables
                this shrinkage.
            trend_strength: Multiplier applied to the estimated short-term trend.
                Zero disables the trend and one applies it in full.
        """

        if half_life is not None and half_life <= 0:
            raise ValueError("half_life must be positive or None")
        if profile_prior_weight < 0:
            raise ValueError("profile_prior_weight must be non-negative")
        
        self.half_life = half_life
        self.profile_prior_weight = profile_prior_weight
        self.trend_strength = trend_strength

    @staticmethod
    def _route_records(vectors, route):
        """Build positive-total daily records for one route.

        Args:
            vectors: Mapping ``(route, date)`` to a 24-value hourly vector.
            route: Route identifier whose observations should be selected.

        Returns:
            A list of ``(date, hourly_vector, daily_total)`` tuples. Days with
            zero total demand are omitted because they provide no hourly shares.
        """
        return [
            (day, values, sum(values))
            for (record_route, day), values in vectors.items()
            if record_route == route and sum(values) > 0
        ]

    @staticmethod
    def _weekday_records(records, weekday):
        """Select exact-weekday and broader workday/weekend samples.

        Args:
            records: Daily ``(date, hourly_vector, daily_total)`` tuples for one
                route.
            weekday: Python weekday number, from 0 (Monday) to 6 (Sunday).

        Returns:
            ``(exact, broad)`` where ``exact`` contains the requested weekday
            and ``broad`` contains all workdays or all weekend days. Missing
            groups fall back first to the broad group and then to all records.
        """
        exact = [row for row in records if row[0].weekday() == weekday]
        broad = [
            row for row in records
            if (row[0].weekday() >= 5) == (weekday >= 5)
        ]
        broad = broad or records
        exact = exact or broad
        return exact, broad

    def _weights(self, records, cutoff):
        """Calculate recency weights for daily records.

        Args:
            records: Daily tuples whose first item is the observation date.
            cutoff: Forecast origin. Observation age is measured relative to it.

        Returns:
            One positive weight per record. With ``half_life=None`` all weights
            are one; otherwise they halve after each ``half_life`` days.
        """
        if self.half_life is None:
            return [1.0] * len(records)
        return [
            2 ** (-(cutoff - day).days / self.half_life)
            for day, _, _ in records
        ]

    def _fit_weekday(self, exact, broad, cutoff):
        """Estimate a daily level and normalized hourly profile for one weekday.

        Args:
            exact: Records for the specific weekday being estimated.
            broad: Records for the corresponding workday or weekend group.
            cutoff: Forecast origin used to calculate recency weights.

        Returns:
            ``(level, profile)`` where ``level`` is the weighted median daily
            total and ``profile`` is a normalized list of 24 hourly shares.
        """
        exact_weights = self._weights(exact, cutoff)
        broad_weights = self._weights(broad, cutoff)
        level = weighted_median([row[2] for row in exact], exact_weights)
        exact_shape = normalize([
            weighted_median(
                [values[hour] / total for _, values, total in exact],
                exact_weights,
            )
            for hour in range(24)
        ])
        broad_shape = normalize([
            weighted_median(
                [values[hour] / total for _, values, total in broad],
                broad_weights,
            )
            for hour in range(24)
        ])

        exact_weight = sum(exact_weights)
        strength = exact_weight / (exact_weight + self.profile_prior_weight)
        profile = normalize([
            strength * exact_share + (1 - strength) * broad_share
            for exact_share, broad_share in zip(exact_shape, broad_shape)
        ])
        return level, profile

    @staticmethod
    def _estimate_slope(records, levels, cutoff):
        """Estimate a weekday-adjusted short-term demand trend.

        Args:
            records: Daily records for one route.
            levels: Mapping from weekday number to its estimated daily level.
            cutoff: Forecast origin that defines the two historical windows.

        Returns:
            Daily log growth inferred by comparing the most recent 28 days with
            the preceding 28 days. Returns zero if either window has fewer than
            14 observations. Total change is clipped to the 0.8--1.2 range.
        """
        recent, previous = [], []
        for day, _, total in records:
            age = (cutoff - day).days
            ratio = total / max(levels[day.weekday()], 1e-9)
            if age <= 28:
                recent.append(ratio)
            elif age <= 56:
                previous.append(ratio)

        if len(recent) < 14 or len(previous) < 14:
            return 0.0
        change = median(recent) / max(median(previous), 1e-9)
        return math.log(max(0.8, min(1.2, change))) / 28

    def _fit_route(self, records, cutoff):
        """Fit weekday levels, hourly profiles, and trend for one route.

        Args:
            records: Positive-total daily records belonging to one route.
            cutoff: First forecast date; all records must precede it.

        Returns:
            ``(levels, profiles, slope)``. Levels and profiles are dictionaries
            keyed by weekday, while slope is daily logarithmic growth.
        """
        levels, profiles = {}, {}
        for weekday in range(7):
            exact, broad = self._weekday_records(records, weekday)
            levels[weekday], profiles[weekday] = self._fit_weekday(
                exact, broad, cutoff)
        slope = self._estimate_slope(records, levels, cutoff)
        return levels, profiles, slope

    def fit(self, history, cutoff):
        """Fit model tables independently for every route.

        Args:
            history: Mapping ``(route, date, hour)`` to observed boardings.
            cutoff: First forecast date. Every training date must be earlier.

        Returns:
            The fitted model itself. ``self.tables[route]`` stores weekday
            levels, hourly profiles, and the route's short-term trend.
        """
        if not history or any(day >= cutoff for _, day, _ in history):
            raise ValueError("Training data must strictly precede forecast origin")

        self.cutoff = cutoff
        self.tables = {}
        vectors = daily_vectors(history)
        routes = sorted({route for route, _ in vectors})

        for route in routes:
            records = self._route_records(vectors, route)
            if not records:
                raise ValueError(f"Route {route} has no positive training days")
            self.tables[route] = self._fit_route(records, cutoff)
        return self

    def trend_factor(self, route, day):
        """Return the damped trend multiplier for a route and forecast date.

        Args:
            route: Route identifier present in the fitted model tables.
            day: Date for which the multiplier is requested. Historical dates
                receive a neutral factor of one.

        Returns:
            A positive multiplier for the route's baseline daily level.
        """
        if day < self.cutoff:
            return 1.0
        slope = self.tables[route][2]
        horizon = (day - self.cutoff).days + 1
        # Damping limits extrapolation to 28 effective days, even for long horizons.
        effective_days = 28 * (1 - math.exp(-horizon / 28))
        return math.exp(self.trend_strength * slope * effective_days)

    def predict_day(self, route, day):
        """Predict all 24 hours for one route and date.

        Args:
            route: Route identifier present in the fitted model tables.
            day: Forecast date on or after the model cutoff.

        Returns:
            ``(prediction, total, profile)`` containing 24 hourly values, their
            daily total, and the normalized hourly profile used to split it.
        """
        if day < self.cutoff:
            raise ValueError("Prediction must be after training cutoff")
        
        levels, profiles, _ = self.tables[route]
        total = levels[day.weekday()] * self.trend_factor(route, day)
        profile = profiles[day.weekday()]
        prediction = [total * share for share in profile]

        if not all(math.isfinite(p) and p >= 0 for p in prediction):
            raise ValueError("Invalid prediction")
        if not math.isclose(sum(prediction), total, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("Hourly predictions do not reconstruct daily total")
        
        return prediction, total, profile


def evaluate(model, truth):
    daily = daily_vectors(truth)
    errors = defaultdict(lambda: [0.0, 0.0])
    monthly = defaultdict(lambda: [0.0, 0.0, 0.0])
    rows = []
    hourly_error = daily_error = profile_error = actual_sum = 0.0

    for (route, day), values in sorted(daily.items()):
        prediction, total, profile = model.predict_day(route, day)
        actual_daily = sum(values)
        daily_error += abs(actual_daily - total)
        actual_sum += actual_daily

        for hour, (actual, pred, share) in enumerate(zip(values, prediction, profile)):
            error = abs(actual - pred)
            hourly_error += error
            profile_error += abs(actual - actual_daily * share)
            errors[route][0] += error
            errors[route][1] += actual
            month = day.strftime("%Y-%m")
            monthly[month][0] += error
            monthly[month][1] += actual
            monthly[month][2] += pred
            rows.append({
                    "route": route, "date": day.isoformat(), "hour": hour,
                    "boardings": actual, "prediction": pred
                })
    if actual_sum <= 0:
        raise ValueError("No positive validation targets")
    return {
        "hourly_wape": hourly_error / actual_sum,
        "score": max(0.0, 1 - hourly_error / actual_sum),
        "daily_wape": daily_error / actual_sum,
        "profile_oracle_wape": profile_error / actual_sum,
        "rows": len(rows),
        "by_route": {str(r): err / total for r, (err, total) in errors.items()},
        "by_month": {m: {"wape": err / total, "predicted_to_actual": pred / total}
                     for m, (err, total, pred) in monthly.items()},
    }, rows
