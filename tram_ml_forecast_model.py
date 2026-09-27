"""Rolling-origin ML residual model over the unified structural forecast."""

import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from core_model import ROOT, daily_vectors, evaluate
from model_persistence import save_ml_model
from project_data import (audited_history, calendar_features, load_weather,
                          seasonal_index)
from service_change_features import load_service_changes, service_features
from submission_utils import save_submission
from tram_forecast_model import TramForecastModel


def historical_prediction(model, route, day):
    """Reconstruct a structural forecast for a historical date.

    Args:
        model: Fitted ``TramForecastModel`` whose cutoff follows ``day``.
        route: Route identifier present in the model tables.
        day: Historical date to reconstruct.

    Returns:
        ``(total, profile)`` including calendar, seasonality, and weather but no
        forward trend, because historical dates receive a neutral trend factor.
    """
    total, profile = model._calendar_prediction(route, day)
    total *= model.monthly_index[day.month] ** model.beta
    total *= model._weather_factor(day, apply_weather=model.use_weather)
    return total, profile


def build_origin_state(vectors, baseline, origin, window=28):
    """Summarize demand observed immediately before a forecast origin.

    Args:
        vectors: Mapping ``(route, date)`` to 24 observed hourly values.
        baseline: Structural model fitted using only dates before ``origin``.
        origin: Historical or real forecast origin.
        window: Number of preceding days used for recent-state estimates.

    Returns:
        Route-level dictionaries containing shrunk route/global actual-to-model
        ratios and the number of recent observations. State is frozen throughout
        the forecast horizon, preventing future leakage.
    """
    start = origin - timedelta(days=window)
    route_ratios = defaultdict(list)
    global_actual = 0.0
    global_expected = 0.0

    for (route, day), values in vectors.items():
        if not start <= day < origin or route not in baseline.tables:
            continue
        actual = sum(values)
        if actual <= 0:
            continue
        expected, _ = historical_prediction(baseline, route, day)
        if expected <= 0:
            continue
        route_ratios[route].append(max(0.5, min(1.5, actual / expected)))
        global_actual += actual
        global_expected += expected

    global_ratio = global_actual / global_expected if global_expected > 0 else 1.0
    global_ratio = max(0.7, min(1.3, global_ratio))
    state = {}
    for route in baseline.tables:
        ratios = route_ratios.get(route, [])
        route_ratio = float(np.median(ratios)) if ratios else global_ratio
        strength = len(ratios) / (len(ratios) + 7.0)
        shrunk_ratio = strength * route_ratio + (1.0 - strength) * global_ratio
        state[route] = {
            "route_ratio": max(0.7, min(1.3, shrunk_ratio)),
            "global_ratio": global_ratio,
            "n_recent": len(ratios),
        }
    return state


def month_starts_before(cutoff):
    """Return eligible 2025 month starts used as pseudo forecast origins."""
    return [date(2025, month, 1) for month in range(3, cutoff.month)]


def origin_features(
        route, day, origin, baseline_total, baseline, state, service_events):
    """Build leakage-safe features for one route, day, and forecast origin.

    Args:
        route: Route identifier.
        day: Target forecast date.
        origin: Date at which all recent state is frozen.
        baseline_total: Structural daily forecast before ML correction.
        baseline: Fitted structural model providing season and weather features.
        state: Output of ``build_origin_state`` for this origin.
        service_events: Published route changes available for feature generation.

    Returns:
        Numeric calendar, weather, structural, horizon, and recent-state features.
    """
    calendar = calendar_features(day)
    day_of_year = day.timetuple().tm_yday
    year_angle = 2.0 * math.pi * day_of_year / 365.0
    week_angle = 2.0 * math.pi * calendar["effective_weekday"] / 7.0
    horizon = (day - origin).days
    horizon_scaled = min(max(horizon, 0) / 60.0, 1.0)
    route_state = state[route]
    weather = baseline.weather.get(day)
    if baseline.use_weather and weather is None:
        raise ValueError(f"No weather features for {day}")
    weather = weather or [0.0, 0.0, 0.0, 0.0]

    return [
        float(route),
        float(day.month),
        float(day.day),
        float(day_of_year),
        float(calendar["effective_weekday"]),
        float(calendar["is_holiday"]),
        float(calendar["is_workday"]),
        float(calendar["is_working_saturday"]),
        math.sin(year_angle),
        math.cos(year_angle),
        math.sin(week_angle),
        math.cos(week_angle),
        float(baseline.monthly_index[day.month]),
        math.log(max(baseline_total, 1.0)),
        *map(float, weather),
        *service_features(service_events, route, day),
        float(route_state["route_ratio"]),
        float(route_state["global_ratio"]),
        min(float(route_state["n_recent"]) / 28.0, 1.0),
        float(horizon),
        horizon_scaled,
        (route_state["route_ratio"] - 1.0) * (1.0 - horizon_scaled),
        (route_state["global_ratio"] - 1.0) * (1.0 - horizon_scaled),
    ]


class OriginResidualMLModel:
    """Correct unified structural daily totals with rolling-origin ExtraTrees."""

    def __init__(
        self,
        monthly_index,
        weather,
        state_window=28,
        horizon_days=61,
        blend=0.5,
        n_estimators=600,
        service_events=None,
    ):
        """Configure structural baseline and residual learner.

        Args:
            monthly_index: External monthly demand index.
            weather: Historical and forecast daily weather features.
            state_window: Days before each origin used for recent state.
            horizon_days: Maximum pseudo-forecast horizon used for ML samples.
            blend: Strength of the predicted log-residual correction.
            n_estimators: Number of trees in ``ExtraTreesRegressor``.
            service_events: Parsed, pre-announced route service changes.
        """
        self.monthly_index = monthly_index
        self.weather = weather
        self.state_window = state_window
        self.horizon_days = horizon_days
        self.blend = blend
        self.n_estimators = n_estimators
        self.service_events = service_events or []

    def _new_baseline(self):
        """Create the exact structural configuration used at every origin."""
        return TramForecastModel(
            monthly_index=self.monthly_index,
            weather=self.weather,
            season_strength=0.5,
            trend_strength=0.5,
            use_weather=True,
            weather_mode="forecast_weather",
        )

    def fit(self, history, cutoff):
        """Fit final baseline and ExtraTrees on historical pseudo-origins.

        Args:
            history: Mapping ``(route, date, hour)`` to observed boardings.
            cutoff: Real forecast origin. ML targets and state for every training
                example are constructed without observations at or after its
                corresponding pseudo-origin.

        Returns:
            The fitted model.
        """
        self.cutoff = cutoff
        self.baseline = self._new_baseline().fit(history, cutoff)
        self.tables = self.baseline.tables
        all_vectors = daily_vectors(history)
        self.final_state = build_origin_state(
            all_vectors, self.baseline, cutoff, self.state_window)

        features = []
        targets = []
        self.training_origins = month_starts_before(cutoff)
        for origin in self.training_origins:
            origin_history = {
                key: value for key, value in history.items() if key[1] < origin
            }
            if not origin_history:
                continue
            origin_baseline = self._new_baseline().fit(origin_history, origin)
            origin_vectors = daily_vectors(origin_history)
            state = build_origin_state(
                origin_vectors, origin_baseline, origin, self.state_window)
            horizon_end = min(
                cutoff - timedelta(days=1),
                origin + timedelta(days=self.horizon_days - 1),
            )

            for (route, day), actual_vector in sorted(all_vectors.items()):
                if not origin <= day <= horizon_end:
                    continue
                if route not in origin_baseline.tables:
                    continue
                actual_total = sum(actual_vector)
                if actual_total <= 0:
                    continue
                _, baseline_total, _ = origin_baseline.predict_day(route, day)
                if baseline_total <= 0:
                    continue
                features.append(origin_features(
                    route,
                    day,
                    origin,
                    baseline_total,
                    origin_baseline,
                    state,
                    self.service_events,
                ))
                residual = math.log(actual_total) - math.log(baseline_total)
                targets.append(max(-0.8, min(0.8, residual)))

        if not features:
            raise RuntimeError("No rolling-origin ML training samples")
        self.training_samples = len(features)
        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            np.asarray(features, dtype=float),
            np.asarray(targets, dtype=float),
        )
        return self

    def predict_day(self, route, day):
        """Apply an ML residual correction to the structural daily forecast."""
        return self.predict_days([(route, day)])[route, day]

    def predict_days(self, keys):
        """Predict many route-day pairs with one vectorized ExtraTrees call.

        Args:
            keys: Iterable of ``(route, date)`` pairs.

        Returns:
            Mapping from each key to ``(hourly_prediction, total, profile)``.
        """
        keys = list(keys)
        baseline_predictions = []
        feature_rows = []
        for route, day in keys:
            _, baseline_total, profile = self.baseline.predict_day(route, day)
            baseline_predictions.append((baseline_total, profile))
            feature_rows.append(origin_features(
                route,
                day,
                self.cutoff,
                baseline_total,
                self.baseline,
                self.final_state,
                self.service_events,
            ))

        residuals = self.model.predict(np.asarray(feature_rows, dtype=float))
        residuals = np.clip(residuals, -0.5, 0.5)
        result = {}
        for key, (baseline_total, profile), residual in zip(
                keys, baseline_predictions, residuals):
            total = baseline_total * math.exp(self.blend * float(residual))
            prediction = (total * np.asarray(profile)).tolist()
            result[key] = (prediction, total, profile)
        return result


def evaluate_ml(model, truth):
    """Evaluate ML predictions after batching the expensive tree inference."""
    keys = sorted(daily_vectors(truth))
    predictions = model.predict_days(keys)

    class CachedPredictions:
        def predict_day(self, route, day):
            return predictions[route, day]

    return evaluate(CachedPredictions(), truth)


def main():
    """Backtest structural and ML forecasts, then create separate ML artifacts."""
    history, history_source = audited_history()
    weather, _ = load_weather()
    monthly_index = seasonal_index()
    service_events = load_service_changes(
        ROOT / "dataset" / "tram_service_changes_2025.csv")
    folds = [
        (date(2025, 5, 1), date(2025, 6, 30)),
        (date(2025, 7, 1), date(2025, 8, 31)),
        (date(2025, 9, 1), date(2025, 10, 31)),
    ]
    report = {"history_source": history_source, "folds": {}}

    for start, end in folds:
        training = {key: value for key, value in history.items() if key[1] < start}
        truth = {
            key: value
            for key, value in history.items()
            if start <= key[1] <= end
        }
        model = OriginResidualMLModel(
            monthly_index, weather, service_events=service_events).fit(
                training, start)
        structural_metrics, _ = evaluate(model.baseline, truth)
        ml_metrics, _ = evaluate_ml(model, truth)
        report["folds"][start.isoformat()] = {
            "structural": structural_metrics,
            "origin_residual_ml": ml_metrics,
            "training_origins": [day.isoformat() for day in model.training_origins],
            "training_samples": model.training_samples,
        }
        print(
            start,
            f"structural={structural_metrics['hourly_wape']:.6f}",
            f"ml={ml_metrics['hourly_wape']:.6f}",
            f"samples={model.training_samples}",
            flush=True,
        )

    final_model = OriginResidualMLModel(
        monthly_index, weather, service_events=service_events).fit(
        history, date(2025, 11, 1))
    output = ROOT / "subm"
    output.mkdir(parents=True, exist_ok=True)
    submission_path = output / "READY_TO_UPLOAD_UNIFIED_ML.csv"
    submission = save_submission(final_model, submission_path)
    model_path = output / "tram_ml_forecast_model.joblib"
    save_ml_model(final_model, model_path)
    report.update({
        "model": "OriginResidualMLModel",
        "submission": submission,
        "training_samples": final_model.training_samples,
        "model_path": str(model_path.relative_to(ROOT)),
        "submission_path": str(submission_path.relative_to(ROOT)),
    })
    (output / "unified_ml_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("ML submission:", submission_path)


if __name__ == "__main__":
    main()
