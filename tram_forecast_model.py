"""Unified tram demand model built as an explicit prediction pipeline.

This module leaves the earlier experimental model files untouched. The public
``TramForecastModel`` combines baseline profiles, calendar effects, trend,
seasonality, and weather without model inheritance.
"""

import json
import math
from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.linear_model import Ridge

from core_model import ROOT, TwoStageModel, daily_vectors, normalize
from model_persistence import save_model
from project_data import (SEASON_YEARS, WEATHER_FEATURES, audited_history,
                          calendar_features, load_weather, seasonal_index)
from submission_utils import save_submission


class TramForecastModel:
    """Forecast hourly boardings with one explicit, configurable pipeline."""

    def __init__(
        self,
        monthly_index,
        weather,
        season_strength=0.5,
        trend_strength=0.5,
        use_weather=True,
        weather_mode="forecast_weather",
    ):
        """Configure the components used by the unified model.

        Args:
            monthly_index: Mapping from month number to an external demand index.
            weather: Mapping from date to temperature, rain, snow, and wind.
            season_strength: Number from 0 to 1, or ``"learned"``. It controls
                how strongly the external monthly index scales demand.
            trend_strength: Multiplier for the recent 28-day demand trend.
            use_weather: Whether to estimate and apply weather effects.
            weather_mode: ``"history_only"`` uses weather only to clean training
                data. ``"forecast_weather"`` also applies supplied future weather.
        """
        if weather_mode not in ("history_only", "forecast_weather"):
            raise ValueError("Unknown weather mode")
        if season_strength != "learned" and not 0 <= float(season_strength) <= 1:
            raise ValueError("season_strength must be between 0 and 1 or 'learned'")

        self.monthly_index = monthly_index
        self.weather = weather
        self.season_strength = season_strength
        self.trend_strength = trend_strength
        self.use_weather = use_weather
        self.weather_mode = weather_mode

    def _fit_calendar(self, history):
        """Fit ordinary weekday patterns and special-day corrections.

        Args:
            history: Mapping ``(route, date, hour)`` to boardings. Ordinary days
                train the baseline; holidays train separate ratios and profiles.
        """
        ordinary = {
            key: value
            for key, value in history.items()
            if calendar_features(key[1])["kind"] == "ordinary"
        }
        self.baseline = TwoStageModel(
            half_life=None,
            profile_prior_weight=0,
            trend_strength=self.trend_strength,
        ).fit(ordinary, self.cutoff)
        self.tables = self.baseline.tables

        route_special = defaultdict(list)
        pooled_special = defaultdict(list)
        for (route, day), vector in daily_vectors(history).items():
            kind = calendar_features(day)["kind"]
            if kind == "ordinary":
                continue
            reference = self.tables[route][0][6]
            ratio = sum(vector) / max(reference, 1e-9)
            route_special[route, kind].append((ratio, normalize(vector)))
            pooled_special[kind].append(ratio)

        self.special = {}
        for route in self.tables:
            for kind in ("new_year", "holiday"):
                records = route_special[route, kind]
                pooled_ratio = (
                    float(np.median(pooled_special[kind]))
                    if pooled_special[kind] else 1.0
                )
                strength = len(records) / (len(records) + 6)
                route_ratio = float(np.median(
                    [record[0] for record in records] or [pooled_ratio]
                ))
                ratio = strength * route_ratio + (1 - strength) * pooled_ratio

                broad_profile = self.tables[route][1][6]
                route_profile = (
                    normalize(np.median(
                        np.asarray([record[1] for record in records]),
                        axis=0,
                    ))
                    if records else broad_profile
                )
                profile = normalize([
                    strength * exact + (1 - strength) * broad
                    for exact, broad in zip(route_profile, broad_profile)
                ])
                self.special[route, kind] = (
                    max(0.05, min(2.0, ratio)),
                    profile,
                )

    def _calendar_prediction(self, route, day):
        """Return calendar- and trend-adjusted total and profile for one day.

        Args:
            route: Route identifier present in the fitted baseline tables.
            day: Historical or forecast date to evaluate.

        Returns:
            ``(total, profile)`` before monthly seasonality and weather are added.
        """
        info = calendar_features(day)
        levels, profiles, _ = self.tables[route]
        if info["kind"] == "ordinary":
            total = levels[info["effective_weekday"]]
            profile = profiles[info["effective_weekday"]]
        else:
            ratio, profile = self.special[route, info["kind"]]
            total = levels[6] * ratio
        return total * self.baseline.trend_factor(route, day), profile

    def _fit_weather(self, history):
        """Fit robust ridge coefficients for daily weather anomalies.

        Args:
            history: Mapping ``(route, date, hour)`` to boardings. Each ordinary
                date becomes one target after route residuals are pooled.
        """
        responses = defaultdict(list)
        for (route, day), vector in daily_vectors(history).items():
            if calendar_features(day)["kind"] != "ordinary":
                continue
            if day not in self.weather:
                raise ValueError(f"No training weather for {day}")
            baseline, _ = self._calendar_prediction(route, day)
            response = math.log(max(sum(vector), 1) / max(baseline, 1))
            responses[day].append(response)

        days = sorted(responses)
        month_numbers = np.asarray([day.month for day in days])
        months = np.unique(month_numbers)
        weather_matrix = np.asarray([self.weather[day] for day in days], dtype=float)
        self.weather_centers = {
            int(month): weather_matrix[month_numbers == month].mean(axis=0)
            for month in months
        }
        self.default_weather_center = weather_matrix.mean(axis=0)

        daily_response = np.asarray([
            np.median(responses[day]) for day in days
        ])
        response_center = {
            int(month): daily_response[month_numbers == month].mean()
            for month in months
        }
        centers = np.vstack([
            self.weather_centers[day.month] for day in days
        ])
        raw_features = weather_matrix - centers
        self.weather_scales = np.maximum(
            np.sqrt(np.mean(raw_features ** 2, axis=0)),
            1e-6,
        )
        features = raw_features / self.weather_scales
        target = daily_response - np.asarray([
            response_center[day.month] for day in days
        ])

        coefficients = np.zeros(len(WEATHER_FEATURES))
        regression = Ridge(alpha=30.0, fit_intercept=False)
        for _ in range(5):
            residuals = target - features @ coefficients
            weights = np.minimum(
                1.0,
                0.10 / np.maximum(np.abs(residuals), 1e-9),
            )
            regression.fit(features, target, sample_weight=weights)
            coefficients = regression.coef_

        self.weather_coefficients = coefficients.tolist()
        self.weather_training_days = len(days)

    def _weather_factor(self, day, apply_weather):
        """Return the fitted multiplicative weather effect for one date.

        Args:
            day: Date whose supplied weather should be evaluated.
            apply_weather: False returns the neutral factor one.

        Returns:
            A weather multiplier clipped to the 0.8--1.25 interval.
        """
        if not self.use_weather or not apply_weather:
            return 1.0
        if day not in self.weather:
            raise ValueError(f"No forecast weather for {day}")

        center = self.weather_centers.get(
            day.month, self.default_weather_center)
        standardized_weather = (
            np.asarray(self.weather[day]) - center
        ) / self.weather_scales
        effect = float(
            np.asarray(self.weather_coefficients) @ standardized_weather
        )
        return math.exp(max(math.log(0.8), min(math.log(1.25), effect)))

    def _fit_season_strength(self, history):
        """Set the fixed seasonal strength or estimate it from monthly residuals.

        Args:
            history: Original, unadjusted boarding history used to compare actual
                monthly demand with the external monthly index.
        """
        if self.season_strength != "learned":
            self.beta = float(self.season_strength)
            return

        residuals = defaultdict(list)
        for (route, day), vector in daily_vectors(history).items():
            if calendar_features(day)["kind"] != "ordinary":
                continue
            expected, _ = self._calendar_prediction(route, day)
            residuals[day.month].append(
                math.log(max(sum(vector), 1) / max(expected, 1))
            )

        predictors = np.asarray([
            math.log(self.monthly_index[month])
            for month in sorted(residuals)
        ])
        targets = np.asarray([
            np.median(residuals[month])
            for month in sorted(residuals)
        ])
        centered_predictors = predictors - predictors.mean()
        centered_targets = targets - targets.mean()
        numerator = centered_predictors @ centered_targets
        denominator = centered_predictors @ centered_predictors + 0.02
        self.beta = max(0.0, min(1.0, numerator / denominator))

    def _adjust_history(self, history):
        """Remove fitted seasonality and weather from training targets.

        Args:
            history: Original mapping ``(route, date, hour)`` to boardings.

        Returns:
            A new history mapping representing neutral month and weather effects.
        """
        adjusted = {}
        for key, value in history.items():
            day = key[1]
            season = self.monthly_index[day.month] ** self.beta
            weather = self._weather_factor(day, apply_weather=self.use_weather)
            adjusted[key] = value / (season * weather)
        return adjusted

    def fit(self, history, cutoff):
        """Fit the complete pipeline in an explicit order.

        Args:
            history: Mapping ``(route, date, hour)`` to observed boardings.
            cutoff: First forecast date; all training dates must precede it.

        Returns:
            The fitted model.
        """
        if not history or any(day >= cutoff for _, day, _ in history):
            raise ValueError("Training dates must strictly precede cutoff")
        self.cutoff = cutoff

        self._fit_calendar(history)
        if self.use_weather:
            self._fit_weather(history)
            weather_adjusted = {
                key: value / self._weather_factor(key[1], apply_weather=True)
                for key, value in history.items()
            }
            self._fit_calendar(weather_adjusted)
        else:
            self.weather_coefficients = [0.0] * len(WEATHER_FEATURES)
            self.weather_training_days = 0

        self._fit_season_strength(history)
        self._fit_calendar(self._adjust_history(history))
        return self

    def predict_day(self, route, day):
        """Predict 24 hourly boarding values for one route and date.

        Args:
            route: Route identifier present in fitted model tables.
            day: Forecast date on or after the model cutoff.

        Returns:
            ``(prediction, total, profile)`` with hourly values, daily total, and
            the normalized profile used to distribute that total.
        """
        if day < self.cutoff:
            raise ValueError("Prediction precedes forecast origin")

        total, profile = self._calendar_prediction(route, day)
        total *= self.monthly_index[day.month] ** self.beta
        total *= self._weather_factor(
            day,
            apply_weather=self.weather_mode == "forecast_weather",
        )
        prediction = (total * np.asarray(profile)).tolist()
        if not all(math.isfinite(value) and value >= 0 for value in prediction):
            raise ValueError("Invalid prediction")
        if not math.isclose(sum(prediction), total, abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError("Daily reconstruction failed")
        return prediction, total, profile


def main():
    """Train the unified final configuration and write separate artifacts."""
    history, history_source = audited_history()
    weather, weather_metadata = load_weather()
    monthly_index = seasonal_index()
    model = TramForecastModel(monthly_index, weather).fit(
        history,
        date(2025, 11, 1),
    )

    output = ROOT / "subm"
    output.mkdir(parents=True, exist_ok=True)
    submission_path = output / "READY_TO_UPLOAD_UNIFIED.csv"
    submission = save_submission(model, submission_path)
    model_path = save_model(model, output / "tram_forecast_model.joblib")
    submission.update({
        "season_strength": model.beta,
        "trend_strength": model.trend_strength,
        "weather_mode": model.weather_mode,
        "weather_coefficients": dict(
            zip(WEATHER_FEATURES, model.weather_coefficients)
        ),
        "weather_training_days": model.weather_training_days,
    })
    report = {
        "model": "TramForecastModel",
        "history_source": history_source,
        "external_years": SEASON_YEARS,
        "weather_metadata": weather_metadata,
        "model_path": str(model_path.relative_to(ROOT)),
        "submission": submission,
    }
    report_path = output / "unified_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Unified submission:", submission_path)


if __name__ == "__main__":
    main()
