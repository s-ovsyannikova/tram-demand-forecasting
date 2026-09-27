"""Persistence helpers for the unified tram forecast model."""

from pathlib import Path

import joblib
import numpy as np


MODEL_FORMAT_VERSION = 1


def save_model(model, path):
    """Save fitted parameters without embedding a weather forecast.

    Args:
        model: Fitted ``TramForecastModel`` instance.
        path: Destination path for the Joblib artifact.

    Returns:
        The resolved artifact ``Path``.
    """
    required = ("cutoff", "tables", "special", "beta")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"Model is not fitted; missing attributes: {missing}")

    state = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "TramForecastModel",
        "config": {
            "season_strength": model.season_strength,
            "trend_strength": model.trend_strength,
            "use_weather": model.use_weather,
            "weather_mode": model.weather_mode,
        },
        "cutoff": model.cutoff,
        "monthly_index": model.monthly_index,
        "beta": float(model.beta),
        "tables": model.tables,
        "special": model.special,
        "weather_coefficients": model.weather_coefficients,
        "weather_centers": getattr(model, "weather_centers", {}),
        "default_weather_center": getattr(
            model, "default_weather_center", np.zeros(4)),
        "weather_scales": getattr(model, "weather_scales", np.ones(4)),
        "weather_training_days": model.weather_training_days,
    }

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path)
    return path


def load_model(path, weather):
    """Restore a fitted model and attach current forecast weather.

    Args:
        path: Path to an artifact created by ``save_model``.
        weather: Mapping from date to four daily weather features. This is kept
            outside the artifact so forecasts can be refreshed without training.

    Returns:
        A fitted ``TramForecastModel`` ready for ``predict_day`` calls.
    """
    from tram_forecast_model import TramForecastModel
    from core_model import TwoStageModel

    state = joblib.load(Path(path))
    if state.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format: {state.get('format_version')}")
    if state.get("model_type") != "TramForecastModel":
        raise ValueError(f"Unexpected model type: {state.get('model_type')}")

    config = state["config"]
    model = TramForecastModel(
        monthly_index=state["monthly_index"],
        weather=weather,
        season_strength=config["season_strength"],
        trend_strength=config["trend_strength"],
        use_weather=config["use_weather"],
        weather_mode=config["weather_mode"],
    )
    model.cutoff = state["cutoff"]
    model.beta = state["beta"]
    model.tables = state["tables"]
    model.special = state["special"]
    model.weather_coefficients = state["weather_coefficients"]
    model.weather_centers = state["weather_centers"]
    model.default_weather_center = state["default_weather_center"]
    model.weather_scales = state["weather_scales"]
    model.weather_training_days = state["weather_training_days"]

    model.baseline = TwoStageModel(
        half_life=None,
        profile_prior_weight=0,
        trend_strength=model.trend_strength,
    )
    model.baseline.cutoff = model.cutoff
    model.baseline.tables = model.tables
    return model


def save_ml_model(model, path):
    """Save an ``OriginResidualMLModel`` as an import-safe state dictionary.

    Args:
        model: Fitted rolling-origin residual model.
        path: Destination path for the Joblib artifact.

    Returns:
        The resolved artifact ``Path``.
    """
    required = ("cutoff", "baseline", "model", "final_state")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(f"ML model is not fitted; missing attributes: {missing}")

    state = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "OriginResidualMLModel",
        "config": {
            "state_window": model.state_window,
            "horizon_days": model.horizon_days,
            "blend": model.blend,
            "n_estimators": model.n_estimators,
        },
        "cutoff": model.cutoff,
        "monthly_index": model.monthly_index,
        "baseline": model.baseline,
        "final_state": model.final_state,
        "estimator": model.model,
        "training_origins": model.training_origins,
        "training_samples": model.training_samples,
    }
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path)
    return path


def load_ml_model(path, weather, service_events=None):
    """Restore a fitted rolling-origin residual model with refreshed weather."""
    from tram_ml_forecast_model import OriginResidualMLModel

    state = joblib.load(Path(path))
    if state.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model format: {state.get('format_version')}")
    if state.get("model_type") != "OriginResidualMLModel":
        raise ValueError(f"Unexpected model type: {state.get('model_type')}")

    config = state["config"]
    model = OriginResidualMLModel(
        monthly_index=state["monthly_index"],
        weather=weather,
        state_window=config["state_window"],
        horizon_days=config["horizon_days"],
        blend=config["blend"],
        n_estimators=config["n_estimators"],
        service_events=service_events,
    )
    model.cutoff = state["cutoff"]
    model.baseline = state["baseline"]
    model.baseline.weather = weather
    model.tables = model.baseline.tables
    model.final_state = state["final_state"]
    model.model = state["estimator"]
    model.training_origins = state["training_origins"]
    model.training_samples = state["training_samples"]
    return model


def load_forecast_model(path, weather, service_events=None):
    """Load either a structural or ML forecast artifact by its model type."""
    state = joblib.load(Path(path))
    model_type = state.get("model_type")
    if model_type == "TramForecastModel":
        return load_model(path, weather)
    if model_type == "OriginResidualMLModel":
        return load_ml_model(path, weather, service_events)
    raise ValueError(f"Unknown model type: {model_type}")
