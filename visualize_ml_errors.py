"""Build validation diagnostics for the unified rolling-origin ML model."""

import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_model import ROOT, daily_vectors
from project_data import audited_history, load_weather, seasonal_index
from service_change_features import load_service_changes, service_features
from tram_ml_forecast_model import OriginResidualMLModel


FOLDS = [
    (date(2025, 5, 1), date(2025, 6, 30)),
    (date(2025, 7, 1), date(2025, 8, 31)),
    (date(2025, 9, 1), date(2025, 10, 31)),
]


def validation_rows(history, weather, monthly_index, service_events):
    """Train each historical fold and return hourly structural/ML predictions."""
    rows = []
    for start, end in FOLDS:
        training = {key: value for key, value in history.items() if key[1] < start}
        truth = {
            key: value
            for key, value in history.items()
            if start <= key[1] <= end
        }
        daily_truth = daily_vectors(truth)
        keys = sorted(daily_truth)
        model = OriginResidualMLModel(
            monthly_index,
            weather,
            service_events=service_events,
        ).fit(training, start)
        ml_predictions = model.predict_days(keys)

        for route, day in keys:
            actual = daily_truth[route, day]
            structural, _, _ = model.baseline.predict_day(route, day)
            ml, _, _ = ml_predictions[route, day]
            service = service_features(service_events, route, day)
            if service[1] > 0:
                service_status = "suspended"
            elif service[2] > 0:
                service_status = "shortened"
            elif service[3] > 0:
                service_status = "rerouted"
            elif service[4] > 0:
                service_status = "late_night"
            else:
                service_status = "normal"
            for hour in range(24):
                rows.append({
                    "fold": start.strftime("%b-%b"),
                    "fold_start": start.isoformat(),
                    "route": route,
                    "date": day.isoformat(),
                    "hour": hour,
                    "actual": actual[hour],
                    "structural": structural[hour],
                    "ml": ml[hour],
                    "service_status": service_status,
                    "service_severity": service[6],
                })
        print(f"Prepared {start} .. {end}", flush=True)
    return pd.DataFrame(rows)


def aggregate_errors(data, group):
    """Aggregate actual demand and absolute ML error by one dimension."""
    result = data.groupby(group, as_index=False).agg(
        actual=("actual", "sum"),
        predicted=("ml", "sum"),
        absolute_error=("ml_error", "sum"),
    )
    total_actual = data["actual"].sum()
    result["wape"] = result["absolute_error"] / result["actual"]
    result["wape_contribution"] = result["absolute_error"] / total_actual
    result["passenger_share"] = result["actual"] / total_actual
    result["bias"] = (result["predicted"] - result["actual"]) / result["actual"]
    return result


def detailed_error_tables(data):
    """Return service-segment and worst route-day diagnostics."""
    data = data.copy()
    data["ml_error"] = (data["actual"] - data["ml"]).abs()
    service = aggregate_errors(data, "service_status").sort_values(
        "wape_contribution", ascending=False)

    route_days = data.groupby(
        ["route", "date", "fold_start", "service_status"], as_index=False
    ).agg(
        actual=("actual", "sum"),
        predicted=("ml", "sum"),
        absolute_error=("ml_error", "sum"),
        service_severity=("service_severity", "max"),
    )
    total_actual = data["actual"].sum()
    route_days["wape"] = (
        route_days["absolute_error"] / route_days["actual"].replace(0, np.nan)
    )
    route_days["wape_contribution"] = (
        route_days["absolute_error"] / total_actual)
    route_days["bias"] = (
        (route_days["predicted"] - route_days["actual"])
        / route_days["actual"].replace(0, np.nan)
    )
    worst = route_days.sort_values(
        "wape_contribution", ascending=False).head(50)
    return service, worst


def write_csv(path, frame):
    """Write an analysis DataFrame using the project's semicolon CSV format."""
    frame.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_MINIMAL)


def build_dashboard(data, output):
    """Render fold, route, hour, and route-hour error diagnostics."""
    data = data.copy()
    data["ml_error"] = (data["actual"] - data["ml"]).abs()
    data["structural_error"] = (data["actual"] - data["structural"]).abs()
    total_actual = data["actual"].sum()
    overall_wape = data["ml_error"].sum() / total_actual

    folds = data.groupby("fold_start", as_index=False).agg(
        actual=("actual", "sum"),
        structural_error=("structural_error", "sum"),
        ml_error=("ml_error", "sum"),
    ).sort_values("fold_start")
    folds["structural_wape"] = folds["structural_error"] / folds["actual"]
    folds["ml_wape"] = folds["ml_error"] / folds["actual"]
    folds["period"] = ["Май-июнь", "Июль-август", "Сентябрь-октябрь"]

    routes = aggregate_errors(data, "route").sort_values("wape", ascending=False)
    hours = aggregate_errors(data, "hour")
    heatmap = data.pivot_table(
        index="route",
        columns="hour",
        values=["actual", "ml_error"],
        aggfunc="sum",
        fill_value=0,
    )
    local_wape = (
        heatmap["ml_error"] / heatmap["actual"].replace(0, np.nan)
    ).fillna(0)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#C7CDD4",
        "axes.labelcolor": "#27313A",
        "xtick.color": "#46515C",
        "ytick.color": "#46515C",
    })
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#F7F8FA")
    fig.subplots_adjust(
        left=0.07,
        right=0.94,
        bottom=0.07,
        top=0.86,
        hspace=0.34,
        wspace=0.28,
    )
    for axis in axes.flat:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#E8EBEF", linewidth=0.8)
        axis.set_axisbelow(True)

    x = np.arange(len(folds))
    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        100 * folds["structural_wape"],
        width,
        label="Structural",
        color="#7A8793",
    )
    axes[0, 0].bar(
        x + width / 2,
        100 * folds["ml_wape"],
        width,
        label="ML residual",
        color="#138A72",
    )
    axes[0, 0].set_xticks(x, folds["period"])
    axes[0, 0].set_ylabel("WAPE, %")
    axes[0, 0].set_title("Ошибка по временным фолдам")
    axes[0, 0].legend(frameon=False)
    for position, value in zip(x + width / 2, folds["ml_wape"]):
        axes[0, 0].text(position, 100 * value + 0.25, f"{100 * value:.1f}%", ha="center")

    route_labels = routes["route"].astype(str)
    route_wape = 100 * routes["wape"]
    colors = ["#D1495B" if value >= route_wape.median() else "#E5A84B" for value in route_wape]
    axes[0, 1].barh(route_labels, route_wape, color=colors)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Локальный WAPE, %")
    axes[0, 1].set_ylabel("Маршрут")
    axes[0, 1].set_title("Где ML ошибается по маршрутам")
    for index, (_, row) in enumerate(routes.iterrows()):
        axes[0, 1].text(
            100 * row["wape"] + 0.2,
            index,
            f"вклад {100 * row['wape_contribution']:.1f} п.п.",
            va="center",
            fontsize=8,
        )

    axes[1, 0].plot(
        hours["hour"],
        100 * hours["wape_contribution"],
        marker="o",
        linewidth=2,
        color="#D1495B",
        label="Вклад в WAPE",
    )
    axes[1, 0].plot(
        hours["hour"],
        100 * hours["passenger_share"],
        marker="s",
        linewidth=1.7,
        color="#2878B5",
        label="Доля пассажиров",
    )
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].set_xlabel("Час")
    axes[1, 0].set_ylabel("Доля общего потока, %")
    axes[1, 0].set_title("Из каких часов складывается ошибка")
    axes[1, 0].legend(frameon=False)

    image = axes[1, 1].imshow(
        100 * local_wape.to_numpy(),
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=float(np.nanpercentile(100 * local_wape.to_numpy(), 95)),
    )
    axes[1, 1].grid(False)
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].set_xticklabels(range(0, 24, 2))
    axes[1, 1].set_yticks(range(len(local_wape.index)))
    axes[1, 1].set_yticklabels(local_wape.index)
    axes[1, 1].set_xlabel("Час")
    axes[1, 1].set_ylabel("Маршрут")
    axes[1, 1].set_title("Локальный WAPE: маршрут × час")
    colorbar = fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.03)
    colorbar.set_label("WAPE, %")

    fig.suptitle(
        "Диагностика ошибок OriginResidualMLModel",
        y=0.975,
        fontsize=18,
        fontweight="bold",
        color="#17212B",
    )
    fig.text(
        0.5,
        0.935,
        f"WAPE = Σ|факт − прогноз| / Σфакт = {overall_wape:.4f} ({100 * overall_wape:.2f}%)",
        ha="center",
        fontsize=11,
        color="#46515C",
    )
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return folds, routes, hours, local_wape


def main():
    """Train validation folds and save the dashboard plus aggregate CSV files."""
    history, _ = audited_history()
    weather, _ = load_weather()
    service_events = load_service_changes(
        ROOT / "dataset" / "tram_service_changes_2025.csv")
    data = validation_rows(
        history, weather, seasonal_index(), service_events)
    output = ROOT / "subm" / "ml_error_analysis.png"
    folds, routes, hours, heatmap = build_dashboard(data, output)
    service, worst_days = detailed_error_tables(data)
    write_csv(ROOT / "subm" / "ml_error_by_fold.csv", folds)
    write_csv(ROOT / "subm" / "ml_error_by_route.csv", routes)
    write_csv(ROOT / "subm" / "ml_error_by_hour.csv", hours)
    heatmap.to_csv(ROOT / "subm" / "ml_error_route_hour_wape.csv", sep=";")
    write_csv(ROOT / "subm" / "ml_error_by_service.csv", service)
    write_csv(ROOT / "subm" / "ml_error_worst_route_days.csv", worst_days)
    print("Saved:", output)


if __name__ == "__main__":
    main()
