"""Reproducible CPU, memory, latency, and throughput benchmark for inference."""

import argparse
import ctypes
import json
import math
import os
import platform
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from model_persistence import load_forecast_model
from project_data import load_weather
from service_change_features import load_service_changes


ROOT = Path(__file__).resolve().parent


def parse_args():
    """Read benchmark paths and repetition counts from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "subm" / "tram_ml_forecast_model.joblib",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=ROOT / "open-meteo-55.78N37.58E151m.csv",
    )
    parser.add_argument(
        "--service-changes",
        type=Path,
        default=ROOT / "dataset" / "tram_service_changes_2025.csv",
    )
    parser.add_argument("--single-requests", type=int, default=200)
    parser.add_argument("--batch-repeats", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "subm" / "performance_benchmark.json",
    )
    return parser.parse_args()


def memory_bytes():
    """Return current and peak resident memory of this process."""
    if sys.platform == "win32":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        process = get_current_process()
        success = get_memory_info(
            process, ctypes.byref(counters), counters.cb)
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize, counters.PeakWorkingSetSize

    status = Path("/proc/self/status")
    if status.exists():
        values = {}
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                values[key.rstrip(":")] = int(value) * 1024
        return values["VmRSS"], values["VmHWM"]

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    return peak, peak


def total_memory_bytes():
    """Return physical host memory when the operating system exposes it."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys
        return None

    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return page_size * pages


def percentile(values, fraction):
    """Return a nearest-rank percentile for a non-empty sequence."""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def cpu_metrics(cpu_seconds, wall_seconds, logical_cpus):
    """Express consumed CPU time as cores and 2/4-vCPU utilization."""
    cores = cpu_seconds / wall_seconds if wall_seconds else 0.0
    return {
        "cpu_seconds": cpu_seconds,
        "average_cpu_cores": cores,
        "host_cpu_percent": 100.0 * cores / logical_cpus,
        "two_vcpu_percent": 100.0 * cores / 2.0,
        "four_vcpu_percent": 100.0 * cores / 4.0,
    }


def timed_block(action, logical_cpus):
    """Measure wall time, process CPU time, and resident memory for an action."""
    action_start_rss, _ = memory_bytes()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    result = action()
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    rss, peak = memory_bytes()
    metrics = {
        "wall_seconds": wall_seconds,
        "rss_mb": rss / 1024**2,
        "rss_growth_mb": (rss - action_start_rss) / 1024**2,
        "peak_rss_mb": peak / 1024**2,
    }
    metrics.update(cpu_metrics(cpu_seconds, wall_seconds, logical_cpus))
    return result, metrics


def main():
    """Load the model and benchmark single and batched route-day forecasts."""
    args = parse_args()
    if args.single_requests <= 0 or args.batch_repeats <= 0:
        raise ValueError("Repetition counts must be positive")

    logical_cpus = os.cpu_count() or 1
    initial_rss, initial_peak = memory_bytes()
    total_memory = total_memory_bytes()
    weather, _ = load_weather(args.weather)
    events = load_service_changes(args.service_changes)

    model, loading = timed_block(
        lambda: load_forecast_model(args.model, weather, events), logical_cpus)
    routes = sorted(model.tables)
    start = date(2025, 11, 1)
    days = [start + timedelta(days=offset) for offset in range(61)]
    keys = [(route, day) for route in routes for day in days]

    # Warm both the Python path and the tree predictor before latency sampling.
    model.predict_day(routes[0], start)
    model.predict_days(keys)

    latencies_ms = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for index in range(args.single_requests):
        key = keys[index % len(keys)]
        request_start = time.perf_counter()
        model.predict_day(*key)
        latencies_ms.append((time.perf_counter() - request_start) * 1000.0)
    single_wall = time.perf_counter() - wall_start
    single_cpu = time.process_time() - cpu_start
    rss, peak = memory_bytes()
    single = {
        "request_definition": "one route-day with 24 hourly predictions",
        "requests": args.single_requests,
        "wall_seconds": single_wall,
        "requests_per_second": args.single_requests / single_wall,
        "latency_p50_ms": percentile(latencies_ms, 0.50),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "latency_p99_ms": percentile(latencies_ms, 0.99),
        "rss_mb": rss / 1024**2,
        "peak_rss_mb": peak / 1024**2,
    }
    single.update(cpu_metrics(single_cpu, single_wall, logical_cpus))

    batch_start_rss, _ = memory_bytes()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for _ in range(args.batch_repeats):
        model.predict_days(keys)
    batch_wall = time.perf_counter() - wall_start
    batch_cpu = time.process_time() - cpu_start
    rss, peak = memory_bytes()
    route_days = len(keys) * args.batch_repeats
    hourly_values = route_days * 24
    batch = {
        "batch_shape": f"{len(routes)} routes x 61 days",
        "batch_repeats": args.batch_repeats,
        "route_days": route_days,
        "hourly_predictions": hourly_values,
        "wall_seconds": batch_wall,
        "route_days_per_second": route_days / batch_wall,
        "hourly_predictions_per_second": hourly_values / batch_wall,
        "mean_batch_latency_ms": 1000.0 * batch_wall / args.batch_repeats,
        "rss_mb": rss / 1024**2,
        "rss_growth_mb": (rss - batch_start_rss) / 1024**2,
        "peak_rss_mb": peak / 1024**2,
    }
    batch.update(cpu_metrics(batch_cpu, batch_wall, logical_cpus))

    report = {
        "scope": "forecast engine only; file loading included separately; no HTTP",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": logical_cpus,
            "total_memory_gb": (
                total_memory / 1024**3 if total_memory is not None else None),
        },
        "initial_process": {
            "rss_mb": initial_rss / 1024**2,
            "peak_rss_mb": initial_peak / 1024**2,
        },
        "model_loading": loading,
        "single_request": single,
        "batch_inference": batch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
