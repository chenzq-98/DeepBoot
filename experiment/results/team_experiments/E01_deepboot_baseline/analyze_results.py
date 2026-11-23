import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent

with (base / "summary.json").open() as f:
    summary = json.load(f)
with (base / "gpu.log").open() as f:
    gpu = json.load(f)
with (base / "jobinfo.log").open() as f:
    jobs = json.load(f)

jcts = summary["jcts"]
training_jcts = {name: value for name, value in jcts.items() if "infer" not in name}
inference_jcts = {name: value for name, value in jcts.items() if "infer" in name}


def describe(values):
    values = np.asarray(list(values), dtype=float)
    return {
        "count": int(values.size),
        "mean_seconds": float(values.mean()),
        "p50_seconds": float(np.percentile(values, 50)),
        "p75_seconds": float(np.percentile(values, 75)),
        "p95_seconds": float(np.percentile(values, 95)),
        "max_seconds": float(values.max()),
    }

training_stats = describe(training_jcts.values())
inference_stats = describe(inference_jcts.values())
all_stats = describe(jcts.values())

clock = np.asarray(gpu["clock"])
gpu_use = np.asarray(gpu["gpu_use"])
real_gpu_use = np.asarray(gpu["real_gpu_use"])
real_running_gpu_use = np.asarray(gpu["real_running_gpu_use"])

gpu_stats = {
    "samples": int(clock.size),
    "clock_start": int(clock.min()),
    "clock_end": int(clock.max()),
    "allocated_mean_percent": float(gpu_use.mean() / 64 * 100),
    "allocated_peak_percent": float(gpu_use.max() / 64 * 100),
    "real_mean_percent": float(real_gpu_use.mean() / 64 * 100),
    "real_peak_percent": float(real_gpu_use.max() / 64 * 100),
    "real_running_mean_percent": float(real_running_gpu_use.mean() / 64 * 100),
    "real_running_peak_percent": float(real_running_gpu_use.max() / 64 * 100),
}

analysis = {
    "all_jcts": all_stats,
    "training_jcts": training_stats,
    "inference_jcts": inference_stats,
    "gpu": gpu_stats,
    "submitted_jobs": len(jobs["submitted_jobs"]),
}
with (base / "analysis.json").open("w") as f:
    json.dump(analysis, f, indent=2)

# Training JCT CDF
training_values = np.sort(np.asarray(list(training_jcts.values()), dtype=float))
inference_values = np.sort(np.asarray(list(inference_jcts.values()), dtype=float))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(training_values / 60, np.arange(1, training_values.size + 1) / training_values.size,
        color="#c62828", linewidth=2, label="Training jobs")
ax.plot(inference_values / 60, np.arange(1, inference_values.size + 1) / inference_values.size,
        color="#1976d2", linewidth=1.5, alpha=0.8, label="Inference jobs")
ax.set_xlabel("Job completion time / JCT (minutes)")
ax.set_ylabel("CDF")
ax.set_title("DeepBoot small-workload JCT distribution")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(base / "jct_cdf.png", dpi=160)
plt.close(fig)

# GPU utilization timeline
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(clock / 3600, gpu_use / 64 * 100, linewidth=0.8, color="#455a64", label="Allocated GPUs")
ax.plot(clock / 3600, real_gpu_use / 64 * 100, linewidth=0.8, color="#1565c0", label="Real used GPUs")
ax.plot(clock / 3600, real_running_gpu_use / 64 * 100, linewidth=0.8, color="#ef6c00", label="Real running GPUs")
ax.set_ylim(0, 105)
ax.set_xlabel("Simulated time (hours)")
ax.set_ylabel("GPU utilization (%)")
ax.set_title("DeepBoot small-workload GPU utilization")
ax.grid(True, alpha=0.25)
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(base / "gpu_utilization.png", dpi=160)
plt.close(fig)

# Training JCT percentiles
labels = ["Mean", "P50", "P75", "P95", "Max"]
values = [
    training_stats["mean_seconds"],
    training_stats["p50_seconds"],
    training_stats["p75_seconds"],
    training_stats["p95_seconds"],
    training_stats["max_seconds"],
]
values_minutes = np.asarray(values) / 60

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values_minutes, color=["#546e7a", "#1976d2", "#7b1fa2", "#ef6c00", "#c62828"])
ax.set_ylabel("Training JCT (minutes)")
ax.set_title("DeepBoot small-workload training JCT summary")
ax.grid(axis="y", alpha=0.25)
for bar, value in zip(bars, values_minutes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(base / "training_jct_summary.png", dpi=160)
plt.close(fig)

print(json.dumps(analysis, indent=2))
