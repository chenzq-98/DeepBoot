import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "simulator-output"

with (RESULTS / "summary.json").open() as f:
    summary = json.load(f)
with (RESULTS / "gpu.log").open() as f:
    gpu = json.load(f)
with (RESULTS / "jobinfo.log").open() as f:
    jobinfo = json.load(f)
with (ROOT.parent.parent.parent / "workload" / "middle.csv").open(newline="") as f:
    workload = list(csv.DictReader(f))

jcts = summary["jcts"]
training = [value for name, value in jcts.items() if not name.startswith("inference-")]
inference = [value for name, value in jcts.items() if name.startswith("inference-")]


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean_seconds": float(values.mean()),
        "p50_seconds": float(np.percentile(values, 50)),
        "p75_seconds": float(np.percentile(values, 75)),
        "p95_seconds": float(np.percentile(values, 95)),
        "max_seconds": float(values.max()),
    }


clocks = np.asarray(gpu["clock"])
capacity = 16 * 4
gpu_stats = {
    "samples": int(clocks.size),
    "clock_start": int(clocks.min()),
    "clock_end": int(clocks.max()),
}
for field, label in [
    ("gpu_use", "allocated"),
    ("real_gpu_use", "real"),
    ("real_running_gpu_use", "real_running"),
]:
    values = np.asarray(gpu[field], dtype=float)
    gpu_stats[label + "_mean_percent"] = float(values.mean() / capacity * 100)
    gpu_stats[label + "_peak_percent"] = float(values.max() / capacity * 100)

input_counts = {
    "rows": len(workload),
    "training_jobs": sum(not row["name"].startswith("inference-") for row in workload),
    "inference_requests": sum(row["name"].startswith("inference-") for row in workload),
    "inference_application_families": len({
        row["application"].split("-")[1]
        for row in workload
        if row["name"].startswith("inference-")
    }),
}

analysis = {
    "input_workload": input_counts,
    "completed_jobs_in_final_snapshot": len(jobinfo["submitted_jobs"]),
    "completed_jcts": {
        "all": describe(list(jcts.values())),
        "training": describe(training),
        "inference": describe(inference),
    },
    "gpu_utilization": gpu_stats,
    "summary_avgs_field": summary["avgs"],
    "metric_log_samples": len(json.load((RESULTS / "metric.log").open())["clock"]),
}

with (ROOT / "analysis.json").open("w") as f:
    json.dump(analysis, f, indent=2)
print(json.dumps(analysis, indent=2))
