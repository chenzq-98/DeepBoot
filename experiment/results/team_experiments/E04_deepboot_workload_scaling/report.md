# E_04 — DeepBoot Workload Scaling: Middle Workload under Baseline Settings

## Experiment ID and Title

**ID:** E_04  
**Title:** DeepBoot workload scaling with the middle mixed workload

## Objective

Measure DeepBoot's job completion times and GPU utilization on the repository's middle mixed workload while keeping the E01 baseline scheduler, ATS-I/AISS, cluster size, and other settings fixed. The middle workload contains five times as many inference requests as the small workload while retaining the same training jobs.

## Hypothesis

With the number and identity of training jobs held constant, five times as many inference requests may increase inference JCT and can affect training JCT and GPU utilization through additional inference scheduling and lifecycle activity. This single run is descriptive; a change relative to E01 cannot by itself establish that request count alone caused the difference.

## Configuration

| Parameter | Value |
|---|---|
| Scheduler/policy | DeepBoot, `dp` |
| AFE | `0` (disabled) |
| AISS / ATS-I | `1` (enabled) |
| Workload | `experiment/workload/middle.csv` |
| Workload composition | 160 training jobs, 16,000 inference requests; 6 inference application families |
| Comparison workload | `small.csv`: 160 training jobs, 3,200 inference requests; middle has 5× the inference requests |
| Nodes | 16 |
| GPUs per node | 4 (64 total) |
| Training scheduling interval | 60 simulated seconds (`--interval` default) |
| Maximum nodes | Unset; no autoscaling limit requested |
| Inference scheduler | Enabled (`INFER_SCHEDULER=1`, default) |
| Training repair | Enabled (`REPAIR_TRAIN=1`, default) |
| Inference priority | 1 (default) |
| Protect time | 30 seconds; `protect_times=1.0` |
| Interference | 0.0 (default) |
| Random inference allocation | 0 (default) |

The 160 training rows in `small.csv` and `middle.csv` are identical. The inference request rows are not a simple nested prefix: only 525 inference names overlap between the files, and none of those shared-name rows is identical. Thus, E01 is a useful setting/reference comparison but not a paired trace comparison.

## Environment

| Item | Value |
|---|---|
| Branch | `experiment/E_04-deepboot-workload-scaling` |
| Git commit used | `6772f554dd104d908328f63e4ab719184c6ff4d3` |
| Environment | Conda environment `pollux` |
| Python | 3.8.12 |
| NumPy | 1.21.4 |
| pandas | 1.3.2 |
| SciPy | 1.7.2 |
| pymoo | 0.5.0 |

The commit adds the prior E01 report/results and paper PDF; it does not change simulator or scheduler implementation files relative to its parent commit. The dependency versions above were queried from the environment used for this run.

## Methodology

`simulator.py` reads `middle.csv` using pandas, creates jobs from its rows, and attaches trace-backed application performance data from `traces/`. With policy `dp`, it invokes `DeepBoot`; AISS enables ATS-I inference lifecycle behavior and the inference scheduler remains enabled by its default. The simulation clock advances in one-second steps. Training scheduling is triggered on the configured 60-second interval, while inference lifecycle events can also trigger scheduling. The run ends when all workload jobs are complete.

The simulator computes a completed job's JCT as completion time minus submission time. The reported training and inference distributions below are split from the actual `summary.json` JCT dictionary by job name. GPU utilization is sampled once per simulated second and normalized to the 64-GPU cluster. `gpu_use` counts allocated placements; `real_gpu_use` excludes placements whose jobs are rescaling; `real_running_gpu_use` also counts running inference pods and running training placements.

The run reproduces the E01 policy and cluster settings but substitutes the middle workload. The small workload's training rows are identical, while the inference request stream differs; therefore the E01 comparison is observational and does not isolate request count as the sole cause.

## Execution

**Exact simulator command:**

```bash
conda run --no-capture-output -n pollux python simulator.py \
  --workload ./experiment/workload/middle.csv \
  --AFE 0 --AISS 1 --log_file 0 \
  --output experiment/results/team_experiments/E04_deepboot_workload_scaling/simulator-output \
  --min-nodes 16 --num-gpus 4 --policy dp
```

The command was wrapped with `/usr/bin/time -v`, start/end timestamp capture, and stdout/stderr redirection. See [command.txt](command.txt) and the guarded replay helper [run_e04.sh](run_e04.sh).

| Item | Value |
|---|---|
| Start | 2026-11-23 15:27:46 +05:00 |
| End | 2026-11-23 16:09:53 +05:00 |
| Wall-clock runtime | 42 minutes 6.66 seconds (`/usr/bin/time -v`) |
| Exit status | 0 |
| Execution log | Complete captured stdout/stderr is preserved locally as [execution.log.gz](execution.log.gz) (1,908,859,597 bytes); the file is excluded from the Git commit because of its size |
| Execution log SHA-256 | `18f356057fb22236bf80be01edd7ec0b591ef95ee2c890b293c47565416180e8` ([execution_log_integrity.txt](execution_log_integrity.txt)) |
| Timing details | [runtime.txt](runtime.txt) |

## Results

### Job completion time

JCT is in seconds. Percentiles use NumPy's linear percentile calculation over completed jobs.

| Job group | Completed | Mean | P50 | P75 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| All jobs | 16,160 | 40.55 | 12 | 17 | 23 | 55,712 |
| Training | 160 | 2,857.70 | 1,089.50 | 1,650.50 | 7,176.95 | 55,712 |
| Inference | 16,000 | 12.379 | 12 | 17 | 23 | 32 |

The simulator's `summary.json` `avgs` value is **40.5505 seconds**, the mean over all 16,160 training and inference jobs. It is not a training-only average.

### GPU utilization

| Measure | Mean | Peak |
|---|---:|---:|
| Allocated (`gpu_use`) | 82.77% | 100.00% |
| Real (`real_gpu_use`) | 72.41% | 100.00% |
| Real running (`real_running_gpu_use`) | 75.34% | 100.00% |

The GPU timeline contains **77,940 one-second samples**, from simulated second 1 through 77,940.

### Other generated metric

`metric.log` contains empty arrays for clock, goodput, and speedup. The simulator's goodput/speedup calculation is commented out in the main loop, so these metrics are unavailable for this run.

## Interpretation

- All 16,160 jobs completed and produced JCTs. Inference requests had a mean JCT of 12.379 seconds and a maximum of 32 seconds.
- Training mean JCT was 2,857.70 seconds (47.63 minutes). Its P95 was 7,176.95 seconds (119.62 minutes), and the longest observed training JCT was 55,712 seconds (15.48 hours). Training jobs dominate the high JCT tail even though they are only 160 of 16,160 jobs.
- Mean allocated utilization (82.77%) exceeded mean real use (72.41%) by 10.36 percentage points. The simulator distinguishes placements from active use during rescaling. The real-running series includes running inference pods and is 75.34% on average.
- The stored E01 small-workload report reports training mean JCT 2,442.58 seconds, inference mean JCT 7.84 seconds, and mean allocated utilization 94.42%. E_04's corresponding observed values are higher training and inference JCT and lower allocated utilization. The training records are the same, but inference traces differ and E01 was a separate run. Treat this as a workload-level observation, not proof that the fivefold inference request count alone caused the changes.
- The raw all-job mean is much lower than the training mean because inference jobs make up 99% of the workload and have short JCTs.

## Limitations

- This is one deterministic simulator run, not a live GPU-cluster measurement or a multi-seed study.
- The E01 comparison uses the same scheduler source and settings but a different inference request stream; it does not isolate only inference request count.
- The simulator models application performance, rescaling, and inference pod lifecycle using repository traces and state. It does not measure actual container startup, hardware contention, or physical GPU behavior.
- Goodput and speedup are unavailable because `metric.log` is empty and those calculations are disabled in the main simulation loop.
- `jobinfo.log` is a final job snapshot rather than a complete event history. The separate compressed execution log captures the simulator's complete console output.
- The simulator writes into and clears files in its `--output` directory at startup. Reproduction must use a fresh output path; the supplied helper refuses to run if its output directory already exists.
- The complete compressed stdout/stderr log is 1.91 GB. It is retained in the workspace with a SHA-256 checksum but excluded from the pushed branch to keep the Git object set practical; the remaining result files and report are committed.

## Conclusion

DeepBoot completed the middle workload successfully under the E01 baseline policy and cluster settings. It completed 160 training jobs and 16,000 inference requests, with 2,857.70 seconds mean training JCT, 12.379 seconds mean inference JCT, and 82.77% mean allocated-GPU utilization. Relative to the stored small-workload run, the observed training and inference JCTs were higher and allocated utilization lower. Because the inference rows differ between small and middle, the evidence supports a descriptive comparison but does not establish a causal effect from request count alone.

## Reproducibility

1. Check out the E_04 experiment branch and confirm that the simulator source commit recorded in [commit.txt](commit.txt) is `6772f554dd104d908328f63e4ab719184c6ff4d3`. The experiment commit adds only report/results/helpers; it does not change simulator code.
2. Ensure `experiment/workload/middle.csv` and the repository `traces/` directory are present.
3. Choose a new, nonexistent simulator output directory. The simulator deletes contents of an existing output directory when it starts.
4. Run the exact command shown above, changing only `--output` to that fresh directory. Use `/usr/bin/time -v` and redirect both stdout and stderr to preserve the complete log.
5. To reproduce the analysis of the committed E_04 result, run `conda run -n pollux python experiment/results/team_experiments/E04_deepboot_workload_scaling/analyze_results.py`. For a rerun in a different output directory, update the `RESULTS` path in the script to that new directory first.

## Files Generated

- [report.md](report.md) — this experiment report.
- [summary.json](simulator-output/summary.json) — per-job JCT dictionary and overall mean JCT.
- [jobinfo.log](simulator-output/jobinfo.log) — final submitted-job snapshot.
- [gpu.log](simulator-output/gpu.log) — per-second GPU utilization series.
- [metric.log](simulator-output/metric.log) — empty goodput/speedup metric arrays.
- [analysis.json](analysis.json) — derived JCT and GPU utilization statistics.
- [execution.log.gz](execution.log.gz) — complete compressed stdout/stderr execution log, retained locally and excluded from the Git commit due to its size.
- [execution_log_integrity.txt](execution_log_integrity.txt) — SHA-256 checksum for the complete compressed log.
- [runtime.txt](runtime.txt), [start_time.txt](start_time.txt), [end_time.txt](end_time.txt), [exit_status.txt](exit_status.txt) — measured execution metadata.
- [command.txt](command.txt), [environment.txt](environment.txt), [commit.txt](commit.txt) — exact simulator invocation, relevant environment versions, and source commit.
- [analyze_results.py](analyze_results.py), [run_e04.sh](run_e04.sh) — analysis and guarded execution helpers.
