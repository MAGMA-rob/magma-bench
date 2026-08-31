# magma-bench
Official repository for MAGMA-BENCH. Evaluate your agents on highly-interactive robotic task, controlled via natural langage.

---

Version 0.1 : MARS 2026

OFFICIAL CODE RELEASE **V1** : SEPTEMBER 2026

## Annotated episode videos

Install the optional encoder dependencies and enable videos from the benchmark
launcher:

```bash
pip install -e '.[video]'
python -m magma_bench.launch history_reactive \
  --benchmark_root /path/to/magma-benchmark-files \
  --videos --video_fps 20 --video_hold_seconds 1
```

Each executed episode produces an annotated MP4 in its scenario `videos/`
directory. Enabling videos while resuming an existing run records only the
episodes that are actually replayed.
