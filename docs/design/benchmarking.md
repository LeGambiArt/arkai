# Design: benchmarking

## Scope

The benchmark feature measures response latency and generation throughput
for an inference model using built-in prompt sets or a custom prompt
file. It is a CLI evaluation utility, not a hardware telemetry or
statistically rigorous model-quality suite.

## Execution model

The runner resolves model, backend, port, context, and GPU settings from
the CLI and Arkai configuration. It can start an inference server for the
run or use an existing server with `--no-inference`. By default it
performs a warmup request, then executes each requested prompt for the
configured iterations with a fixed sampling setup. Results are rendered
as a table or JSON.

Built-in prompt sets cover general AI, code review, and coding at short,
medium, and long sizes. Prompt specifications and custom prompt files are
resolved before the run. Benchmark startup and cleanup follow Arkai's
inference ownership rules.

## Implementation map

- CLI, prompt sets, and runner: `arkai/benchmark.py`
- Inference lifecycle: `arkai/inference.py`
- Unit/BDD coverage: `tests/unit/test_benchmark.py`,
  `features/benchmark.feature`
- User instructions: [Benchmarking](../guides/benchmarking.md)
