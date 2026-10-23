"""Step definitions for benchmark feature tests."""

import shlex
from unittest.mock import patch

from behave import then, when

from arkai import benchmark, utils


@when('I run benchmark command "{cmd}"')  # ty: ignore[call-non-callable]
def step_run_benchmark_command(context, cmd):
    """Run a benchmark command."""
    parts = cmd.split()
    code, stdout, stderr = utils.run_command(["arkai"] + parts)
    context.exit_code = code
    context.stdout = stdout
    context.stderr = stderr


@when('I run benchmark command "{cmd}" with a reused Strata service')  # ty: ignore[call-non-callable]
def step_run_benchmark_with_reused_strata(context, cmd):
    """Exercise no-inference benchmarking without stopping a borrowed Strata profile."""
    context.benchmark_observed_service = None
    context.benchmark_stop = None
    context.benchmark_start = None
    state = {
        "backend": "strata",
        "model": "Q2_0",
        "context_size": 131072,
        "gpu_layers": "managed by Strata",
        "strata_owned": False,
    }
    metric = benchmark.MetricResult(mean=1.0, stddev=0.0)

    def run_iterations(runner, _prompt):
        context.benchmark_observed_service = (
            runner.config.backend,
            runner.config.model,
        )
        return {
            "prefill_throughput": metric,
            "generation_throughput": metric,
            "ttft_ms": metric,
        }

    with (
        patch.object(benchmark.inference, "is_inference_running", return_value=True),
        patch.object(benchmark.inference, "get_inference_model", return_value="Q2_0"),
        patch.object(benchmark.utils, "load_yaml", return_value=state),
        patch.object(benchmark.BenchmarkRunner, "_run_benchmark_iterations", run_iterations),
        patch.object(benchmark.BenchmarkRunner, "_estimate_peak_rss", return_value=1.0),
        patch.object(benchmark.inference, "cmd_inference_stop") as stop,
        patch.object(benchmark.inference, "cmd_inference_start") as start,
    ):
        context.run_command(["arkai", "benchmark", *shlex.split(cmd)])

    context.benchmark_stop = stop
    context.benchmark_start = start


@then("the benchmark kept using the borrowed Strata service")  # ty: ignore[call-non-callable]
def step_benchmark_kept_borrowed_strata(context):
    """The benchmark adopts the recorded profile and does not stop or relaunch it."""
    assert context.benchmark_observed_service == ("strata", "Q2_0")
    context.benchmark_stop.assert_not_called()
    context.benchmark_start.assert_not_called()
