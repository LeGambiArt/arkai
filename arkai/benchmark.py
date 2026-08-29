"""LLM model benchmarking utilities."""

import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import requests

from arkai import engine, utils

BUILTIN_PROMPTS = {
    "ai": {
        "short": "What is artificial intelligence? Explain in one sentence.",
        "medium": (
            "Explain machine learning and its applications in modern technology. "
            "Include practical examples of how it's used in everyday life."
        ),
        "long": (
            "Write a comprehensive overview of artificial intelligence, including "
            "its history, current applications, and future potential. Discuss "
            "machine learning, deep learning, neural networks, and their impact "
            "on society. Consider both opportunities and challenges."
        ),
    },
    "code-review": {
        "short": (
            "Review this Python function for bugs:\n\n"
            "def add(a, b):\n"
            "    return a + b\n\n"
            "List any issues found."
        ),
        "medium": (
            "Review the following Python code for bugs, security issues, and "
            "performance problems:\n\n"
            "def process_user_data(users):\n"
            "    results = []\n"
            "    for user in users:\n"
            "        if user['age'] > 18:\n"
            "            results.append(user['email'])\n"
            "    return results\n\n"
            "Provide specific recommendations for improvement."
        ),
        "long": (
            "Review the following Python code thoroughly for bugs, security issues, "
            "performance problems, style violations, and architectural concerns:\n\n"
            "class UserManager:\n"
            "    def __init__(self):\n"
            "        self.users = []\n"
            "    def add_user(self, name, email, password):\n"
            "        import hashlib\n"
            "        hashed = hashlib.md5(password.encode()).hexdigest()\n"
            "        self.users.append({'name': name, 'email': email, 'pwd': hashed})\n"
            "    def find_user(self, email):\n"
            "        for u in self.users:\n"
            "            if u['email'] == email:\n"
            "                return u\n"
            "        return None\n"
            "    def get_all_emails(self):\n"
            "        return [u['email'] for u in self.users]\n\n"
            "Provide detailed recommendations for security, design patterns, error handling, "
            "and testing strategies. Suggest improved implementation."
        ),
    },
    "coding": {
        "short": (
            "Write a Rust function that takes a vector of integers and returns "
            "the sum of all even numbers. Include proper error handling and type hints."
        ),
        "medium": (
            "Write a Rust module that implements a simple key-value store with the "
            "following functionality: insert, retrieve, remove, and list all keys. "
            "Use appropriate data structures, error handling with Result types, and "
            "include unit tests. Follow Rust idioms and best practices."
        ),
        "long": (
            "Design and implement a Rust library for concurrent file processing. "
            "Requirements: (1) Read files asynchronously using tokio; (2) Parse each file "
            "with error recovery for malformed data; (3) Use channels to coordinate work "
            "across threads; (4) Implement proper error propagation with custom error types; "
            "(5) Include comprehensive unit and integration tests; (6) Add documentation "
            "with examples; (7) Follow Rust 2021 edition idioms and provide efficient "
            "memory management. Consider edge cases like large files, concurrent access, "
            "and partial failures."
        ),
    },
}


def print_prompt_set(set_name: str) -> None:
    """Print all prompts from a specific set.

    Args:
        set_name: Name of the prompt set (e.g., 'ai', 'code-review')

    Raises:
        ValueError: If prompt set is invalid
    """
    if set_name not in BUILTIN_PROMPTS:
        available = ", ".join(BUILTIN_PROMPTS.keys())
        raise ValueError(f"Unknown prompt set '{set_name}'. Available: {available}")

    print(f"Prompts in set '{set_name}':")
    print()

    for size in ("short", "medium", "long"):
        if size not in BUILTIN_PROMPTS[set_name]:
            continue

        prompt = BUILTIN_PROMPTS[set_name][size]
        print(f"--- {size.upper()} ---")
        print(prompt)
        print()


def resolve_prompts(prompt_specs: list[str]) -> dict[str, str]:
    """Resolve prompt specifications to actual prompts.

    Supports format: "<set>:<size>" where size can be "short", "medium", "long", or "all".
    Examples: "review:short", "ai:all", "review:medium"

    Args:
        prompt_specs: List of prompt specifications

    Returns:
        Dictionary mapping prompt names to prompt text

    Raises:
        ValueError: If prompt set or size is invalid
    """
    resolved = {}

    for spec in prompt_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid prompt spec '{spec}'. Use format '<set>:<size>' "
                f"(e.g., 'review:short', 'ai:all')"
            )

        set_name, size = spec.split(":", 1)

        if set_name not in BUILTIN_PROMPTS:
            available = ", ".join(BUILTIN_PROMPTS.keys())
            raise ValueError(f"Unknown prompt set '{set_name}'. Available: {available}")

        if size == "all":
            sizes = ["short", "medium", "long"]
        elif size in ("short", "medium", "long"):
            sizes = [size]
        else:
            raise ValueError(
                f"Invalid prompt size '{size}'. Use 'short', 'medium', 'long', or 'all'"
            )

        for s in sizes:
            if s not in BUILTIN_PROMPTS[set_name]:
                raise ValueError(f"Size '{s}' not available in set '{set_name}'")
            prompt_name = f"{set_name}:{s}"
            resolved[prompt_name] = BUILTIN_PROMPTS[set_name][s]

    return resolved


@dataclass
class MetricResult:
    """Result of a single metric across iterations."""

    mean: float
    stddev: float


@dataclass
class PromptResult:
    """Benchmark results for a single prompt."""

    prompt_name: str
    prompt_text: str
    prefill_throughput: MetricResult
    generation_throughput: MetricResult
    ttft_ms: MetricResult
    peak_rss_mb: MetricResult


@dataclass
class BenchmarkResult:
    """Complete benchmark results with metadata."""

    model: str
    backend: str
    gpu_layers: int
    context_size: int
    iterations: int
    warmup_included: bool
    results: list[PromptResult]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model: str
    port: int
    iterations: int
    warmup: bool
    token_limit: int
    temperature: float
    seed: int
    prompts: list[str]
    custom_prompt_file: Optional[str] = None
    gpu_layers: int = -1
    context_size: int = 65536
    backend: str = "llama-cpp"
    no_inference: bool = False


class BenchmarkRunner:
    """Orchestrates benchmark execution and metrics collection."""

    def __init__(self, bench_config: BenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            bench_config: Benchmark configuration
        """
        self.config = bench_config
        self.started_server = False

    def _ensure_server_running(self) -> None:
        """Check if inference server is running, start if needed."""
        if self.config.no_inference:
            if not engine.is_inference_running():
                raise RuntimeError("Inference server not running (required with -I/--no-inference)")
            return

        if engine.is_inference_running():
            return

        utils.info("Starting inference server for benchmark...")
        engine.cmd_engine_start(
            model=self.config.model,
            gpu_layers=self.config.gpu_layers,
            context_size=self.config.context_size,
            port=self.config.port,
        )
        self.started_server = True

    def _cleanup_server(self) -> None:
        """Stop inference server if we started it."""
        if self.started_server:
            utils.info("Stopping inference server...")
            engine.cmd_engine_stop()
            self.started_server = False

    def _send_completion_request(self, prompt: str) -> tuple[dict, float, float, float]:
        """Send completion request and capture timing.

        Returns:
            Tuple of (response_dict, prefill_time_ms, generation_time_ms, total_time_ms)
        """
        url = f"http://127.0.0.1:{self.config.port}/v1/completions"
        payload = {
            "prompt": prompt,
            "n_predict": self.config.token_limit,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
        }

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        response.raise_for_status()
        data = response.json()

        # Extract timing data from response
        timings = data.get("timings", {})
        prefill_time_ms = timings.get("prompt_ms", 0)
        generation_time_ms = timings.get("predicted_ms", 0)

        return data, prefill_time_ms, generation_time_ms, total_time

    def _compute_prefill_throughput(self, prompt_tokens: int, prefill_time_ms: float) -> float:
        """Compute prefill throughput in tokens/second.

        Args:
            prompt_tokens: Number of tokens in prompt
            prefill_time_ms: Time to process prompt in milliseconds

        Returns:
            Tokens per second
        """
        if prefill_time_ms <= 0:
            return 0.0
        return (prompt_tokens / prefill_time_ms) * 1000

    def _compute_generation_throughput(
        self, generated_tokens: int, generation_time_ms: float
    ) -> float:
        """Compute generation throughput in tokens/second.

        Args:
            generated_tokens: Number of tokens generated
            generation_time_ms: Time to generate tokens in milliseconds

        Returns:
            Tokens per second
        """
        if generation_time_ms <= 0:
            return 0.0
        return (generated_tokens / generation_time_ms) * 1000

    def _compute_ttft(self, prefill_time_ms: float) -> float:
        """Compute time to first token in milliseconds.

        Args:
            prefill_time_ms: Time to process prompt

        Returns:
            Time to first token in milliseconds
        """
        return prefill_time_ms

    def _run_single_iteration(self, prompt: str) -> dict:
        """Run a single benchmark iteration.

        Args:
            prompt: Prompt text to benchmark

        Returns:
            Dictionary with metrics for this iteration
        """
        response, prefill_time_ms, generation_time_ms, _ = self._send_completion_request(prompt)

        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Compute metrics
        prefill_throughput = self._compute_prefill_throughput(prompt_tokens, prefill_time_ms)
        generation_throughput = self._compute_generation_throughput(
            completion_tokens, generation_time_ms
        )
        ttft = self._compute_ttft(prefill_time_ms)

        return {
            "prefill_throughput": prefill_throughput,
            "generation_throughput": generation_throughput,
            "ttft_ms": ttft,
        }

    def _run_benchmark_iterations(self, prompt: str) -> dict:
        """Run multiple iterations of benchmark and collect metrics.

        Returns:
            Dictionary with collected metrics from all iterations
        """
        iterations_to_run = self.config.iterations
        if self.config.warmup:
            iterations_to_run += 1

        all_results = []
        for i in range(iterations_to_run):
            result = self._run_single_iteration(prompt)
            all_results.append(result)

        # Filter out warmup run if present
        if self.config.warmup:
            results = all_results[1:]
        else:
            results = all_results

        # Compute statistics
        prefill_throughputs = [r["prefill_throughput"] for r in results]
        generation_throughputs = [r["generation_throughput"] for r in results]
        ttfts = [r["ttft_ms"] for r in results]

        def _stats(values: list[float]) -> MetricResult:
            return MetricResult(
                mean=statistics.mean(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
            )

        return {
            "prefill_throughput": _stats(prefill_throughputs),
            "generation_throughput": _stats(generation_throughputs),
            "ttft_ms": _stats(ttfts),
        }

    def _load_prompts(self) -> dict[str, str]:
        """Load prompts for benchmarking.

        Returns:
            Dictionary of prompt_name -> prompt_text
        """
        prompts = {}

        # Add requested built-in prompts
        for prompt_spec in self.config.prompts:
            if prompt_spec.startswith("file:"):
                # Load from file
                file_path = prompt_spec[5:]
                if not os.path.exists(file_path):
                    raise RuntimeError(f"Prompt file not found: {file_path}")
                with open(file_path) as f:
                    prompts[os.path.basename(file_path)] = f.read()
            elif ":" in prompt_spec:
                # Resolve built-in prompts with <set>:<size> format
                resolved = resolve_prompts([prompt_spec])
                prompts.update(resolved)
            else:
                raise ValueError(
                    f"Invalid prompt spec '{prompt_spec}'. "
                    "Use '<set>:<size>' (e.g., 'review:short') or 'file:<path>'"
                )

        # Add custom prompt file if specified
        if self.config.custom_prompt_file:
            if not os.path.exists(self.config.custom_prompt_file):
                raise RuntimeError(f"Prompt file not found: {self.config.custom_prompt_file}")
            with open(self.config.custom_prompt_file) as f:
                prompts["custom"] = f.read()

        if not prompts:
            raise RuntimeError("No prompts specified")

        return prompts

    def run(self) -> BenchmarkResult:
        """Run complete benchmark.

        Returns:
            BenchmarkResult with all metrics
        """
        try:
            self._ensure_server_running()

            prompts = self._load_prompts()
            prompt_results = []

            for prompt_name, prompt_text in prompts.items():
                utils.info(f"Benchmarking prompt: {prompt_name}")
                metrics = self._run_benchmark_iterations(prompt_text)

                # Estimate peak RSS (simple approximation via process info)
                peak_rss_mb = self._estimate_peak_rss()

                result = PromptResult(
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    prefill_throughput=metrics["prefill_throughput"],
                    generation_throughput=metrics["generation_throughput"],
                    ttft_ms=metrics["ttft_ms"],
                    peak_rss_mb=MetricResult(mean=peak_rss_mb, stddev=0.0),
                )
                prompt_results.append(result)

            return BenchmarkResult(
                model=self.config.model,
                backend=self.config.backend,
                gpu_layers=self.config.gpu_layers,
                context_size=self.config.context_size,
                iterations=self.config.iterations,
                warmup_included=self.config.warmup,
                results=prompt_results,
            )
        finally:
            self._cleanup_server()

    def _estimate_peak_rss(self) -> float:
        """Estimate peak RSS memory in MB.

        Returns:
            Peak resident set size in MB
        """
        # Read from inference server process
        pid_path = engine.get_inference_pid_path()
        pid = utils.read_pid(pid_path)

        if pid is None:
            return 0.0

        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Extract memory in KB and convert to MB
                        match = re.search(r"(\d+)", line)
                        if match:
                            kb = int(match.group(1))
                            return kb / 1024
        except (FileNotFoundError, ValueError):
            pass

        return 0.0


def format_metric(value: float, precision: int = 1) -> str:
    """Format a metric with mean and optional stddev.

    Args:
        value: Metric value
        precision: Decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"


def format_table(result: BenchmarkResult) -> str:
    """Format benchmark result as human-readable table.

    Args:
        result: BenchmarkResult to format

    Returns:
        Formatted table string
    """
    lines = []
    metadata = (
        f"Model: {result.model} | Backend: {result.backend} | "
        f"GPU Layers: {result.gpu_layers} | Context: {result.context_size}"
    )
    lines.append(metadata)
    lines.append("")

    # Header
    header = (
        f"{'Prompt':<15} {'Prefill (t/s)':<18} {'Generate (t/s)':<18} "
        f"{'TTFT (ms)':<15} {'Peak RSS (MB)':<15}"
    )
    lines.append(header)
    lines.append("─" * len(header))

    # Rows
    for prompt_result in result.results:
        prefill = prompt_result.prefill_throughput
        generation = prompt_result.generation_throughput
        ttft = prompt_result.ttft_ms
        rss = prompt_result.peak_rss_mb

        prefill_str = f"{prefill.mean:.1f} ± {prefill.stddev:.1f}"
        generation_str = f"{generation.mean:.1f} ± {generation.stddev:.1f}"
        ttft_str = f"{ttft.mean:.1f} ± {ttft.stddev:.1f}"
        rss_str = f"{rss.mean:.0f} ± {rss.stddev:.0f}"

        row = (
            f"{prompt_result.prompt_name:<15} {prefill_str:<18} "
            f"{generation_str:<18} {ttft_str:<15} {rss_str:<15}"
        )
        lines.append(row)

    return "\n".join(lines)


def format_json(result: BenchmarkResult) -> str:
    """Format benchmark result as JSON.

    Args:
        result: BenchmarkResult to format

    Returns:
        JSON string
    """
    output = {
        "metadata": {
            "model": result.model,
            "backend": result.backend,
            "gpu_layers": result.gpu_layers,
            "context_size": result.context_size,
            "iterations": result.iterations,
            "warmup_included": result.warmup_included,
        },
        "results": [
            {
                "prompt": r.prompt_name,
                "prefill_throughput": {
                    "mean": round(r.prefill_throughput.mean, 1),
                    "stddev": round(r.prefill_throughput.stddev, 1),
                },
                "generation_throughput": {
                    "mean": round(r.generation_throughput.mean, 1),
                    "stddev": round(r.generation_throughput.stddev, 1),
                },
                "ttft_ms": {
                    "mean": round(r.ttft_ms.mean, 1),
                    "stddev": round(r.ttft_ms.stddev, 1),
                },
                "peak_rss_mb": {
                    "mean": round(r.peak_rss_mb.mean, 1),
                    "stddev": round(r.peak_rss_mb.stddev, 1),
                },
            }
            for r in result.results
        ],
    }
    return json.dumps(output, indent=2)
