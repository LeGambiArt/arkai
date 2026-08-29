"""Unit tests for benchmark module."""

import json

import pytest

from arkai import benchmark


class TestMetricsComputation:
    """Test metric computation functions."""

    def test_compute_prefill_throughput(self):
        """Test prefill throughput calculation."""
        runner = benchmark.BenchmarkRunner(
            benchmark.BenchmarkConfig(
                model="test.gguf",
                port=8081,
                iterations=1,
                warmup=False,
                token_limit=128,
                temperature=0.0,
                seed=42,
                prompts=["short"],
            )
        )

        # 100 tokens in 100ms = 1000 tokens/sec
        result = runner._compute_prefill_throughput(100, 100)
        assert result == 1000.0

        # 50 tokens in 200ms = 250 tokens/sec
        result = runner._compute_prefill_throughput(50, 200)
        assert result == 250.0

        # Zero time edge case
        result = runner._compute_prefill_throughput(100, 0)
        assert result == 0.0

    def test_compute_generation_throughput(self):
        """Test generation throughput calculation."""
        runner = benchmark.BenchmarkRunner(
            benchmark.BenchmarkConfig(
                model="test.gguf",
                port=8081,
                iterations=1,
                warmup=False,
                token_limit=128,
                temperature=0.0,
                seed=42,
                prompts=["short"],
            )
        )

        # 128 tokens in 1280ms = 100 tokens/sec
        result = runner._compute_generation_throughput(128, 1280)
        assert result == 100.0

        # 64 tokens in 640ms = 100 tokens/sec
        result = runner._compute_generation_throughput(64, 640)
        assert result == 100.0

        # Zero time edge case
        result = runner._compute_generation_throughput(128, 0)
        assert result == 0.0

    def test_compute_ttft(self):
        """Test time to first token calculation."""
        runner = benchmark.BenchmarkRunner(
            benchmark.BenchmarkConfig(
                model="test.gguf",
                port=8081,
                iterations=1,
                warmup=False,
                token_limit=128,
                temperature=0.0,
                seed=42,
                prompts=["short"],
            )
        )

        # TTFT equals prefill time
        result = runner._compute_ttft(50.5)
        assert result == 50.5


class TestStatisticsComputation:
    """Test statistics computation."""

    def test_metric_result_creation(self):
        """Test MetricResult dataclass creation."""
        metric = benchmark.MetricResult(mean=100.0, stddev=5.0)
        assert metric.mean == 100.0
        assert metric.stddev == 5.0

    def test_compute_statistics_with_values(self):
        """Test statistics computation with sample values."""
        values = [100.0, 110.0, 90.0, 105.0, 95.0]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        stddev = variance**0.5

        # Python statistics module should give same results
        import statistics as stats

        computed_mean = stats.mean(values)
        computed_stddev = stats.stdev(values)

        assert abs(computed_mean - mean) < 0.01
        assert abs(computed_stddev - stddev) < 0.01


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test BenchmarkConfig creation with default values."""
        config = benchmark.BenchmarkConfig(
            model="test.gguf",
            port=8081,
            iterations=5,
            warmup=True,
            token_limit=128,
            temperature=0.0,
            seed=42,
            prompts=["short"],
        )

        assert config.model == "test.gguf"
        assert config.port == 8081
        assert config.iterations == 5
        assert config.warmup is True
        assert config.token_limit == 128
        assert config.temperature == 0.0
        assert config.seed == 42
        assert config.prompts == ["short"]
        assert config.gpu_layers == -1
        assert config.context_size == 65536
        assert config.backend == "llama-cpp"
        assert config.no_inference is False
        assert config.custom_prompt_file is None

    def test_config_with_custom_values(self):
        """Test BenchmarkConfig with custom values."""
        config = benchmark.BenchmarkConfig(
            model="hf:meta-llama/Llama-2-7b",
            port=9999,
            iterations=3,
            warmup=False,
            token_limit=256,
            temperature=0.5,
            seed=100,
            prompts=["custom"],
            custom_prompt_file="/tmp/prompt.txt",
            gpu_layers=20,
            context_size=4096,
            no_inference=True,
        )

        assert config.model == "hf:meta-llama/Llama-2-7b"
        assert config.port == 9999
        assert config.iterations == 3
        assert config.warmup is False
        assert config.token_limit == 256
        assert config.temperature == 0.5
        assert config.seed == 100
        assert config.gpu_layers == 20
        assert config.context_size == 4096
        assert config.no_inference is True


class TestFormatters:
    """Test output formatting functions."""

    def test_format_metric(self):
        """Test metric formatting."""
        result = benchmark.format_metric(123.456, precision=1)
        assert result == "123.5"

        result = benchmark.format_metric(123.456, precision=2)
        assert result == "123.46"

        result = benchmark.format_metric(100.0, precision=0)
        assert result == "100"

    def test_format_json(self):
        """Test JSON format output."""
        metric1 = benchmark.MetricResult(mean=100.0, stddev=5.0)
        metric2 = benchmark.MetricResult(mean=50.0, stddev=2.0)
        metric3 = benchmark.MetricResult(mean=25.5, stddev=1.5)
        metric4 = benchmark.MetricResult(mean=2048.0, stddev=32.0)

        prompt_result = benchmark.PromptResult(
            prompt_name="test",
            prompt_text="test prompt",
            prefill_throughput=metric1,
            generation_throughput=metric2,
            ttft_ms=metric3,
            peak_rss_mb=metric4,
        )

        result = benchmark.BenchmarkResult(
            model="test.gguf",
            backend="llama-cpp",
            gpu_layers=-1,
            context_size=65536,
            iterations=5,
            warmup_included=True,
            results=[prompt_result],
        )

        json_str = benchmark.format_json(result)
        data = json.loads(json_str)

        assert data["metadata"]["model"] == "test.gguf"
        assert data["metadata"]["backend"] == "llama-cpp"
        assert data["metadata"]["gpu_layers"] == -1
        assert data["metadata"]["context_size"] == 65536
        assert data["metadata"]["iterations"] == 5
        assert data["metadata"]["warmup_included"] is True

        assert len(data["results"]) == 1
        assert data["results"][0]["prompt"] == "test"
        assert data["results"][0]["prefill_throughput"]["mean"] == 100.0
        assert data["results"][0]["prefill_throughput"]["stddev"] == 5.0

    def test_format_table(self):
        """Test table format output."""
        metric1 = benchmark.MetricResult(mean=425.3, stddev=12.1)
        metric2 = benchmark.MetricResult(mean=89.4, stddev=3.2)
        metric3 = benchmark.MetricResult(mean=11.2, stddev=0.8)
        metric4 = benchmark.MetricResult(mean=2145.0, stddev=32.0)

        prompt_result = benchmark.PromptResult(
            prompt_name="short",
            prompt_text="test",
            prefill_throughput=metric1,
            generation_throughput=metric2,
            ttft_ms=metric3,
            peak_rss_mb=metric4,
        )

        result = benchmark.BenchmarkResult(
            model="test.gguf",
            backend="llama-cpp",
            gpu_layers=-1,
            context_size=65536,
            iterations=5,
            warmup_included=True,
            results=[prompt_result],
        )

        table_str = benchmark.format_table(result)

        assert "test.gguf" in table_str
        assert "llama-cpp" in table_str
        assert "short" in table_str
        assert "425.3" in table_str
        assert "12.1" in table_str


class TestBuiltinPrompts:
    """Test built-in prompts."""

    def test_builtin_prompt_sets_exist(self):
        """Test that built-in prompt sets are defined."""
        assert "ai" in benchmark.BUILTIN_PROMPTS
        assert "code-review" in benchmark.BUILTIN_PROMPTS

    def test_builtin_prompt_sizes_exist(self):
        """Test that all prompt sizes exist in each set."""
        for set_name, sizes in benchmark.BUILTIN_PROMPTS.items():
            assert "short" in sizes
            assert "medium" in sizes
            assert "long" in sizes

    def test_builtin_prompts_not_empty(self):
        """Test that built-in prompts have content."""
        for set_name, sizes in benchmark.BUILTIN_PROMPTS.items():
            for size, prompt in sizes.items():
                assert len(prompt) > 0
                assert isinstance(prompt, str)

    def test_resolve_prompts_short(self):
        """Test resolving short prompt spec."""
        result = benchmark.resolve_prompts(["ai:short"])
        assert "ai:short" in result
        assert len(result) == 1

    def test_resolve_prompts_all(self):
        """Test resolving 'all' expands to all sizes."""
        result = benchmark.resolve_prompts(["code-review:all"])
        assert len(result) == 3
        assert "code-review:short" in result
        assert "code-review:medium" in result
        assert "code-review:long" in result

    def test_resolve_prompts_multiple(self):
        """Test resolving multiple prompt specs."""
        result = benchmark.resolve_prompts(["ai:short", "code-review:medium"])
        assert len(result) == 2
        assert "ai:short" in result
        assert "code-review:medium" in result

    def test_resolve_prompts_invalid_set(self):
        """Test error on invalid prompt set."""
        with pytest.raises(ValueError, match="Unknown prompt set"):
            benchmark.resolve_prompts(["invalid:short"])

    def test_resolve_prompts_invalid_size(self):
        """Test error on invalid prompt size."""
        with pytest.raises(ValueError, match="Invalid prompt size"):
            benchmark.resolve_prompts(["ai:tiny"])

    def test_print_prompt_set_ai(self, capsys):
        """Test printing ai prompt set."""
        benchmark.print_prompt_set("ai")
        captured = capsys.readouterr()
        assert "Prompts in set 'ai'" in captured.out
        assert "SHORT" in captured.out
        assert "MEDIUM" in captured.out
        assert "LONG" in captured.out
        assert "artificial intelligence" in captured.out

    def test_print_prompt_set_code_review(self, capsys):
        """Test printing code-review prompt set."""
        benchmark.print_prompt_set("code-review")
        captured = capsys.readouterr()
        assert "Prompts in set 'code-review'" in captured.out
        assert "SHORT" in captured.out
        assert "MEDIUM" in captured.out
        assert "LONG" in captured.out
        assert "Review" in captured.out

    def test_print_prompt_set_invalid(self):
        """Test error on invalid prompt set."""
        with pytest.raises(ValueError, match="Unknown prompt set"):
            benchmark.print_prompt_set("invalid")

    def test_print_prompt_set_coding(self, capsys):
        """Test printing coding prompt set."""
        benchmark.print_prompt_set("coding")
        captured = capsys.readouterr()
        assert "Prompts in set 'coding'" in captured.out
        assert "SHORT" in captured.out
        assert "MEDIUM" in captured.out
        assert "LONG" in captured.out
        assert "Rust" in captured.out
        assert "vector" in captured.out
