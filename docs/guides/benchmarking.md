# Benchmarking

`arkai benchmark` measures responses from a selected model using built-in
or custom prompts. By default it uses the configured model and backend,
runs five iterations, warms up first, and outputs a table.

```bash
arkai benchmark
arkai benchmark --model hf:owner/model --iterations 3 --tokens 128
arkai benchmark --prompts ai:short,code-review:medium --json
arkai benchmark --show-prompts code-review
```

Use `--prompt-file` for custom prompts, `--no-inference` to use an
existing server without starting or stopping it, and `--port` to select
an inference port. See `arkai benchmark --help` for all options and the
[benchmark design](../design/benchmarking.md) for measurement behavior.
