# Design: inference backends

## Scope

The inference feature presents a common local OpenAI-compatible API while
allowing multiple backend-specific server commands. Backend command
construction is separated from process lifecycle management.

## Backend contract

`arkai.inference_backend.InferenceBackend` defines environment validation
and command construction for a backend. Built-ins are registered lazily
for llama.cpp, MLX-LM, Splash, and Strata. Each implementation translates
the selected model, port, context, GPU settings, and supported sampling
values to the backend's native interface.

## Lifecycle

`arkai.inference` resolves and validates the effective configuration,
checks for an existing service, asks the backend to build a command, and
then waits for the backend's readiness endpoint. For regular process
backends it records PID and state files, writes backend output to a log,
and removes state on failed startup or stop. Strata delegates service
management to its own CLI and tracks profile ownership separately.

Arkai reuses an existing healthy regular backend. It does not silently
replace a service occupying the requested port. Strata is stricter about
matching profiles and provisioning: Arkai checks the launch plan and
requires explicit `--allow-provisioning` before downloads or engine setup.

## Implementation map

- Registry/contract: `arkai/inference_backend.py`
- Shared process and health lifecycle: `arkai/inference.py`
- Backend implementations: `arkai/inference_llama_cpp.py`,
  `arkai/inference_mlx.py`, `arkai/inference_splash.py`,
  `arkai/inference_strata.py`
- Unit/BDD coverage: `tests/unit/test_inference_backend.py`,
  `tests/unit/test_inference.py`, `features/inference.feature`
- User instructions: [Models and
inference](../guides/models-and-inference.md)
