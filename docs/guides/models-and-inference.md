# Models and inference

Arkai can download model references into its local cache, start an
OpenAI-compatible inference server, and manage local inference backends.
The default backend is `llama-cpp` and uses `llama-server`.

## Download and inspect models

Model downloads use a provider-qualified reference. Hugging Face
references start with `hf:`; Ollama references start with `ollama:`:

```bash
arkai model download hf:ibm-granite/granite-4.1-8b-GGUF
arkai model download ollama:llama3.2:latest
arkai model list
arkai model resolve hf:ibm-granite/granite-4.1-8b-GGUF
```

GGUF files are stored under `~/.local/share/arkai/models/`. Some
repositories can be served directly; convert only models that the
selected backend cannot load in their source format:

```bash
arkai model convert hf:owner/model -q Q6_K
```

Other model operations include `arkai model remove`, `arkai model prune
--backend splash --dry-run`, and `arkai model resolve --backend splash`.
Use `arkai model --help` for the complete syntax. Splash model pruning is
limited to unreferenced data in Arkai's default managed caches; review
the dry run before removing anything.

## Start and manage inference

```bash
arkai inference start
arkai inference status
arkai inference stop
```

Set `inference.model` in configuration or pass `--model`. The server
defaults to port `8081`; use `--backend`, `--port`, `--gpu-layers`,
`--context`, or `--allow-provisioning` when appropriate. `arkai agent
start` starts the configured local inference service automatically if one
is not already available. A running service can be kept warm between
sessions.

## Backends

- **llama.cpp:** default backend; install `llama-server`. Accepts local
  models and supported provider references.
- **MLX-LM:** optional on Apple Silicon. Install Arkai's extra with
  `python -m pip install 'arkai[mlx]'` (or `python -m pip install -e
  '.[mlx]'` from a checkout), then select `inference.backend: mlx`.
- **Splash:** install Splash separately and select `inference.backend:
  splash`. Arkai uses Splash's native model metadata and manages downloads
  through the supported installer interface; models are not converted to
  GGUF.
- **Strata:** install with `arkai tools install strata` or follow
  Strata's official instructions. Strata uses its own catalog IDs, such as
  `q2_0`, and may request model or engine provisioning. Arkai refuses
  provisioning unless `--allow-provisioning` is passed. Its own setup and
  download phase may take longer than Arkai's API readiness timeout. Arkai
  reuses only a matching Strata profile and does not stop profiles it did
  not start.

Splash model downloads and serving use the same Hub and weight-cache
locations. Defaults are `~/.local/share/arkai/models/splash/hub` and
`~/.local/share/arkai/models/splash/weights`;
`inference.splash.hub_cache` and `inference.splash.weight_cache` can
override them. Native Splash selections remain in Splash's installation
directory, and existing downloads are not moved automatically. Stop
Splash servers and provisioning operations before removing or pruning
models. Pruning is limited to recognized, unreferenced data in the
default Arkai-owned caches; external and custom caches are not pruned.

Backend-specific flags can be unsupported or rejected instead of silently
ignored. See the [inference backend
design](../design/inference-backends.md) and [model management
design](../design/model-management.md) for the implementation and
lifecycle details.
