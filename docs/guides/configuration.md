# Configuration

Arkai reads YAML configuration from built-in defaults, the user
configuration, and (when present) `.arkai.yaml` in the current project.
Project values override user values; command-line options override
configuration for that invocation.

- User configuration: `~/.config/arkai/arkai.yaml`
- Project configuration: `.arkai.yaml`

Initialize a project file and validate it after editing:

```bash
arkai config init
arkai config validate
arkai config validate --file /path/to/arkai.yaml
```

`config init` refuses to replace an existing `.arkai.yaml`. The complete
key reference, with defaults and descriptions, is in
[arkai-full-config.yaml](../arkai-full-config.yaml).

## Common settings

| Key | Purpose |
| --- | --- |
| `agent.name` | Default agent: `opencode`, `crush`, `claude`, `codex`, `pi`, or `omp`. |
| `agent.path` | Optional path to the agent executable. |
| `agent.mcp` | Whether agent launches start the configured MCP integration; defaults to enabled. |
| `inference.backend` | `llama-cpp` (default), `mlx`, `splash`, or `strata`. |
| `inference.model` | Model reference or a name in `inference.profiles`. |
| `inference.profiles` | Mapping of profile names to a model and optional sampling settings. |
| `inference.path` | Optional executable path for the selected backend. |
| `inference.port` | Local inference port; defaults to `8081`. |
| `inference.gpu_layers` | GPU layers for backends that support the setting; defaults to `-1` (all). |
| `inference.context_size` | Context window; accepts a positive integer or `k`/`m` suffix. |
| `inference.startup_timeout` | Readiness timeout in seconds; defaults to `600`. |
| `wtmcp.path`, `wtmcp.port` | MCP server executable and port; default port is `8080`. |
| `wtmcp.workdir`, `wtmcp.config` | Optional working directory and base wtmcp config path. |
| `wtmcp.plugins` | Project-level plugin entries injected into the wtmcp config. |
| `sandbox.path` | `arapuca` executable path. |
| `sandbox.profiles` | Named resource, volume, and environment presets for sandbox runs. |
| `sandbox.active_profile` | Profile selected for agent launches unless overridden with `--sandbox`. |
| `sandbox.volume`, `sandbox.environment` | Root-level extra mounts and environment values. |
| `vectordb.*` | ChromaDB executable, port, and data directory. |
| `rag.*` | Document chunking, embedding model, and retrieval defaults. |

Sampling settings (`temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, and `repetition_penalty`) can be
set at `inference` level or overridden per model profile. Not all
backends support all settings. A custom cloud provider can be configured
under `agent.cloud.custom`; see [Cloud providers](cloud-providers.md).
See the [models and inference guide](models-and-inference.md) for
backend-specific behavior and the [configuration design](../design/configuration.md)
for implementation details.

## Example

```yaml
agent:
  name: opencode

inference:
  backend: llama-cpp
  model: hf:ibm-granite/granite-4.1-8b-GGUF
  port: 8081
  context_size: 65536

wtmcp:
  port: 8080

sandbox:
  memory_mb: 2048
  cpus: 2
  pids: 256
```

Use `arkai --help` and the per-command `--help` output for command-line
overrides. The CLI is authoritative for supported options.
