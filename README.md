# aitool

aitool is a CLI for running AI agents with local models on your own hardware. It orchestrates three services — a llama.cpp inference server, an MCP plugin server (wtmcp), and optionally an arapuca sandbox — so you can launch a fully configured agent session with a single command. Configuration is layered: user-level defaults, per-project overrides, and CLI flags.

## How It Works

When you run `aitool agent`, it starts `llama-server` (loading your GGUF model), starts wtmcp (which exposes MCP tools to the agent over HTTP), then launches the agent binary configured to talk to both over localhost. Optionally it wraps the agent in an arapuca sandbox that restricts filesystem and network access. On exit, aitool tears down wtmcp — and optionally the inference server — unless `--keep-mcp` or `--keep-inference` are passed.

```
┌─────────────────────────────────────────────────────┐
│  arkai agent start                                  │
│                                                     │
│  ┌──────────────┐   HTTP/v1   ┌─────────────────┐   │
│  │  llama-server│◄────────────│                 │   │
│  │  (inference) │             │   agent binary  │   │
│  └──────────────┘             │ (opencode/crush │   │
│                               │   /claude)      │   │
│  ┌──────────────┐   MCP/HTTP  │                 │   │
│  │    wtmcp     │◄────────────│                 │   │
│  │  (plugins)   │             └─────────────────┘   │
│  └──────────────┘                    │              │
│                               ┌──────┴──────┐       │
│                               │   arapuca   │       │
│                               │  (sandbox)  │       │
│                               └─────────────┘       │
└─────────────────────────────────────────────────────┘
```

## Requirements


** Commom development and CLI tools **

- `Python`
- `pip`
- `git`
- `shasum`
- `grep`, `sed`, `bash` (modern version)

**External tools:**

- `hf` (HuggingFace CLI) — for model download and listing
- `llama.cpp` (`llama-server` and `llama-quantize`)
- `wtmcp` — for MCP tool integration with agents *(optional)*
- `arapuca` — for sandboxed agent execution *(optional)*
- Agent binary: `opencode`, `crush`, or `claude` — at least one required


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/aitool.git
   cd aitool
   ```

2. Create and activate a Python 3.9 virtual environment:

   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

3. Install the package:

   ```bash
   pip install -e .
   ```

   For development tools (pytest, ruff, etc.):

   ```bash
   pip install -e ".[dev]"
   ```

> On Linux, `arapuca` is available as a Fedora package.
  On macOS, `hf`, `llama.cpp`, and `arapuca` are available through Homebrew.

## Quick Start

Download a GGUF model:

```
aitool model download ibm-granite/granite-4.1-8b-GGUF
```

Start agent:

```
aitool agent start -m ibm-granite/granite-4.1-8b-GGUF -a opencode
```

If you think you only live once (or don't have `wtmcp` or `arapuca` installed):

```
aitool agent start \
    -m ibm-granite/granite-4.1-8b-GGUF \
    -a opencode \
    --no-sandbox \
    --no-mcp
```

## Managing Models

Models are stored as GGUF files in `~/.local/share/aitool/models/`. The
`inference.model` config key refers to the filename within that directory.

### Download from HuggingFace

```bash
aitool model download <hf-repo>
```

Downloads the repository to the HuggingFace local cache. Requires the `hf` CLI.
The downloaded files are not yet in a format usable by llama-server — use
`aitool model convert` next.

```bash
aitool model download ibm-granite/granite-4.1-8b-instruct-GGUF
```

### Convert to GGUF

```bash
aitool model convert <hf-repo-or-name> [-q QUANTIZATION] [-o OUTPUT]
```

Converts a cached HuggingFace model to GGUF format and places it in the
models directory. This is the step that makes a model usable by llama-server.

| Flag | Default | Description |
|------|---------|-------------|
| `-q/--quantization` | `Q6_K` | Quantization level |
| `-o/--output` | `~/.local/share/aitool/models/MODEL-QUANTIZATION.gguf` | Output path |

Common quantization levels: `Q4_K_M` (smaller, faster), `Q5_K_M`, `Q6_K` (default, good balance).

```bash
aitool model convert ibm-granite/granite-4.1-8b-instruct-GGUF -q Q6_K
```

### List models

```bash
aitool model list
```

Shows two categories: local GGUF files in the models directory, and
HuggingFace-cached models (requires `hf` CLI).

### Remove a model

```bash
aitool model remove <model-name>
```

Deletes a local GGUF file by filename (not full path).

```bash
aitool model remove granite-4.1-8b-instruct-Q6_K.gguf
```

## Running an Inference Server

`aitool inference` manages the llama-server process. The server runs in the
background and stays running between agent sessions.

### Start

```bash
aitool inference start [--model NAME] [--gpu-layers N] [--context N]
```

Starts `llama-server` on `localhost:8081` (default). CLI flags override the
config values for that run only.

| Flag | Description |
|------|-------------|
| `--model NAME` | Override model filename from config |
| `--gpu-layers N` | Override number of GPU layers (`-1` = all) |
| `--context N` | Override context window size in tokens |

**GPU detection:** aitool auto-detects Metal (Apple Silicon), CUDA (NVIDIA),
ROCm (AMD), or falls back to CPU. Setting `gpu_layers: -1` in config offloads
all layers to GPU.

The inference server exposes an OpenAI-compatible API at `http://127.0.0.1:<port>/v1`.

### Status

```bash
aitool inference status
```

Shows: running/stopped, PID, model, port, GPU layers, context size, health,
and detected GPU type.

### Stop

```bash
aitool inference stop
```

Terminates the server and cleans up PID and state files.

## Running an Agent

`aitool agent` is the primary command. It auto-starts the inference server if
not already running, starts wtmcp (unless `--no-mcp`), launches the agent, and
tears down services on exit.

```bash
aitool agent [-a AGENT] [-m MODEL] [options]
```

**Supported agents:** `opencode`, `crush`, `claude`

### Flags

| Flag | Description |
|------|-------------|
| `-a/--agent AGENT` | Override agent from config |
| `-m/--model MODEL` | Override model from config |
| `--no-mcp` | Skip wtmcp entirely |
| `--no-sandbox` | Skip sandbox for this run |
| `-s/--sandbox PROFILE` | Use a specific sandbox profile for this run |
| `-I/--keep-inference` | Leave inference server running after exit |
| `-M/--keep-mcp` | Leave wtmcp running after exit |
| `-v/--volume PATH` | Mount extra volume in sandbox (repeatable) |
| `-e/--env KEY=VALUE` | Set env var in sandbox (repeatable) |
| `--no-cwd` | Do not mount current directory in sandbox |
| `--cwd PATH` | Override directory mounted as cwd in sandbox |

### Lifecycle

On exit: wtmcp stops automatically. The inference server also stops unless
`--keep-inference` is passed. Use `--keep-inference` when you want the server
warm for the next session.

## Managing Plugins (wtmcp)

wtmcp is an MCP server that exposes tools (plugins) to the agent over HTTP.
`aitool agent` manages wtmcp automatically, but you can also control it directly.

### Enabling and disabling plugins

These commands edit `.aitool.yaml` in the current directory (must exist — run
`aitool config init` first):

```bash
aitool wtmcp enable <plugin>    # add plugin to .aitool.yaml
aitool wtmcp disable <plugin>   # remove plugin from .aitool.yaml
```

### Plugin list semantics

`wtmcp.plugins` in `.aitool.yaml` is a list of plugin names:
- Key **absent**: inherit plugins from user config
- `plugins: []`: explicitly disable all plugins
- `plugins: [workspace, terminal]`: use only these plugins

### List plugins

```bash
aitool wtmcp list [--port N]
```

Shows all discovered plugins with status:
- 🟢 enabled (in config and discovered by wtmcp)
- ⚪ discovered but not enabled
- 🔴 enabled but not discovered by wtmcp

Requires a running wtmcp instance. If multiple instances are running, specify `--port`.

### Server lifecycle

```bash
aitool wtmcp start [--port N] [--enable PLUGIN] [--disable PLUGIN]
aitool wtmcp stop  [--port N]
aitool wtmcp status [--port N]
```

`--enable`/`--disable` on `start` override the project config plugins for that
run only. `--port` is required for `stop` and `list` when multiple instances
are running.

## Sandboxing

aitool can wrap agent execution in an arapuca sandbox. The sandbox restricts
what the agent can access:

- **Filesystem:** only explicitly mounted paths are visible
- **Network (Linux):** only the inference server and wtmcp ports are reachable;
  all other network access is denied
- **Network (macOS):** `baseline` seccomp profile is applied

The current working directory is mounted read-write by default (override with
`--cwd` or disable with `--no-cwd`).

### Profiles

Named profiles are presets stored in your user config under
`sandbox.profiles.<name>`. When `aitool agent` runs, it uses the active profile
(or the root `sandbox` defaults if no profile is active).

Profile names must be alphanumeric + underscores. The names `default` and
`active` are reserved.

### Managing profiles

```bash
# List all profiles and which is active
aitool sandbox list

# Show full details of a profile
aitool sandbox show <profile-name>
aitool sandbox show default   # root sandbox defaults
aitool sandbox show active    # currently active profile

# Create a profile
aitool sandbox create <name> [--from PROFILE] [--memory MB] [--cpus N] \
    [--pids N] [--timeout S] [-v /path[:ro]] [-e KEY=VALUE]

# Delete a profile
aitool sandbox delete <name>

# Promote a profile's settings to root defaults
aitool sandbox set-default <name>

# Set the active profile (used by aitool agent)
aitool sandbox active <name>

# Clear the active profile (use root defaults)
aitool sandbox active
```

### Example workflow

```bash
# Create a restricted profile for untrusted code
aitool sandbox create restricted --memory 1024 --cpus 1 --pids 128

# Make it the active profile
aitool sandbox active restricted

# Run the agent (uses restricted profile automatically)
aitool agent

# Override to a different profile for one run
aitool agent -s dev
```

### Per-run overrides

These flags on `aitool agent` override the active profile for that run only:

| Flag | Effect |
|------|--------|
| `-s/--sandbox PROFILE` | Use a specific profile |
| `--no-sandbox` | Disable sandbox entirely |
| `-v/--volume PATH` | Add an extra volume mount |
| `-e/--env KEY=VALUE` | Add an env var inside the sandbox |
| `--cwd PATH` | Override cwd mounted in sandbox |
| `--no-cwd` | Do not mount cwd at all |

## Configuration Reference

Configuration is YAML. Precedence (lowest to highest):

1. Built-in defaults
2. User config: `~/.config/aitool/aitool.yaml`
3. Project config: `.aitool.yaml` in current directory
4. CLI flags

```bash
aitool config init              # scaffold .aitool.yaml in current directory
aitool config validate          # validate .aitool.yaml
aitool config validate --file PATH  # validate a specific file
```

### `agent`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.name` | string | `opencode` | Agent to use: `opencode`, `crush`, or `claude` |
| `agent.path` | string | *(agent name)* | Custom path to agent binary |
| `agent.mcp` | bool | `true` | Whether to start wtmcp for agent sessions |

### `inference`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inference.model` | string | — | GGUF filename in models dir (mutually exclusive with `hf`) |
| `inference.hf` | string | — | HuggingFace repo ID to load directly (mutually exclusive with `model`) |
| `inference.backend` | string | `llama-cpp` | Inference backend (only `llama-cpp` supported) |
| `inference.path` | string | `llama-server` | Path to `llama-server` binary |
| `inference.port` | int | `8081` | Port for inference server (1024–65535) |
| `inference.gpu_layers` | int | `-1` | GPU layers to offload (`-1` = all) |
| `inference.context_size` | int | `65536` | Context window size in tokens |

### `wtmcp`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `wtmcp.path` | string | `wtmcp` | Path to `wtmcp` binary |
| `wtmcp.port` | int | `8080` | Port for wtmcp server (1024–65535) |
| `wtmcp.workdir` | string | — | Working directory for wtmcp |
| `wtmcp.plugins` | list | *(inherit)* | Plugin list; `[]` = disable all; absent = inherit from user config |

### `sandbox`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sandbox.enabled` | bool | `true` | Enable sandboxing |
| `sandbox.path` | string | `arapuca` | Path to `arapuca` binary |
| `sandbox.memory_mb` | int | `2048` | Memory limit in MB |
| `sandbox.cpus` | int | `2` | CPU count |
| `sandbox.pids` | int | `256` | PID limit |
| `sandbox.timeout` | int | `0` | Timeout in seconds (`0` = no limit) |
| `sandbox.volume` | list | `[]` | Extra volume mounts (`/path` or `/path:ro`) |
| `sandbox.environment` | map | — | Extra environment variables passed into sandbox |
| `sandbox.active_profile` | string | — | Name of the active sandbox profile |
| `sandbox.profiles.<name>` | map | — | Named sandbox profile (same keys as above) |

### Example user config

```yaml
agent:
  name: opencode

inference:
  backend: llama-cpp
  path: /usr/local/bin/llama-server
  port: 8081
  gpu_layers: -1
  context_size: 65536

wtmcp:
  path: /usr/local/bin/wtmcp
  port: 8080

sandbox:
  enabled: true
  path: /usr/local/bin/arapuca
  memory_mb: 2048
  cpus: 2
  pids: 256
  timeout: 0
```

### Example project config

```yaml
inference:
  model: granite-4.1-8b-instruct-Q6_K.gguf

wtmcp:
  plugins: [workspace, terminal, github]
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
