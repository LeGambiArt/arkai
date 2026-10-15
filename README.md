# arkai

For troubleshooting, add the global `--debug` option before a command. It enables
maximum log output and prints a full exception traceback when a command fails:

```bash
arkai --debug inference start
```

arkai is a CLI for running AI agents with local models on your own hardware. It orchestrates three services — an OpenAI-compatible inference server (llama.cpp or optional MLX-LM), an MCP plugin server (wtmcp), and optionally an arapuca sandbox — so you can launch a fully configured agent session with a single command. Configuration is layered: user-level defaults, per-project overrides, and CLI flags.

## How It Works

When you run `arkai agent start`, it starts `llama-server` (loading your GGUF model), starts wtmcp (which exposes MCP tools to the agent over HTTP), then launches the agent binary configured to talk to both over localhost. Optionally it wraps the agent in an arapuca sandbox that restricts filesystem and network access. On exit, arkai tears down wtmcp — and optionally the inference server — unless `--keep-mcp` or `--keep-inference` are passed.

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


**Common development and CLI tools**

- `Python`
- `pip`
- `git`
- `shasum`
- `grep`, `sed`, `bash` (modern version)

**External tools:**

- `git` — for downloading and updating Hugging Face model repositories
- `llama.cpp` (`llama-server` and `llama-quantize`)
- `mlx-lm` (`pip install -e '.[mlx]'`, optional; Apple Silicon)
- `wtmcp` — for MCP tool integration with agents *(optional)*
- `arapuca` — for sandboxed agent execution *(optional)*
- Agent binary: `opencode`, `crush`, `claude`, `pi`, or `omp` — at least one required


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/arkai.git
   cd arkai
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
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
  On macOS, `llama.cpp` and `arapuca` are available through Homebrew.

## Quick Start

Download a model repository:

```
arkai model download hf:ibm-granite/granite-4.1-8b-GGUF
```

Start agent:

```
arkai agent start -m hf:ibm-granite/granite-4.1-8b-GGUF -a opencode
```

Install pi.dev and its standard MCP, web-access, and subagent packages:

```
arkai agent install pi
```

Run `arkai agent install` without a name to list installable agents. The command
warns that installation writes files to disk and requires confirmation.
It installs `@earendil-works/pi-coding-agent`, `pi-mcp-adapter`, `pi-web-access`,
and `pi-subagents`.

Pi is installed only for arkai under `~/.local/state/arkai/pi/`; it is not added to
your global npm installation or shell `PATH`. Node.js and npm must be installed.

Install [oh-my-pi](https://github.com/can1357/oh-my-pi) using its verified official
macOS/Linux binary installer:

```bash
arkai agent install omp
```

OMP is installed under `~/.local/state/arkai/omp/`; it is not added to your global
`PATH` or OMP installation. Arkai configures OMP to use its local inference server
and, when enabled, its wtmcp MCP server. OMP also supports non-interactive prompts:

```bash
arkai agent prompt -a omp "Summarize this project"
```

The automatic OMP installer currently supports macOS and Linux. On other platforms,
install OMP using its [official instructions](https://github.com/can1357/oh-my-pi#install)
and set `agent.path` to the executable in Arkai's config.

**Known issue:** On macOS, copying from sandboxed pi.dev to the system clipboard
does not work in Terminal.app because it does not support OSC 52 clipboard writes.
Use a terminal that supports OSC 52, such as iTerm2, kitty, or Alacritty.

If you think you only live once (or don't have `wtmcp` or `arapuca` installed):

```
arkai agent start \
    -m hf:ibm-granite/granite-4.1-8b-GGUF \
    -a opencode \
    --no-sandbox \
    --no-mcp
```

## Managing Models

Models are stored as GGUF files in `~/.local/share/arkai/models/`. The
`inference.model` config key refers to the filename within that directory.
Named model profiles can be kept in the same configuration file. Set
`inference.model` to the profile name; its `model` value is started and its
sampling values are passed to the selected backend:

```yaml
inference:
  model: balanced
  profiles:
    fast:
      model: hf:mlx-community/Llama-3.2-3B-Instruct-4bit
      temperature: 0.2
      top_k: 40
    balanced:
      model: granite-4.1-8b-instruct-Q6_K.gguf
      temperature: 0.7
      top_p: 0.9
      min_p: 0.05
      presence_penalty: 0.0
      frequency_penalty: 0.0
      repetition_penalty: 1.1
```

### Download from HuggingFace

```bash
arkai model download <provider>:<model-reference>
```

Downloads the model into Arkai's local provider cache. Every download must use
an explicit provider, such as `hf:owner/model` or `ollama:model:tag`. Some
repositories contain files that llama.cpp can use directly; others must be
converted to GGUF before they can be loaded locally.
Large model repositories may require the repository's configured Git large-file
support (`git lfs`) to materialize the weight files.

```bash
arkai model download hf:ibm-granite/granite-4.1-8b-instruct-GGUF
arkai model download ollama:llama3.2:latest
```

For Hugging Face GGUF repositories containing multiple quantizations, append the
quantization name to download only matching files (including all shards):

```bash
arkai model download hf:unsloth/Qwen-3.8-27B-GGUF:UD-Q4_K_M
```

After downloading model, use the same provider-qualified reference for inference,
for example `arkai agent start -m ollama:llama3.2:latest`.

### Convert to GGUF

```bash
arkai model convert <provider>:<model-reference> [-q QUANTIZATION] [-o OUTPUT]
```

Converts a cached provider model to GGUF format and places it in the models
directory. Use this only when the downloaded repository is not already supported
directly by llama.cpp.

| Flag | Default | Description |
|------|---------|-------------|
| `-q/--quantization` | `Q6_K` | Quantization level |
| `-o/--output` | `~/.local/share/arkai/models/MODEL-QUANTIZATION.gguf` | Output path |

Common quantization levels: `Q4_K_M` (smaller, faster), `Q5_K_M`, `Q6_K` (near original inference quality).

```bash
arkai model convert hf:ibm-granite/granite-4.1-8b-instruct-GGUF -q Q6_K
```

### List models

```bash
arkai model list
```

Shows locally converted and provider models in Arkai's model cache.

### Remove a model

```bash
arkai model remove <model-name>
```

Deletes a local GGUF file by filename (not full path).

```bash
arkai model remove granite-4.1-8b-instruct-Q6_K.gguf
```

## Running an Inference Server

`arkai inference` manages the configured OpenAI-compatible inference server.
The server runs in the background and stays running between agent sessions.

### Start

```bash
arkai inference start [--model NAME] [--gpu-layers N] [--context N] [--port N]
```

Starts `llama-server` on `localhost:8081` (default). CLI flags override the
config values for that run only.

| Flag | Description |
|------|-------------|
| `--model NAME` | Override model from config |
| `--gpu-layers N` | Override number of GPU layers (`-1` = all) |
| `--context N` | Override context window size in tokens |
| `--port N` | Override port from config |

> Note: use `hf:org/model` as the model value to load a HuggingFace model.

**GPU detection:** arkai auto-detects Metal (Apple Silicon), CUDA (NVIDIA),
ROCm (AMD), or falls back to CPU. Setting `gpu_layers: -1` in config offloads
all layers to GPU.

The inference server exposes an OpenAI-compatible API at `http://127.0.0.1:<port>/v1`.

### MLX-LM backend

Install the optional backend and select an MLX model from Hugging Face:

```bash
pip install -e '.[mlx]'
arkai model download hf:mlx-community/Llama-3.2-3B-Instruct-4bit
arkai inference start --model hf:mlx-community/Llama-3.2-3B-Instruct-4bit
```

```yaml
inference:
  backend: mlx
  model: hf:mlx-community/Llama-3.2-3B-Instruct-4bit
  port: 8081
```

Arkai resolves `hf:` models from its local provider cache before starting
MLX-LM, so the server does not download an already cached model. `gpu_layers`
and `context_size` are llama.cpp-specific settings and are not passed to MLX-LM.

### Splash inference server

Install Splash separately, then start it with a supported Hugging Face model:

```bash
arkai inference start --backend splash --model hf:incoai/Qwen3.8-27B-Splash
```

Alternatively, set `inference.backend: splash` and `inference.model` in your
configuration; `--backend` is only an override. `inference.path` can point to a
custom Splash executable. The
startup timeout includes downloading and loading; increase
`inference.startup_timeout` for a cold start if necessary. GPU-layer settings
are not passed to Splash, and configured server sampling defaults are rejected.

### Managing Splash models

Splash installations use native metadata and links to Hugging Face Hub snapshots,
not the Git-based provider cache. Arkai's model inventory reads native selection
links directly, independent of the installed Splash version. This fast listing
does not verify model artifacts. Download, resolve, remove, and prune use the
verified **Splash 1.1.0** installer interface to manage the target, required draft,
tokenizer, and applicable vision sources. The installer runs in Splash's bundled
Python; it does not start inference.

```bash
arkai model download --backend splash hf:incoai/Qwen3.8-27B-Splash
arkai model list
arkai model list --backend splash
arkai model resolve --backend splash hf:incoai/Qwen3.8-27B-Splash
arkai model remove --backend splash hf:incoai/Qwen3.8-27B-Splash
arkai model prune --backend splash --dry-run
```

Download, resolve, remove, and prune use the configured inference backend when
`--backend` is omitted. Unfiltered `model list` always includes local GGUF files,
provider-cached models, and installed Splash selections, regardless of the
configured backend or Splash version. It does not download or repair files. Splash
listing shows installation IDs and paths; artifact completeness is reported as
unverified. Use `--id ID` for resolution or removal when multiple native
installations match.

Arkai automatically sets both cache locations for downloads and serving:

| Variable | Default directory |
| --- | --- |
| `HF_HUB_CACHE` | `~/.local/share/arkai/models/splash/hub` |
| `SPLASH_WEIGHT_CACHE` | `~/.local/share/arkai/models/splash/weights` |

The lightweight selection links and metadata remain in Splash's native directory,
normally `~/Library/Application Support/Splash/models`. Existing downloads are not
moved automatically; their recorded paths remain valid. Git downloads are not
automatically shared with Hub downloads. Optional `inference.splash.hub_cache` and
`weight_cache` override the bulk cache locations. Download and serve operations
use identical paths and source-selection settings:

```yaml
inference:
  backend: splash
  model: hf:unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
  startup_timeout: 1800
  splash:
    max_memory: 28G
    kv_format: int8
    # revision: <commit>
    # language_only: true
    # draft_model: owner/compatible-draft
    # hub_cache: /custom/hub
    # weight_cache: /custom/weights
```

Revision, language-only, and draft overrides apply to upstream models, not legacy
packed `incoai/*-Splash` packages. Downloading sources does not prepare Metal
weights; the first serve may need additional time and disk space. Splash models
are used directly, without Arkai GGUF conversion.

Removal unlinks only the selected installation and its pins, preserving shared
source files. Stop all Splash servers and provisioning operations before removal
or pruning; native locks also protect externally started processes. Pruning
collects only unreferenced data in the default Arkai-owned caches and never prunes
custom/external caches. Native installation pins and unknown Hub references are
retained; a legacy package's unused `main` cache reference does not prevent
collection after uninstall. Prepared entries are retained while any native model
remains installed because they may be shared by content. After all selections are
removed, pruning can collect recognized prepared entries with removed sources.
Unknown metadata and cache entries are never deleted.

For a nonstandard package layout, set `inference.splash.installation_root` to the
directory containing `release.json` and `install/models.py`; `python_path` may
override its bundled Python executable. Unknown installer versions fail clearly
instead of attempting unsafe filesystem operations. Agent and benchmark behavior
is unchanged by this model-management integration.

### Status

```bash
arkai inference status
```

Shows: running/stopped, PID, model, port, GPU layers, context size, health,
and detected GPU type.

### Stop

```bash
arkai inference stop
```

Terminates the server and cleans up PID and state files.

## Running an Agent

`arkai agent start` is the primary command. It auto-starts the inference server if
not already running, starts wtmcp (unless `--no-mcp`), launches the agent, and
tears down services on exit. OpenCode, Pi, and OMP can use a cloud provider instead
of the local inference server with `--cloud`.

```bash
arkai agent start [-a AGENT] [-m MODEL] [options]
arkai agent start -a pi --cloud openai
arkai agent start -a omp --cloud openai
arkai agent start -a opencode --cloud openai
arkai agent start -a pi --cloud vertex
arkai agent start -a omp --cloud vertex
arkai agent start -a opencode --cloud vertex
```

**Supported agents:** `opencode`, `crush`, `claude`, `pi`, `omp`

### Flags

| Flag | Description |
|------|-------------|
| `-a/--agent AGENT` | Override agent from config |
| `-m/--model MODEL` | Override the local model, or select the cloud agent model with `--cloud` |
| `--cloud SERVICE` | Use a cloud AI provider instead of local inference (`openai` or `vertex`; OpenCode, Pi, and OMP) |
| `--no-mcp` | Skip wtmcp entirely |
| `--no-sandbox` | Skip sandbox for this run |
| `-s/--sandbox PROFILE` | Use a specific sandbox profile for this run |
| `-I/--keep-inference` | Leave inference server running after exit |
| `-M/--keep-mcp` | Leave wtmcp running after exit |
| `-v/--volume PATH` | Mount extra volume in sandbox (repeatable) |
| `-e/--env KEY=VALUE` | Set env var in sandbox (repeatable) |
| `--cwd PATH` | Override directory mounted as cwd in sandbox |

> Note: use `hf:org/model` as the model value to load a HuggingFace model.

With `--cloud openai`, Arkai does not start or require the local inference
engine. Pi and OMP use their user-level agent configuration and saved OpenAI
credentials; OpenCode uses its configured model and native OpenAI provider. If
no OpenAI key is configured there or in `OPENAI_API_KEY`, Arkai prompts for it
once and saves it for future launches (Pi `auth.json`, OMP's global `models.yml`,
or OpenCode's global `auth.json`). In Pi, choose an OpenAI model with `/model`
and press `Ctrl+S` in the picker to save it as the default for future launches;
OMP uses its configured default model, and OpenCode uses its normal model
selection and settings.
With `--cloud vertex`, authenticate with Google Application Default Credentials
(`gcloud auth application-default login`) or set `GOOGLE_APPLICATION_CREDENTIALS`,
and configure `GOOGLE_CLOUD_PROJECT` plus `GOOGLE_CLOUD_LOCATION` (or the
`GOOGLE_VERTEX_PROJECT` and `GOOGLE_VERTEX_LOCATION` aliases) as needed. Pi and
OMP select their native Vertex providers; OpenCode uses the Vertex model
selected in its normal model settings. Pi registers a separate `google-vertex-anthropic`
provider and refreshes its Claude model catalog from Models.dev daily, so Haiku,
Sonnet, Opus, and newer catalog entries are available in `/model`; existing Pi
models are preserved, and a cached catalog is used when refresh fails. Sandboxed
runs mount an ADC credential file read-only when one is available and permit
Vertex AI and OAuth endpoints.
With `--cloud`, `-m/--model` selects the model for the cloud provider and is passed
to the agent without being treated as a local inference model (it will not be
downloaded). Use the model identifier expected by the selected agent/provider.
Without `-m`, the agent's normal cloud model setting remains in effect. Pi and OMP
use their global agent directories in the sandbox; OpenCode's OpenAI API key is
passed through the sandbox environment. On Linux, the network deny rule remains
enabled with provider HTTPS endpoints allowlisted; macOS uses Arapuca's baseline
profile. Use `--no-sandbox` to run without those boundaries.

### Lifecycle

On exit: wtmcp stops automatically. The inference server also stops unless
`--keep-inference` is passed. Use `--keep-inference` when you want the server
warm for the next session.

## Managing Plugins (wtmcp)

wtmcp is an MCP server that exposes tools (plugins) to the agent over HTTP.
`arkai agent start` manages wtmcp automatically, but you can also control it directly.

### Enabling and disabling plugins

These commands edit `.arkai.yaml` in the current directory (must exist — run
`arkai config init` first):

```bash
arkai wtmcp enable <plugin>    # add plugin to .arkai.yaml
arkai wtmcp disable <plugin>   # remove plugin from .arkai.yaml
```

### Plugin list semantics

`wtmcp.plugins` in `.arkai.yaml` is a list of plugin names:
- Key **absent**: inherit plugins from user config
- `plugins: []`: explicitly disable all plugins
- `plugins: [workspace, terminal]`: use only these plugins

### List plugins

```bash
arkai wtmcp list [--port N]
```

Shows all discovered plugins with status:
- 🟢 enabled (in config and discovered by wtmcp)
- ⚪ discovered but not enabled
- 🔴 enabled but not discovered by wtmcp

Requires a running wtmcp instance. If multiple instances are running, specify `--port`.

### Server lifecycle

```bash
arkai wtmcp start [--port N] [--enable PLUGIN] [--disable PLUGIN]
arkai wtmcp stop  [--port N]
arkai wtmcp status [--port N]
```

`--enable`/`--disable` on `start` override the project config plugins for that
run only. `--port` is required for `stop` and `list` when multiple instances
are running.

## Sandboxing

arkai can wrap agent execution in an arapuca sandbox. The sandbox restricts
what the agent can access:

- **Filesystem:** only explicitly mounted paths are visible
- **Network (Linux):** only the inference server and wtmcp ports are reachable;
  all other network access is denied
- **Network (macOS):** `baseline` seccomp profile is applied

OpenCode starts a private local server for each invocation. Arkai therefore
requests Arapuca's `--allow-network-listeners` capability for sandboxed OpenCode
runs. On macOS, Seatbelt cannot reliably constrain listeners to loopback, so a
sandboxed OpenCode process may bind to a non-loopback interface. Use this only
on trusted networks; disable the sandbox explicitly if you do not accept that
macOS limitation or use another isolation boundary.

Sandboxed OpenCode also skips project-level OpenCode configuration and rules
(`opencode.json`, `.opencode/`, and project `AGENTS.md` discovery). Arkai provides
the per-run OpenCode config, and project discovery otherwise traverses outside
the mounted workspace.

The current working directory is mounted read-write by default (override path with
`--cwd`).

### Profiles

Named profiles are presets stored in your user config under
`sandbox.profiles.<name>`. When `arkai agent start` runs, it uses the active profile
(or the root `sandbox` defaults if no profile is active).

Profile names must be alphanumeric + underscores. The names `default` and
`active` are reserved.

### Managing profiles

```bash
# List all profiles and which is active
arkai sandbox list

# Show full details of a profile
arkai sandbox show <profile-name>
arkai sandbox show default   # root sandbox defaults
arkai sandbox show active    # currently active profile

# Create a profile
arkai sandbox create <name> [--from PROFILE] [--memory MB] [--cpus N] \
    [--pids N] [--timeout S] [-v /path[:ro]] [-e KEY=VALUE]

# Delete a profile
arkai sandbox delete <name>

# Promote a profile's settings to root defaults
arkai sandbox set-default <name>

# Set the active profile (used by arkai agent)
arkai sandbox active <name>

# Clear the active profile (use root defaults)
arkai sandbox active
```

### Example workflow

```bash
# Create a restricted profile for untrusted code
arkai sandbox create restricted --memory 1024 --cpus 1 --pids 128

# Make it the active profile
arkai sandbox active restricted

# Run the agent (uses restricted profile automatically)
arkai agent start

# Override to a different profile for one run
arkai agent start -s dev
```

### Per-run overrides

These flags on `arkai agent start` override the active profile for that run only:

| Flag | Effect |
|------|--------|
| `-s/--sandbox PROFILE` | Use a specific profile |
| `--no-sandbox` | Disable sandbox entirely |
| `-v/--volume PATH` | Add an extra volume mount |
| `-e/--env KEY=VALUE` | Add an env var inside the sandbox |
| `--cwd PATH` | Override cwd mounted in sandbox |

## Configuration Reference

Configuration is YAML. Precedence (lowest to highest):

1. Built-in defaults
2. User config: `~/.config/arkai/arkai.yaml`
3. Project config: `.arkai.yaml` in current directory
4. CLI flags

```bash
arkai config init              # scaffold .arkai.yaml in current directory
arkai config validate          # validate .arkai.yaml
arkai config validate --file PATH  # validate a specific file
```

### `agent`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `agent.name` | string | `opencode` | Agent to use: `opencode`, `crush`, `claude`, `pi`, or `omp` |
| `agent.path` | string | *(agent name)* | Custom path to agent binary |
| `agent.mcp` | bool | `true` | Whether to start wtmcp for agent sessions |

### `inference`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inference.model` | string | — | Local model filename, or `hf:<repo>` for a HuggingFace model |
| `inference.profiles` | mapping | — | Named model profiles; each profile has `model` and optional sampling settings |
| `inference.backend` | string | `llama-cpp` | `llama-cpp`, optional `mlx`, or `splash` |
| `inference.path` | string | backend-dependent | Path to `llama-server`, `mlx_lm.server`, or `splash` |
| `inference.port` | int | `8081` | Port for inference server (1024–65535) |
| `inference.gpu_layers` | int | `-1` | GPU layers to offload (`-1` = all) |
| `inference.context_size` | int | `65536` | Context window size in tokens |
| `inference.startup_timeout` | int | `600` | Maximum seconds to wait for the server to become ready |

The sampling settings can also be placed directly under `inference` to provide
defaults for the selected model: `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, and `repetition_penalty`. Profile values
override these defaults.

Model sampling settings are `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, and `repetition_penalty`. They are
translated to the corresponding `llama-server` or `mlx_lm.server` CLI options.

Inference backends are registered command builders. The manager supplies the
common model, port, GPU-layer, and context settings, then handles the process
lifecycle and OpenAI-compatible REST API. Add a backend by implementing
`InferenceBackend.build_command()` and registering it with
`register_backend()`.

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
