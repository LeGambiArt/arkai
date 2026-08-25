# aitool

Local AI inference workflow management for Linux and macOS.

Orchestrates inference servers (llama.cpp), wtmcp integration, and AI agents (opencode, crush, claude) with configuration layering and sandboxed execution.

## Installation

### Requirements

- Python 3.9+
- jq
- git
- bash (modern version)
- hf (HuggingFace CLI)
- llama.cpp (`llama-server` binary)
- wtmcp (optional; required for agent integration)
- arapuca (optional; for sandboxed execution)
- Agent binaries: opencode, crush, or claude (depending on which agents you use)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/aitool.git
cd aitool
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the package:

```bash
pip install -e ".[dev]"
```

4. Initialize the system:

```bash
aitool engine start
```

This scaffolds the user config at `~/.aitool/aitool.yaml`.

## Quick Start

### Create a Project

```bash
mkdir my-ai-project
cd my-ai-project
cat > .aitool.yaml << 'EOF'
agent:
  name: opencode

inference:
  model: granite-4.1-8b-instruct-Q6_K.gguf

wtmcp:
  plugins: [workspace, terminal]
EOF

git init
```

### Run an Interactive Agent

```bash
aitool agent
```

Launches opencode with local inference and wtmcp tools.

### Run a One-Off Prompt

```bash
aitool prompt "Explain how to use this tool"
```

Runs the prompt and exits (wtmcp is stopped).

### Manage Models

```bash
# Download a model
aitool model download ibm-granite/granite-4.1-8b-instruct-GGUF

# List available models
aitool model list

# Remove a model
aitool model remove granite-4.1-8b-instruct-Q6_K.gguf
```

### Manage Inference Server

```bash
# Start inference server
aitool engine start --model granite-4.1-8b-instruct-Q6_K.gguf

# Check status
aitool engine status

# Stop inference server
aitool engine stop
```

## Configuration

Configuration is layered:

1. **User config** (`~/.aitool/aitool.yaml`): System defaults
2. **Project config** (`.aitool.yaml`): Project-specific overrides
3. **CLI flags**: Highest priority

### Example User Config

```yaml
agent:
  name: opencode
  # path: /custom/path/to/opencode  # Optional: custom agent binary path

inference:
  backend: llama-cpp
  path: /usr/local/bin/llama-server  # Optional: custom binary path
  port: 8081
  gpu_layers: -1
  context_size: 65536

wtmcp:
  path: /usr/local/bin/wtmcp         # Optional: custom binary path
  port: 8080
  workdir: ~/.config/wtmcp           # Optional: custom config location

sandbox:
  enabled: true
  path: /usr/local/bin/arapuca       # Optional: custom binary path
  memory_mb: 2048
  cpus: 200
  pids: 256
  timeout: 0
```

### Example Project Config

```yaml
inference:
  model: granite-4.1-8b-instruct-Q6_K.gguf

wtmcp:
  plugins: [workspace, terminal, github]
```

## Commands

### `aitool model`

- `download <hf-repo/model>`: Download model from HuggingFace
- `list`: List available models
- `remove <model-name>`: Delete a model

### `aitool engine`

- `start`: Start inference server
- `stop`: Stop inference server
- `status`: Show server health and config

### `aitool agent`

Start interactive agent session (requires TTY).

```bash
aitool agent [--agent <name>] [--model <name>]
```

### `aitool prompt`

Run one-off prompt.

```bash
aitool prompt "Your prompt here" [--agent <name>] [--model <name>]
```

### `aitool wtmcp`

- `list`: Show available plugins
- `--enable <plugin>`: Add plugin to project config
- `--disable <plugin>`: Remove plugin from project config

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Behavior tests (BDD)
behave

# Coverage report
pytest --cov=aitool --cov-report=html
```

### Code Quality

```bash
# Format code
ruff format aitool/

# Lint
ruff check aitool/

# All checks (via pre-commit)
pre-commit run --all-files
```

## Supported Agents

- **opencode**: Web-based agent (opencode.ai)
- **crush**: Terminal agent (Charm)
- **claude**: Claude Code CLI with local inference

See `docs/AGENTS.md` for agent-specific setup and troubleshooting.

## Architecture

- **Config system**: User → Project → CLI precedence with recursive merging
- **Service lifecycle**: Inference server stays running; wtmcp stops after each session
- **Sandbox execution**: Optional arapuca sandbox for agents
- **BDD testing**: All features covered by Gherkin behavior tests

## Troubleshooting

### Binary Not Found

If a tool (llama-server, wtmcp, etc.) is not in PATH, configure its path in `~/.aitool/aitool.yaml`:

```yaml
inference:
  bin: /usr/local/bin/llama-server
wtmcp:
  bin: /home/user/custom/wtmcp
sandbox:
  bin: /opt/arapuca/bin/arapuca
```

### Port Already in Use

Change the port in your config:

```yaml
inference:
  port: 8082
wtmcp:
  port: 8081
```

### Model Not Found

Ensure the model file exists in `~/.local/share/aitool/models/`:

```bash
aitool model download <hf-repo/model>
```

## License

Apache 2.0 — See LICENSE file.

## Contributing

See CLAUDE.md and AGENTS.md for development guidelines.
