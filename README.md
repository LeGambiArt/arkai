# arkai

Arkai is a command-line tool for running coding agents with local AI
models. It connects an agent to an inference server and can manage MCP
tools and sandboxed execution for a project. Cloud providers are also
supported when you prefer not to run inference locally.

Use Arkai when you want one command to start an agent with a selected
model and the supporting services it needs.

## Requirements

- Python 3.10–3.14
- `git` for Git-backed model downloads; Hugging Face repositories with
  large weight files may also require Git LFS.
- For local inference, an installed inference backend. The default is
  `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp).
- An installed supported agent CLI, such as OpenCode, Crush, Claude Code,
  Codex, Pi, or OMP.
- For the default sandboxed and MCP-enabled agent workflow, `arapuca` and
  `wtmcp`. Both integrations can be skipped for a run, but doing so
  removes sandbox protection or MCP tools respectively.

## Install

You can install from PyPI:

```bash
python -m pip install arkai
```

To install from the GitHub repository:

```bash
git clone https://github.com/LeGambiArt/arkai.git
cd arkai
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Optional extras are described in the [user guides](docs/guides).

## Start using Arkai

Download a model and start an agent (this example skips MCP and
sandboxing, so use it only in a workspace you trust):

```bash
arkai model download hf:ibm-granite/granite-4.1-8b-GGUF
arkai agent start \
  --agent opencode \
  --model hf:ibm-granite/granite-4.1-8b-GGUF \
  --no-mcp \
  --no-sandbox
```

This requires `llama-server` and the OpenCode CLI to be installed. For
the sandboxed, MCP-enabled workflow, install `arapuca` and `wtmcp`, then
omit `--no-mcp` and `--no-sandbox`. Arkai can also start a configured
agent with `arkai agent start`; use `arkai --help` to explore commands.

To create a project configuration and validate it:

```bash
arkai config init
# Edit .arkai.yaml, for example to set inference.model.
arkai config validate
```

For troubleshooting, put `--debug` before the command, for example
`arkai --debug agent start`.

## More

- [User guides](https://github.com/LeGambiArt/arkai/tree/main/docs/guides) —
  configuration, models, agents, sandboxing, plugins, and RAG.
- [Design documents](https://github.com/LeGambiArt/arkai/tree/main/docs/design) —
  architecture and implementation details, one feature per document.
- [Contributing](https://github.com/LeGambiArt/arkai/blob/main/CONTRIBUTING.
md)

Arkai is licensed under [Apache 2.0](https://github.com/LeGambiArt/arkai/blob/main/LICENSE).
