# Running agents

`arkai agent start` is the interactive workflow. It reads configuration,
starts local inference if needed, starts MCP integration unless disabled,
launches the selected agent, and cleans up services it started when the
agent exits.

```bash
arkai agent start --agent opencode --model
hf:ibm-granite/granite-4.1-8b-GGUF
```

The supported agent names are `opencode`, `crush`, `claude`, `codex`,
`pi`, and `omp`. Their own executables must be installed, except for Pi
and OMP which can also be installed with [`arkai tools`](tools.md).

For a non-interactive run, use `arkai agent prompt`:

```bash
arkai agent prompt --agent codex "Summarize this project"
```

Use `--help` for the available options. Common overrides include
`--model`, `--port`, `--backend`, `--no-inference`, `--no-mcp`,
`--no-sandbox`, `--sandbox PROFILE`, `--volume`, `--env`, and `--cwd`.
Disabling sandboxing removes that execution boundary; do so only when
appropriate for the workspace. See [sandboxing](sandboxing.md) and [MCP
plugins](mcp.md) for details.

## Sessions

Interactive sessions started by Arkai can be listed and resumed:

```bash
arkai agent start --list-session
arkai agent start --session refactor-auth
arkai agent start --session
```

Session names are scoped to a working directory. Arkai stores session
metadata and private native state under its state directory; the
transcript itself remains in the agent's native store. When resuming,
Arkai restores the saved working directory and launch settings, but uses
the current configuration and available inference service rather than
pinning the original model. Sandbox environment overrides must be passed
again. See the [session design](../design/agent-sessions.md).

Codex's local inference mode requires a backend with the Responses API at
`/v1/responses`; verify backend compatibility before using it. With
Arkai's sandbox enabled, Codex's nested command sandbox is configured to
avoid nesting two sandbox layers. `--no-sandbox` leaves Codex's native
sandbox settings in place.

## Lifecycle

Arkai stops MCP and inference processes that it started when an agent
exits.  It does not claim ownership of an already-running inference
service. Cloud runs do not need local inference. For provider selection
and credentials, see [Cloud providers](cloud-providers.md).
