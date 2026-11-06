# Design: agent orchestration

## Scope

The agent orchestration feature prepares and runs one supported agent
process.  Its responsibilities are to resolve the effective launch
configuration, coordinate local services when requested, create an
agent-specific launch specification, and clean up resources it owns. MCP
internals, sandbox policy, cloud credential resolution, and session
persistence are documented separately.

## Flow

1. `arkai.cli` dispatches `agent start` or `agent prompt` to
   `arkai.agent`.
2. The orchestrator merges configuration with CLI overrides and validates
   the selected agent and required inference settings.
3. It resolves the agent adapter and prepares the local or cloud provider
   context. Local inference is reused when healthy; otherwise Arkai starts
   it unless the caller requested `--no-inference`.
4. MCP is started unless disabled. The chosen Arapuca profile and mounts
   are resolved before the agent is launched.
5. The adapter produces an `AgentLaunchSpec` containing the command,
   environment, working directory, and temporary cleanup paths.
6. On exit or launch failure, the orchestrator stops only the MCP or
   inference services it started and removes adapter-owned temporary
   files.

## Boundaries and invariants

- Agent-specific configuration belongs in adapter modules
  (`agent_opencode`, `agent_codex`, `agent_pi`, and the other `agent_*`
  modules), not in the central process lifecycle.
- A service already running before the launch is not considered
  Arkai-owned and is not stopped as part of that launch.
- Cloud launches bypass local inference. Provider setup remains delegated
  to `cloud_provider` and `cloud_auth` plus agent adapters.
- Interactive and prompt launches share configuration and infrastructure
  setup, but prompt mode captures output and does not expose the caller's
  TTY to the child process.

## Implementation map

- Core flow and cleanup: `arkai/agent.py`
- CLI: `arkai/cli.py`, `features/agent.feature`
- Agent-specific command/config translation: `arkai/agent_*.py`
- Unit coverage: `tests/unit/test_agent_launch.py`,
  `tests/unit/test_agent_tui.py`, and related agent tests
- Related designs: [MCP integration](mcp-integration.md),
  [sandboxing](sandboxing.md), [cloud providers](cloud-providers.md),
  and [agent sessions](agent-sessions.md)
