# Design: MCP integration

## Scope

Arkai's MCP integration starts and supervises `wtmcp` for agent sessions
and can inject selected plugin entries into a generated wtmcp
configuration. It does not implement MCP protocol handling itself;
`wtmcp` owns that service.

## Configuration flow

The wtmcp executable, port, working directory, and optional base config
are resolved from Arkai configuration. Project plugin selections and
per-run enable/disable overrides are combined. When a base configuration
or selected plugins need merging, Arkai writes a temporary configuration
containing the MCP server entries and passes it to wtmcp. With no
Arkai-specific additions, wtmcp is allowed to use its own configuration
defaults.

## Process lifecycle

The supervisor launches the wtmcp child, publishes its PID, forwards
termination signals, and removes its PID file at exit. Arkai records
whether it started the service so an existing instance is not stopped by
an unrelated agent launch. Temporary merged configuration and process
state are cleaned up when the service stops.

## Implementation map

- CLI and plugin config: `arkai/wtmcp.py`
- Agent lifecycle integration: `arkai/agent.py`
- Child supervision: `arkai/wtmcp_supervisor.py`
- Unit/BDD coverage: `tests/unit/test_wtmcp.py`,
  `features/wtmcp.feature`, `features/wtmcp_server.feature`
- User instructions: [MCP plugins](../guides/mcp.md)
