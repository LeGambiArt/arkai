# MCP plugins

Arkai can start and supervise `wtmcp` for an agent session. Install
`wtmcp` separately and configure its executable or port if the defaults
do not fit your setup. `arkai agent start` enables the integration by
default; use `--no-mcp` to skip it for one run.

Project plugin selections are stored in `.arkai.yaml`. A non-empty
`plugins` list adds those entries to the merged wtmcp config. If no
plugins or explicit base config are provided, Arkai leaves wtmcp's own
defaults untouched. For example:

```yaml
wtmcp:
  plugins: [workspace, terminal]
```

Manage project selections and inspect a running server with:

```bash
arkai config init
arkai wtmcp enable workspace
arkai wtmcp disable terminal
arkai wtmcp start
arkai wtmcp list
arkai wtmcp status
arkai wtmcp stop
```

`wtmcp enable` and `disable` edit the current project's `.arkai.yaml`;
create it with `arkai config init` first. Use `arkai wtmcp --help` for
port and per-run plugin overrides. The [MCP design](../design/mcp-integration.md)
describes config merging and process supervision.
