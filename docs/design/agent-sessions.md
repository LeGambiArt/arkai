# Design: agent sessions

## Scope

Arkai session tracking provides a unified way to list and resume
interactive sessions that were launched through Arkai. It stores launch
metadata and agent-specific native session state; it does not replace an
agent's transcript format or session database.

## Data model and storage

`arkai.agent_session` stores one JSON metadata file per Arkai session
under the Arkai state directory's `agent-sessions` folder. A record
includes an Arkai UUID, optional project-scoped name, agent, working
directory, timestamps, status, launch options, and an optional native
session identifier. The per-session `native` directory is available to
adapters that need private agent state.

Directories are created with owner-only permissions; metadata writes use
a private temporary file and atomic replacement. Session transcripts
remain in the native agent's normal session store.

## Lookup and resume behavior

- Full Arkai IDs can be loaded directly.
- Names are resolved within the working directory and optional agent
  filter. Ambiguous names require the full ID.
- Listing does not start agent infrastructure.
- Resume restores the recorded working directory and saved launch
  options, then uses current Arkai configuration and available inference
  services. A session is not pinned to the original model or server.
- Sandbox environment overrides are not persisted; callers must supply
  them again when resuming.

## Implementation map

- Metadata, lookup, and permissions: `arkai/agent_session.py`
- Session launch/resume integration: `arkai/agent.py` and agent adapters
- User behavior: `features/agent.feature`
- Unit coverage: `tests/unit/test_agent_session.py`
- User instructions: [Running agents and sessions](../guides/agents.md)
