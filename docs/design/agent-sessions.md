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
directory, timestamps, status, launch options, an `agents_used` list, and
an optional native session identifier. The per-session `native` directory
is available to adapters that need private agent state.

`agents_used` records every agent that has completed a run inside the
record and therefore has native state under its `native` directory; it
gates native resume. Records written before this field existed are
migrated lazily to `[<recorded agent>]`, which is exact for their
single-agent lifetime.

Directories are created with owner-only permissions; metadata writes use
a private temporary file and atomic replacement. Session transcripts
remain in the native agent's normal session store.

## Lookup and resume behavior

The agents `omp`, `pi`, and `opencode` (`SHARED_SESSION_AGENTS`) share
one session record per working directory: any of them may resume the
record, launching the requested agent into it while each agent keeps its
own native thread. Sharing is at the record level only; native transcript
formats are not interchangeable.

- Full Arkai IDs can be loaded directly.
- Names are resolved within the working directory and an agent filter
  (`agents=...`); a shared agent matches the whole shared set, other
  agents match exactly themselves. Ambiguous names require the full ID.
- `find_latest_session(agents, working_directory)` returns the record
  with the highest `updated_at`, used by plain `arkai agent start`.
- A plain `arkai agent start` (no `--session`) with a shared effective
  agent auto-resumes the directory's most recent shared record and
  creates a new one when none exists. Other agents always start fresh
  unless a session is named.
- Listing does not start agent infrastructure.
- Resume restores the recorded working directory and saved launch
  options, then uses current Arkai configuration and available inference
  services. A session is not pinned to the original model or server.
- Native resume flags (`--continue`, `--resume`, `--session-id`) are
  passed only when the selected agent appears in `agents_used`; adapters
  additionally withhold them when no native state is present.
- Prompt history follows the session where the agent stores it in the
  agent directory rather than the session: OMP's `history.db` is seeded
  from `native/omp-history/` before launch and copied back after the run
  (local and custom-cloud runs; OpenAI and Vertex runs keep OMP's global
  history). Pi rebuilds its input history from the resumed session's own
  user messages, so no extra storage is needed.
- Sandbox environment overrides are not persisted; callers must supply
  them again when resuming.
- `-S` / `--reset-session` deletes the directory's current session (the
  same record auto-resume would use) via `delete_session`, which removes
  the record's metadata and its entire `native` directory including
  transcripts and OMP prompt history, then starts a new record in its
  place without resuming older ones. It is mutually exclusive with
  `--session` and `--list-session`, and only targets shared-set records;
  with a non-shared effective agent it is a plain fresh start.

## Implementation map

- Metadata, lookup, and permissions: `arkai/agent_session.py`
- Session launch/resume integration: `arkai/agent.py` and agent adapters
- User behavior: `features/agent.feature`
- Unit coverage: `tests/unit/test_agent_session.py`
- User instructions: [Running agents and sessions](../guides/agents.md)
