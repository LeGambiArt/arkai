# Configuration

Arkai reads YAML configuration from built-in defaults, the user
configuration, and (when present) `.arkai.yaml` in the current project.
Project values override user values; command-line options override
configuration for that invocation.

- User configuration: `~/.config/arkai/arkai.yaml`
- Project configuration: `.arkai.yaml`

Initialize a project file and validate it after editing:

```bash
arkai config init
arkai config validate
arkai config validate --file /path/to/arkai.yaml
```

`config init` refuses to replace an existing `.arkai.yaml`. Use
`config init --global` to initialize the user configuration file instead of
the project file; `--project` is the default scope, and `--global --project`
together is an error. The complete key reference, with defaults and
descriptions, is in [arkai-full-config.yaml](../arkai-full-config.yaml).

## Editing configuration interactively

`arkai config edit` opens a terminal field editor for the configuration
file. Fields with a fixed set of allowed values (agent name, backend, mode,
booleans) are picked from a list, never typed; free-form fields validate as
you submit them.

```bash
arkai config edit            # .arkai.yaml if it exists, else the global file
arkai config edit --project  # .arkai.yaml
arkai config edit --global   # ~/.config/arkai/arkai.yaml
```

With no flag, the project file is edited when it exists, otherwise the
global file. `--project` requires an existing `.arkai.yaml`; the global file
is created on save if missing. `--global` and `--project` together is an
error. When editing the global file, the project configuration is ignored:
the editor shows and saves only global values, so a project override stays
in `.arkai.yaml` and keeps winning over your global edit until you remove
it there.

Each field row shows the effective value and a source tag:

- `[default]` — no file sets the key; the built-in default applies
- `[global]` — the key comes from the user configuration file
- `[project]` — the key comes from `.arkai.yaml` and wins over the others
  (shown only when editing the project file)

Key bindings: `↑`/`↓` (or `j`/`k`) move, `Enter` edits the selected field,
`r` resets it to the default by deleting the key from the file being edited,
`s` saves and exits, `q` quits without saving, `?` shows help. Choice and
boolean fields offer `(unset)` when the default is empty; selecting it
removes the key. For free-form text fields, submitting an empty value
deletes the key.

Saving rewrites the whole file, so YAML comments are not preserved. Other
keys present in the file (plugin lists, agent cloud providers) pass through
untouched.

### Editing profiles

The field list includes a "Model profiles" row after the inference fields,
a "Sandbox profiles" row after the sandbox fields, and a "Cloud custom
providers" row after the agent fields. `Enter` on any of them opens the
profile list for the file being edited, where `a` adds a profile, `d`
deletes the selected one (type `y` to confirm), `Enter` edits its members,
and `Esc` goes back. Deleting a profile that is also defined in another
configuration layer removes it only from the file being edited.

Profile names are alphanumeric plus underscore; `default` and `active` are
reserved. Adding a profile prompts for its name and then opens the full
member editor with every option listed; a new profile whose required
members (the model, for model profiles) are still empty when you leave the
editor is discarded. Sandbox profiles start empty. Sandbox members left
unset inherit the root `sandbox` values and are shown tagged `[default]`;
setting a member stores it in the profile, and `r` on a member removes it
again.

Sandbox profiles also expose `volume` and `environment` editors. Volumes are
entered as `/path` or `~/path`, optionally suffixed `:ro`; duplicates with
conflicting flags are rejected. Environment entries are entered as
`KEY=VALUE` and overwrite an existing key.

Cloud custom providers edit `agent.cloud.custom` (see
[Cloud providers](cloud-providers.md)). The root fields — `default
provider`, `custom protocol`, `custom base url`, and `custom api key` —
appear in the Agent group and set the inherited defaults for every
provider; `r` on any of them removes the key. The add prompt asks for a
provider ID (letters, digits, `_`, `-`); members are `name`, `protocol`
(picked from the supported protocols), `base_url` (must be HTTPS without
credentials), and `default_model`. Members left unset inherit the
`agent.cloud.custom` root defaults; `r` on a member removes the override.
The `api_key` row (root or per provider) opens a source picker —
`environment` prompts for the variable name, `keychain` for the service
and account, and `(unset)` removes the key. The provider list requires a
valid custom cloud configuration on save: keep at least one provider with
a resolvable `base_url` (directly or through the root default), and set
`default_provider` when more than one provider exists.

### Custom keys

Keys the editor does not manage — plugin settings, model-specific options,
anything — are listed inside the group that owns their namespace (an
unknown `inference.*` key appears with the inference fields). Keys in
namespaces no group owns are listed under "Other". Pressing `a` adds a key
to the group the cursor is in: the prompt pre-fills the group's namespace,
so keys land in the group or subgroup they were added from. Editing the
pre-filled key away (to reach a namespace no group owns) is allowed.
Custom keys also work inside profiles: the profile editor dialog (model
profiles, sandbox profiles, cloud custom providers) lists any key in the
profile it does not manage below the member rows, where `Enter` re-edits
the value, `d` (or `r`) deletes it, and `a` adds a new key relative to the
profile (`labels.env` becomes `sandbox.profiles.dev.labels.env`) — deleting
the last key in a subgroup removes the subgroup. Member rows the dialog
itself manages (like `cpus` or `volume`) cannot be shadowed by a custom
key of the same name.

The value is stored exactly as parsed from YAML, with no schema validation,
so unknown keys round-trip untouched. `Enter` on an existing custom key
re-edits its value (mappings and lists are pre-filled as flow YAML), an
empty value removes the key from the file, and `d` (or `r`) deletes it.
Custom values are not validated: check the
consuming feature's documentation for the expected shape.

`arkai config init --tui` initializes a new file (project by default,
`--global` for the user file) and opens the same editor pre-filled with the
seed values so it can be customized before it is written. Aborting with `q`
writes nothing and prints a cancellation warning; rerun the command to try
again.

## Common settings

| Key | Purpose |
| --- | --- |
| `agent.name` | Default agent: `opencode`, `crush`, `claude`, `codex`, `pi`, or `omp`. |
| `agent.mode` | Inference mode: `local` (default) or `cloud`. A `--cloud` option selects cloud mode for that run; `--local` forces local mode. |
| `agent.cloud.default` | Cloud service for `agent.mode: cloud`; if omitted, Arkai infers it when exactly one cloud configuration is present. |
| `agent.path` | Optional path to the agent executable. |
| `agent.mcp` | Whether agent launches start the configured MCP integration; defaults to enabled. |
| `inference.backend` | `llama-cpp` (default), `mlx`, `splash`, or `strata`. |
| `inference.model` | Model reference or a name in `inference.profiles`. |
| `inference.profiles` | Mapping of profile names to a model and optional sampling settings. |
| `inference.path` | Optional executable path for the selected backend. |
| `inference.port` | Local inference port; defaults to `8081`. |
| `inference.gpu_layers` | GPU layers for backends that support the setting; defaults to `-1` (all). |
| `inference.context_size` | Context window; accepts a positive integer or `k`/`m` suffix. |
| `inference.startup_timeout` | Readiness timeout in seconds; defaults to `600`. |
| `wtmcp.enabled` | Set `false` to skip starting wtmcp and MCP agent configuration; defaults to `true`. |
| `wtmcp.path`, `wtmcp.port` | MCP server executable and port; default port is `8080`. |
| `wtmcp.workdir`, `wtmcp.config` | Optional working directory and base wtmcp config path. |
| `wtmcp.plugins` | Project-level plugin entries injected into the wtmcp config. |
| `sandbox.path` | `arapuca` executable path. |
| `sandbox.profiles` | Named resource, volume, and environment presets for sandbox runs. |
| `sandbox.active_profile` | Profile selected for agent launches unless overridden with `--sandbox`. |
| `sandbox.volume`, `sandbox.environment` | Root-level extra mounts and environment values. |
| `vectordb.*` | ChromaDB executable, port, and data directory. |
| `rag.*` | Document chunking, embedding model, and retrieval defaults. |

Sampling settings (`temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, and `repetition_penalty`) can be
set at `inference` level or overridden per model profile. Not all
backends support all settings. A custom cloud provider can be configured
under `agent.cloud.custom`; see [Cloud providers](cloud-providers.md).
See the [models and inference guide](models-and-inference.md) for
backend-specific behavior and the [configuration design](../design/configuration.md)
for implementation details.

## Example

```yaml
agent:
  name: opencode

inference:
  backend: llama-cpp
  model: hf:ibm-granite/granite-4.1-8b-GGUF
  port: 8081
  context_size: 65536

wtmcp:
  port: 8080

sandbox:
  memory_mb: 2048
  cpus: 2
  pids: 256
```

Use `arkai --help` and the per-command `--help` output for command-line
overrides. The CLI is authoritative for supported options.
