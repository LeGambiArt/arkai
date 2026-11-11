# Cloud providers

Agent launches can use a cloud service instead of local inference:

```bash
arkai agent start --agent opencode --cloud openai
arkai agent prompt --agent codex --cloud openai "Summarize this project"
arkai agent start --agent pi --cloud vertex
```

To make both `agent start` and `agent prompt` use cloud inference by default,
set `agent.mode` to `cloud` and choose a service with `agent.cloud.default`.
The default mode is `local`; passing `--cloud` selects cloud inference for
that invocation and overrides `agent.cloud.default`.
Passing `--local` forces local inference for that invocation and
overrides `agent.mode`, even when it is set to `cloud` or a session
was previously launched with a cloud provider.
When exactly one cloud configuration is present, Arkai uses it automatically.
If several are configured, set `agent.cloud.default` to choose one.

The built-in choices are `openai`, `vertex`, and `custom`. Cloud launches
do not start the local inference backend. Each agent adapter translates
the provider selection into that agent's native configuration and
authentication format; provider support can vary by agent. Codex does not
currently support `--cloud vertex`, and Claude does not support cloud
launches. Crush, OpenCode, Pi, and OMP support all three cloud services.

## OpenAI

Configure credentials in the agent's native setup or export
`OPENAI_API_KEY`. Some adapters can prompt for a key and save it in the
agent's normal credential store. Codex uses its existing Codex login or
API-key setup.

## Vertex AI

Authenticate with Google Application Default Credentials, for example:

```bash
gcloud auth application-default login
```

Alternatively set `GOOGLE_APPLICATION_CREDENTIALS`, and configure the
project and location using the variables required by the selected agent.
Sandboxed launches mount an available ADC credential file read-only and
allow the Vertex and OAuth endpoints needed by the provider.

## Custom endpoints

Custom providers use an HTTPS endpoint and either the OpenAI Chat
Completions or Anthropic Messages protocol. API keys are referenced from
an environment variable or system keychain; do not put secret values in
Arkai YAML.

```yaml
agent:
  mode: cloud
  cloud:
    default: custom
    custom:
      default_provider: gateway
      protocol: openai-completions
      base_url: https://gateway.example/v1
      api_key:
        source: environment
        name: TEAM_MODEL_KEY
      providers:
        gateway:
          name: Team gateway
          default_model: org/model-id
          models:
            - id: org/model-id
              name: Example model
              context_window: 65536
              max_output_tokens: 8192
```

Models without an explicit `context_window` default to 262144 tokens
(256k) instead of the agent's smaller built-in default. Both
`context_window` and `max_output_tokens` accept a token count or a size
unit string with a `k` (`*1024`) or `m` (`*1024*1024`) suffix, so
`context_window: 64k` and `max_output_tokens: 8K` mean 65536 and 8192
tokens.

Export the variable before launching, then select the provider with
`--cloud custom`. If several providers are configured, `default_provider`
is required; `--model` can select a model or `provider-id/model-id`. For
keychain storage, install the optional extra:

```bash
python -m pip install 'arkai[keychain]'
keyring set arkai/cloud gateway
```

Configure `api_key` with `source: keychain`, `service`, and `account` as
shown below; the values must match the `keyring set` arguments.

For example, a keychain source can be declared as:

```yaml
api_key:
  source: keychain
  service: arkai/cloud
  account: gateway
```

If several custom providers use keychain credentials, store them in one
system-keychain item to minimize unlock and access prompts. Run:

```bash
keyring set arkai/credential-cache v1
```

At the hidden password prompt, paste a one-line JSON bundle, replacing the
example keys with the matching service/account pairs and API keys:

```json
{"version":1,"credentials":[{"service":"arkai/cloud","account":"gateway","key":"..."},{"service":"arkai/cloud","account":"private","key":"..."}]}
```

The bundle is stored by the OS keychain (macOS Keychain or Linux Secret
Service). Arkai reads it once per launch and keeps resolved keys in memory
only for that launch. Existing per-provider keyring items remain supported
as a fallback, but each distinct item may trigger its own OS prompt.

See the [cloud provider design](../design/cloud-providers.md) for
credential handling, sandbox network rules, and adapter behavior.
