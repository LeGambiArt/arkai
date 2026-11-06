# Design: cloud providers

## Scope

Cloud provider support configures supported agents to use an external
model service instead of Arkai-managed local inference. Agent-specific
adapters translate provider settings into each agent's native format.
Authentication and custom-provider normalization are separate from the
orchestration flow.

## Provider paths

- Built-in selections include OpenAI and Vertex AI, subject to each agent
  adapter's capabilities.
- Custom providers describe HTTPS endpoints using either the OpenAI Chat
  Completions or Anthropic Messages protocol.
- A custom provider declares model metadata and an API-key source.
  Secrets are referenced from an environment variable or supported system
  keychain; they are not stored as values in Arkai YAML.
- Provider configuration can inherit endpoint, protocol, and key-source
  defaults, with per-provider overrides. A default provider is required
  when more than one provider is configured.

## Security and boundaries

Custom endpoints are validated as HTTPS URLs without embedded
credentials, query strings, or fragments. The resolved endpoint host is
supplied to sandbox-aware adapters for network policy. API keys are
resolved on the host; they are not exposed to the sandbox as a general
host keychain.

Provider launch configuration is staged using the agent's native
configuration mechanisms. Adapters may use private temporary files for
credentials and must remove temporary material after launch. Built-in
providers can use their own native login stores.

## Implementation map

- Custom provider validation and normalization: `arkai/cloud_provider.py`
- Credential resolution: `arkai/cloud_auth.py`
- Provider wiring: `arkai/agent.py` and `arkai/agent_*.py`
- Unit coverage: `tests/unit/test_cloud_provider.py`,
  `tests/unit/test_cloud_auth.py`
- User instructions: [Cloud providers](../guides/cloud-providers.md)
