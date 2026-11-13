Feature: Agent Execution
  Scenario: Agent requires TTY
    Given a valid .arkai.yaml file
    When I run "arkai agent start" with stdin piped
    Then the exit code is 1
    And the error contains "agent start requires a TTY"

  Scenario: Agent skips MCP with --no-mcp flag
    Given a valid .arkai.yaml file
    When I run "arkai agent start" with "--no-mcp" in a TTY
    Then the exit code is 0
    And wtmcp was not started

  Scenario: Agent skips MCP when config disables it
    Given a valid .arkai.yaml file with mcp disabled
    When I run "arkai agent start" in a TTY
    Then the exit code is 0
    And wtmcp was not started

  Scenario: Agent skips MCP when wtmcp is disabled
    Given a valid .arkai.yaml file with wtmcp disabled
    When I run "arkai agent start" in a TTY
    Then the exit code is 0
    And wtmcp was not started

  Scenario: Agent skips sandbox with --no-sandbox flag
    Given a valid .arkai.yaml file
    When I run "arkai agent start" with "--no-sandbox" in a TTY
    Then the exit code is 0
    And the agent was not sandboxed

  Scenario: Agent skips sandbox when config disables it
    Given a valid .arkai.yaml file with sandbox disabled
    When I run "arkai agent start" in a TTY
    Then the exit code is 0
    And the agent was not sandboxed

  Scenario: Agent skips sandbox when sandbox.enabled is false
    Given a valid .arkai.yaml file with sandbox enabled set to false
    When I run "arkai agent start" in a TTY
    Then the exit code is 0
    And the agent was not sandboxed

  Scenario: Sandboxed agent commands satisfy Arapuca's network flag rules
    Given a valid .arkai.yaml file
    When I run "arkai agent start" in a TTY
    Then the exit code is 0
    And the sandbox command does not combine arapuca network flags

  Scenario: Sandboxed agent prompt satisfies Arapuca's network flag rules
    Given a valid .arkai.yaml file
    When I run "arkai agent prompt" with a sandboxed prompt
    Then the exit code is 0
    And the sandbox command does not combine arapuca network flags

  Scenario: Agent accepts -m/--model flag without inference.model in config
    Given a .arkai.yaml file with no model configured
    When I run "arkai agent start" with "-m test-model.gguf" in a TTY
    Then the exit code is 0

  Scenario: Agent uses the Strata model when its model list is empty
    Given a valid Strata inference config
    When I run "arkai agent start" against Strata with an empty model list in a TTY
    Then the exit code is 0
    And the agent config uses Strata model "q2_0"

  Scenario: Custom cloud models default to a 256k context window
    Given a custom cloud provider with a model that has no context window
    When I run "arkai agent start" with "--cloud custom" in a TTY
    Then the agent model catalog uses the default cloud context window

  Scenario: Agent start uses the configured cloud provider by default
    Given a custom cloud provider configured as the default
    When I run "arkai agent start" in a TTY
    Then the configured cloud provider is used

  Scenario: Agent prompt uses the configured cloud provider by default
    Given a custom cloud provider configured as the default
    When I run "arkai agent prompt" with a configured cloud
    Then the configured cloud provider is used for the prompt

  Scenario: Agent prompt requires a default for multiple cloud configurations
    Given multiple cloud configurations without a default
    When I run "arkai agent prompt" without a cloud default
    Then the exit code is 1
    And the error contains "agent.mode is 'cloud' but agent.cloud.default is not configured"

  Scenario: Agent start forces local inference with --local
    Given a custom cloud provider configured as the default
    When I run "arkai agent start" with "--local" in a TTY
    Then the agent uses local inference

  Scenario: Agent prompt forces local inference with --local
    Given a custom cloud provider configured as the default
    When I run "arkai agent prompt" with "--local"
    Then the agent prompt uses local inference

  Scenario: Custom cloud model size units are expanded in the model catalog
    Given a custom cloud provider with a model using size-unit metadata
    When I run "arkai agent start" with "--cloud custom" in a TTY
    Then the agent model catalog expands size units to token counts

  Scenario: Crush uses a custom cloud provider
    Given the Crush agent with a custom cloud provider
    When I run "arkai agent start" with "--agent crush --cloud custom" in a TTY
    Then the Crush configuration registers the custom provider

  Scenario: Custom cloud providers can use the OpenAI Responses protocol
    Given a custom cloud provider using the openai-responses protocol
    When I run "arkai agent start" with "--cloud custom" in a TTY
    Then the agent configuration maps the provider to the Responses client

  Scenario: A custom cloud model can override the provider protocol
    Given a custom cloud provider whose model overrides the provider protocol
    When I run "arkai agent start" with "--agent pi --cloud custom" in a TTY
    Then the agent configuration maps the model to its own protocol

  Scenario: OpenCode rejects a custom cloud model protocol that differs from its provider
    Given a custom cloud provider whose model overrides the provider protocol
    When I run "arkai agent start" with "--cloud custom" in a TTY
    Then the exit code is 1
    And the error contains "per-model protocols"

  Scenario: Agent lists Arkai sessions without starting agent infrastructure
    Given a saved Arkai agent session named "session-to-list"
    When I list Arkai agent sessions
    Then the exit code is 0
    And the output contains "session-to-list"
    And listing does not start agent infrastructure

  Scenario: Agent resumes a named Arkai session
    Given a saved Arkai agent session named "session-to-resume"
    When I resume the Arkai session "session-to-resume" in a TTY
    Then the exit code is 0
    And the named Arkai session was resumed

  Scenario: Pi installation can be cancelled
    When I cancel the pi agent installation
    Then the pi agent installation was not run

  Scenario: Pi prompt returns the complete answer when launched from a TTY
    Given a fake Pi agent and local inference server for prompt testing
    When I run a Pi prompt from a TTY with MCP disabled
    Then the Pi prompt exits successfully
    And the prompt output contains the complete final answer
    And the blocked project MCP warning is not displayed
