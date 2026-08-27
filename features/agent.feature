Feature: Agent Execution
  Scenario: Agent requires TTY
    Given a valid .arkai.yaml file
    When I run "arkai agent start" with stdin piped
    Then the exit code is 1
    And the error contains "agent requires a TTY"

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

  Scenario: Agent accepts -m/--model flag without inference.model or inference.hf in config
    Given a .arkai.yaml file with no model configured
    When I run "arkai agent start" with "-m test-model.gguf" in a TTY
    Then the exit code is 0

  Scenario: Agent start with HuggingFace model, agent, --no-sandbox and --no-mcp flags
    Given a .arkai.yaml file with no model configured
    When I run "arkai agent start -m ibm-granite/granite-4.1-8b-GGUF -a opencode --no-sandbox --no-mcp" in a TTY
    Then the exit code is 0
    And wtmcp was not started
    And the agent was not sandboxed
