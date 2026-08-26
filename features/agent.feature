Feature: Agent Execution
  Scenario: Agent requires TTY
    Given a valid .aitool.yaml file
    When I run "aitool agent" with stdin piped
    Then the exit code is 1
    And the error contains "agent requires a TTY"

  Scenario: Agent skips MCP with --no-mcp flag
    Given a valid .aitool.yaml file
    When I run "aitool agent" with "--no-mcp" in a TTY
    Then the exit code is 0
    And wtmcp was not started

  Scenario: Agent skips MCP when config disables it
    Given a valid .aitool.yaml file with mcp disabled
    When I run "aitool agent" in a TTY
    Then the exit code is 0
    And wtmcp was not started
