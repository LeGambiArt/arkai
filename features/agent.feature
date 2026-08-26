Feature: Agent Execution
  Scenario: Agent requires TTY
    Given a valid .aitool.yaml file
    When I run "aitool agent" with stdin piped
    Then the exit code is 1
    And the error contains "agent requires a TTY"
