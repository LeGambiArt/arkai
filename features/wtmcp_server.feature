Feature: wtmcp Server Lifecycle
  Scenario: Start wtmcp server succeeds
    Given a valid .aitool.yaml file with plugins "workspace"
    When I run "aitool wtmcp start"
    Then the exit code is 0
    And the output contains "started"

  Scenario: Stop wtmcp server when running
    Given a valid .aitool.yaml file with plugins "workspace"
    And the wtmcp server is running
    When I run "aitool wtmcp stop"
    Then the exit code is 0
    And the output contains "stopped"

  Scenario: Stop wtmcp server when not running
    Given a valid .aitool.yaml file
    When I run "aitool wtmcp stop"
    Then the exit code is 0
    And the output contains "not running"

  Scenario: Status of running server
    Given a valid .aitool.yaml file with plugins "workspace"
    And the wtmcp server is running
    When I run "aitool wtmcp status"
    Then the exit code is 0
    And the output contains "running"

  Scenario: Status of stopped server
    Given a valid .aitool.yaml file
    When I run "aitool wtmcp status"
    Then the exit code is 0
    And the output contains "no instances running"

  Scenario: Start with enable plugins override
    Given a valid .aitool.yaml file with plugins "workspace"
    When I run "aitool wtmcp start --enable terminal --enable github"
    Then the exit code is 0
    And the output contains "Effective plugins"
    And the output contains "github"
    And the output contains "terminal"
    And the output contains "workspace"

  Scenario: Start with disable plugins override
    Given a valid .aitool.yaml file with plugins "workspace,terminal,github"
    When I run "aitool wtmcp start --disable github"
    Then the exit code is 0
    And the output contains "Effective plugins"
    And the output contains "workspace"
    And the output contains "terminal"

  Scenario: Start with both enable and disable
    Given a valid .aitool.yaml file with plugins "workspace,terminal"
    When I run "aitool wtmcp start --enable github --disable terminal"
    Then the exit code is 0
    And the output contains "Effective plugins"
    And the output contains "workspace"
    And the output contains "github"
