Feature: wtmcp Plugin Management
  Scenario: List plugins with no running instance fails
    When I run "arkai wtmcp list"
    Then the exit code is 1
    And the error contains "No running wtmcp instances"

  Scenario: Enable a plugin in project
    Given a valid .arkai.yaml file
    When I run "arkai wtmcp enable workspace"
    Then the exit code is 0
    And the project config has plugin "workspace"

  Scenario: Disable a plugin from project
    Given a valid .arkai.yaml file with plugins "workspace,terminal"
    When I run "arkai wtmcp disable workspace"
    Then the exit code is 0
    And the project config does not have plugin "workspace"

  Scenario: List shows plugins for running instance
    Given a valid .arkai.yaml file with plugins "workspace,terminal"
    And the wtmcp server is running
    When I run "arkai wtmcp list"
    Then the exit code is 0
    And the output contains "wtmcp Plugins"
    And the output contains "workspace"
