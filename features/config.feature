Feature: Configuration Management
  Scenario: Validate valid configuration file
    Given a valid .arkai.yaml file
    When I run "arkai config validate"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Validate configuration with explicit file
    Given a valid config file at /tmp/test-config.yaml
    When I run "arkai config validate --file /tmp/test-config.yaml"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Validate configuration with missing model shows warning
    Given an invalid .arkai.yaml file (missing required fields)
    When I run "arkai config validate"
    Then the exit code is 0
    And the output contains "Configuration valid"
    And the error contains "No model configured"

  Scenario: Initialize new configuration
    Given no .arkai.yaml file exists
    When I run "arkai config init"
    Then the exit code is 0
    And the file .arkai.yaml exists with default values
    And the output contains "Configuration initialized"

  Scenario: Initialize fails if config already exists
    Given a valid .arkai.yaml file
    When I run "arkai config init"
    Then the exit code is 1
    And the error contains "Configuration file already exists"
