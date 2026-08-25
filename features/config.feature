Feature: Configuration Management
  Scenario: Validate valid configuration file
    Given a valid .aitool.yaml file
    When I run "aitool config validate"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Validate configuration with explicit file
    Given a valid config file at /tmp/test-config.yaml
    When I run "aitool config validate --file /tmp/test-config.yaml"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Validate invalid configuration fails
    Given an invalid .aitool.yaml file (missing required fields)
    When I run "aitool config validate"
    Then the exit code is 2
    And the error contains "agent.name is required"

  Scenario: Initialize new configuration
    Given no .aitool.yaml file exists
    When I run "aitool config init"
    Then the exit code is 0
    And the file .aitool.yaml exists with default values
    And the output contains "Configuration initialized"

  Scenario: Initialize fails if config already exists
    Given an existing .aitool.yaml file
    When I run "aitool config init"
    Then the exit code is 1
    And the error contains "Configuration file already exists"
