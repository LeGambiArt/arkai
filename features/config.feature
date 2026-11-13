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

  Scenario: Validate configuration with explicit inference backend
    Given a valid .arkai.yaml file with inference backend "llama-cpp"
    When I run "arkai config validate"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Reject unsupported inference backend
    Given a valid .arkai.yaml file with inference backend "unsupported"
    When I run "arkai config validate"
    Then the exit code is 2
    And the error contains "inference.backend must be one of"

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

  Scenario Outline: Validate configuration with context size
    Given a valid .arkai.yaml file with inference context_size of <value>
    When I run "arkai config validate"
    Then the exit code is <code>
    And the <stream> contains "<message>"
    Examples:  Success
        | value   | code | stream | message |
        | 32768   | 0    | output | Configuration valid |
        | 32k     | 0    | output | Configuration valid |
        | 50K     | 0    | output | Configuration valid |
        | 1m      | 0    | output | Configuration valid |
        | 1M      | 0    | output | Configuration valid |
    Examples:  Failure
        | value   | code | stream | message |
        | -100    | 2    | error  | context_size must be positive integer |
        | invalid | 2    | error  | context_size must be positive integer |
        | 0       | 2    | error  | context_size must be positive integer |

  Scenario: Validate configuration with positive integer context size
    Given a valid .arkai.yaml file with inference context_size of 32768
    When I run "arkai config validate"
    Then the exit code is 0
    And the output contains "Configuration valid"

  Scenario: Validate configuration with negative context size fails
    Given a valid .arkai.yaml file with inference context_size of -1000
    When I run "arkai config validate"
    Then the exit code is not 0
    And the error contains "context_size must be positive integer"

  Scenario: Validate configuration with zero context size fails
    Given a valid .arkai.yaml file with inference context_size of 0
    When I run "arkai config validate"
    Then the exit code is not 0
    And the error contains "context_size must be positive integer"

  Scenario: Validate configuration with invalid context size string fails
    Given a valid .arkai.yaml file with inference context_size of "invalid"
    When I run "arkai config validate"
    Then the exit code is not 0
    And the error contains "context_size must be positive integer"

  Scenario: Edit project configuration with the TUI
    Given a valid .arkai.yaml file
    And the config editor sets "inference.port" to 9000 and saves
    When I run "arkai config edit"
    Then the exit code is 0
    And the output contains "Configuration saved"
    And the file .arkai.yaml has inference port 9000

  Scenario: Edit creates an inference profile and preserves other configuration
    Given a valid .arkai.yaml file
    And the config editor adds inference profile "fast" with model "test.gguf" and saves
    When I run "arkai config edit"
    Then the exit code is 0
    And the output contains "Configuration saved"

  Scenario: Edit creates a custom cloud provider and preserves other configuration
    Given a valid .arkai.yaml file
    And the config editor adds cloud custom provider "gateway" with base URL "https://gateway.example/v1" and saves
    When I run "arkai config edit"
    Then the exit code is 0
    And the output contains "Configuration saved"

  Scenario: Edit adds a custom key with a YAML value
    Given a valid .arkai.yaml file
    And the config editor sets custom key "plugin.flags" to "[x, y]" and saves
    When I run "arkai config edit"
    Then the exit code is 0
    And the output contains "Configuration saved"
    And the file .arkai.yaml has custom key "plugin.flags" with value "[x, y]"

  Scenario: Aborting the config editor leaves the file unchanged
    Given a valid .arkai.yaml file
    And the config editor aborts without saving
    When I run "arkai config edit"
    Then the exit code is 0
    And the file .arkai.yaml is unchanged

  Scenario: Editing the project configuration without a project file fails
    Given no .arkai.yaml file exists
    And the config editor sets "inference.port" to 9000 and saves
    When I run "arkai config edit --project"
    Then the exit code is 1
    And the error contains "No project configuration file"

  Scenario: Initialize project configuration interactively
    Given no .arkai.yaml file exists
    And the config editor sets "agent.name" to "crush" and saves
    When I run "arkai config init --tui"
    Then the exit code is 0
    And the output contains "Configuration initialized"
    And the file .arkai.yaml has agent name "crush"

  Scenario: Aborting interactive initialization writes no file
    Given no .arkai.yaml file exists
    And the config editor aborts without saving
    When I run "arkai config init --tui"
    Then the exit code is 0
    And the error contains "no configuration file written"
    And the file .arkai.yaml does not exist

  Scenario: Initialize global configuration interactively
    Given no .arkai.yaml file exists
    And the config editor sets "agent.name" to "crush" and saves
    When I run "arkai config init --tui --global"
    Then the exit code is 0
    And the file <config_home>/arkai.yaml has agent name "crush"
    And the file .arkai.yaml does not exist
