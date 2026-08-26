Feature: Inference Engine Management
  Scenario: Stop inference server when not running
    Given a valid .aitool.yaml file
    When I run "aitool inference stop"
    Then the exit code is 0
    And the output contains "not running"

  Scenario: Check engine status when stopped
    Given a valid .aitool.yaml file
    When I run "aitool inference status"
    Then the exit code is 0
    And the output contains "Inference:"
    And the output contains "stopped"

  Scenario: Start fails if port in use
    Given a valid .aitool.yaml file
    And port 8081 is in use
    When I run "aitool inference start"
    Then the exit code is 1
    And the error contains "Port 8081 already in use"

  Scenario: Start fails if model not found
    Given a valid .aitool.yaml file
    When I run "aitool inference start --model nonexistent.gguf"
    Then the exit code is 1
    And the error contains "Model not found"

  Scenario: Status shows running server
    Given a valid .aitool.yaml file
    And the inference server is running
    When I run "aitool inference status"
    Then the exit code is 0
    And the output contains "running"
