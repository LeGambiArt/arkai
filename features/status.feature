Feature: Server status monitoring
  Scenario: Show status when no servers are running
    When I run "arkai status"
    Then the exit code is 0
    And the output contains "arkai Server Status"
    And the output contains "Inference: stopped"
    And the output contains "wtmcp: stopped"
    And the output contains "Vectordb: stopped"

  Scenario: Show status with inference server running
    Given the inference server is running
    When I run "arkai status"
    Then the exit code is 0
    And the output contains "Inference: running"

  Scenario: Show status with vectordb running
    Given vectordb server is running
    When I run "arkai status"
    Then the exit code is 0
    And the output contains "Vectordb: running"
    And the output contains "Port: 8082"
