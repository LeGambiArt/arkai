Feature: Vector database service management
  Scenario: Start vectordb server
    When I run "arkai vectordb start"
    Then the exit code is 0
    And vectordb server is running

  Scenario: Check vectordb status
    Given vectordb server is running
    When I run "arkai vectordb status"
    Then the exit code is 0
    And the output contains "Vectordb server status"

  Scenario: Start vectordb under a container engine
    When I run "arkai vectordb start"
    Then the exit code is 0
    And the output contains "Vectordb server started"
    And the vectordb container image is the default qdrant image

  Scenario: Initialize new database
    Given vectordb server is running
    When I run "arkai vectordb initdb test_db"
    Then the exit code is 0
    And the output contains "created successfully"

  Scenario: List databases
    Given vectordb server is running
    When I run "arkai vectordb list"
    Then the exit code is 0
    And the output contains "Available databases"

  Scenario: Drop database
    Given vectordb server is running
    When I run "arkai vectordb drop test_db"
    Then the exit code is 0
    And the output contains "dropped successfully"

  Scenario: Stop vectordb server
    Given vectordb server is running
    When I run "arkai vectordb stop"
    Then the exit code is 0
    And vectordb server is not running

  Scenario: Starting with a stopped container engine VM fails with a clear error
    When the container engine VM is not running
    And I run "arkai vectordb start"
    Then the exit code is 1
    And the error contains "podman machine start"
