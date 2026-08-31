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
