Feature: Sandbox Profile Management
  As a user
  I want to create and manage sandbox profiles
  So that I can reuse configurations and switch between them easily

  Scenario: Create a sandbox profile with custom parameters
    When I run "arkai sandbox create minimal --cpus 1 --memory 1024"
    Then the exit code is 0
    And the output contains "Created sandbox profile 'minimal'"
    When I run "arkai sandbox list"
    Then the output contains "minimal"
    And the output contains "cpus=1"
    And the output contains "memory=1024"

  Scenario: Create profile from another profile
    Given I create a sandbox profile "base" with cpus=2 and memory=2048
    When I run "arkai sandbox create derived --from base --cpus 4"
    Then the exit code is 0
    And the output contains "Created sandbox profile 'derived'"
    When I run "arkai sandbox show derived"
    Then the output contains "cpus:      4"
    And the output contains "memory_mb: 2048"

  Scenario: Show default profile
    When I run "arkai sandbox show default"
    Then the exit code is 0
    And the output contains "Default Settings"

  Scenario: Show active profile when none is set
    When I run "arkai sandbox show active"
    Then the exit code is 0
    And the output contains "using defaults"

  Scenario: List profiles
    Given I create a sandbox profile "gpu" with cpus=5 and memory=8192
    Given I create a sandbox profile "minimal" with cpus=1 and memory=1024
    When I run "arkai sandbox list"
    Then the exit code is 0
    And the output contains "gpu"
    And the output contains "minimal"
    And the output contains "Available Profiles"

  Scenario: List with no profiles
    When I run "arkai sandbox list"
    Then the exit code is 0
    And the output contains "No custom profiles defined"

  Scenario: Set active profile
    Given I create a sandbox profile "gpu" with cpus=5 and memory=8192
    When I run "arkai sandbox active gpu"
    Then the exit code is 0
    And the output contains "Set active sandbox profile to 'gpu'"
    When I run "arkai sandbox show active"
    Then the output contains "Active Profile: gpu"
    And the output contains "cpus:      5"

  Scenario: Clear active profile with empty string
    Given I create a sandbox profile "gpu" with cpus=5 and memory=8192
    Given I run "arkai sandbox active gpu"
    When I run "arkai sandbox active"
    Then the exit code is 0
    And the output contains "Cleared active sandbox profile"
    When I run "arkai sandbox show active"
    Then the output contains "using defaults"

  Scenario: Clear active profile with 'none'
    Given I create a sandbox profile "gpu" with cpus=5 and memory=8192
    Given I run "arkai sandbox active gpu"
    When I run "arkai sandbox active none"
    Then the exit code is 0
    And the output contains "Cleared active sandbox profile"

  Scenario: Set default profile
    Given I create a sandbox profile "production" with cpus=8 and memory=16384
    When I run "arkai sandbox set-default production"
    Then the exit code is 0
    And the output contains "Set 'production' as default"
    When I run "arkai sandbox show default"
    Then the output contains "cpus:      8"
    And the output contains "memory_mb: 16384"

  Scenario: Delete profile
    Given I create a sandbox profile "temp" with cpus=1 and memory=1024
    When I run "arkai sandbox delete temp"
    Then the exit code is 0
    And the output contains "Deleted sandbox profile 'temp'"
    When I run "arkai sandbox list"
    Then the output does not contain "temp"

  Scenario: Delete active profile clears active_profile
    Given I create a sandbox profile "gpu" with cpus=5 and memory=8192
    Given I run "arkai sandbox active gpu"
    When I run "arkai sandbox delete gpu"
    Then the exit code is 0
    When I run "arkai sandbox show active"
    Then the output contains "using defaults"

  Scenario: Reject invalid profile name
    When I run "arkai sandbox create invalid-name --cpus 1"
    Then the exit code is 1
    And the error contains "Invalid profile name"

  Scenario: Reject reserved name 'default'
    When I run "arkai sandbox create default --cpus 2"
    Then the exit code is 1
    And the error contains "Invalid profile name"

  Scenario: Reject reserved name 'active'
    When I run "arkai sandbox create active --cpus 2"
    Then the exit code is 1
    And the error contains "Invalid profile name"

  Scenario: Reject duplicate profile name
    Given I create a sandbox profile "unique" with cpus=1 and memory=1024
    When I run "arkai sandbox create unique --cpus 2"
    Then the exit code is 1
    And the error contains "already exists"

  Scenario: Reject non-existent profile operations
    When I run "arkai sandbox delete nonexistent"
    Then the exit code is 1
    And the error contains "not found"
    When I run "arkai sandbox show nonexistent"
    Then the exit code is 1
    And the error contains "not found"
    When I run "arkai sandbox active nonexistent"
    Then the exit code is 1
    And the error contains "not found"

  Scenario: Create a profile with environment variables
    When I run "arkai sandbox create envprofile --cpus 2 -e FOO=bar -e COUNT=42"
    Then the exit code is 0
    And the output contains "Created sandbox profile 'envprofile'"
    When I run "arkai sandbox show envprofile"
    Then the exit code is 0
    And the output contains "environment:"
    And the output contains "FOO=bar"
    And the output contains "COUNT=42"

  Scenario: Create a profile with volume mounts
    When I run "arkai sandbox create myprofile --cpus 2 --volume /data:ro --volume /logs"
    Then the exit code is 0
    And the output contains "Created sandbox profile 'myprofile'"
    When I run "arkai sandbox show myprofile"
    Then the exit code is 0
    And the output contains "volume:"
    And the output contains "/data:ro"
    And the output contains "/logs"

  Scenario: Create a profile deduplicates exact duplicate volumes
    When I run "arkai sandbox create deduped --cpus 1 --volume /data --volume /data"
    Then the exit code is 0
    When I run "arkai sandbox show deduped"
    Then the exit code is 0
    And the output contains "/data"
