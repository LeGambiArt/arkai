Feature: Third-party tool management
  Scenario: List tools supported for installation
    When I run the tools command "install"
    Then the exit code is 0
    And the output contains "Tools supported for installation: strata"
    And no external tool was executed

  Scenario: List when no Arkai-managed tools are installed
    Given no Arkai-managed tools are installed
    When I run the tools command "list"
    Then the exit code is 0
    And the output contains "No Arkai-managed tools installed"

  Scenario: List an installed tool and its version
    Given Arkai-managed Strata is installed
    When I run the tools command "list"
    Then the exit code is 0
    And the output contains "strata: Strata 0.2.3"

  Scenario: Cancel Strata installation
    Given no Arkai-managed tools are installed
    When I run the tools command with confirmation "no" for "install strata"
    Then the exit code is 0
    And the output contains "Strata install cancelled"
    And no external tool was executed

  Scenario: Install Strata using the official installer in Arapuca
    Given no Arkai-managed tools are installed
    When I run the tools command with confirmation "yes" for "install strata"
    Then the exit code is 0
    And the output contains "Strata install complete (Strata 0.2.3)"
    And the installer script bootstrap was requested
    And the external installer ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Update an installed Strata tool in Arapuca
    Given Arkai-managed Strata is installed
    When I run the tools command with confirmation "yes" for "update strata"
    Then the exit code is 0
    And the output contains "Strata update complete (Strata 0.2.3)"
    And the installer script bootstrap was requested
    And the external installer ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Remove an installed Strata tool in Arapuca
    Given Arkai-managed Strata is installed
    When I run the tools command with confirmation "yes" for "remove strata"
    Then the exit code is 0
    And the output contains "Removed Arkai-managed Strata"
    And tool removal ran inside Arapuca without network access

  Scenario: Reject unsupported tools
    When I run the tools command "install unsupported"
    Then the exit code is 1
    And the error contains "Supported tools: strata"
