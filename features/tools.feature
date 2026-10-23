Feature: Third-party tool management
  Scenario: List tools supported for installation
    When I run the tools command "install"
    Then the exit code is 0
    And the output contains "Tools supported for installation: strata, pi, omp"
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
    And the output contains "Arkai will use this executable automatically"
    And the installer script bootstrap was requested
    And all external tool commands ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Use the Arkai-managed Strata executable for inference
    Given a valid Strata inference config
    And Arkai-managed Strata is installed for inference
    When I run "arkai inference start"
    Then the exit code is 0
    And the Strata launch used Arkai's managed executable
    And Arkai records the Strata profile as owned

  Scenario: Install Pi and its extensions inside Arapuca
    Given no Arkai-managed tools are installed
    When I run the tools command with confirmation "yes" for "install pi"
    Then the exit code is 0
    And the output contains "Pi install complete (1.2.3)"
    And all external tool commands ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Update Pi and its extensions inside Arapuca
    Given Arkai-managed Pi is installed
    When I run the tools command with confirmation "yes" for "update pi"
    Then the exit code is 0
    And the output contains "Pi update complete (1.2.3)"
    And Pi's supported extensions were updated
    And all external tool commands ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Remove Pi without deleting unrelated installations
    Given Arkai-managed Pi is installed
    When I run the tools command with confirmation "yes" for "remove pi"
    Then the exit code is 0
    And the output contains "Removed Arkai-managed Pi"
    And tool removal ran inside Arapuca without network access

  Scenario: Install OMP using its official installer inside Arapuca
    Given no Arkai-managed tools are installed
    When I run the tools command with confirmation "yes" for "install omp"
    Then the exit code is 0
    And the output contains "OMP install complete (omp 2.0)"
    And all external tool commands ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Update OMP using its official installer inside Arapuca
    Given Arkai-managed OMP is installed
    When I run the tools command with confirmation "yes" for "update omp"
    Then the exit code is 0
    And the output contains "OMP update complete (omp 2.0)"
    And all external tool commands ran inside Arapuca
    And the sandbox allowed network access

  Scenario: Remove OMP from its Arkai-managed location
    Given Arkai-managed OMP is installed
    When I run the tools command with confirmation "yes" for "remove omp"
    Then the exit code is 0
    And the output contains "Removed Arkai-managed OMP"
    And tool removal ran inside Arapuca without network access

  Scenario: Update an installed Strata tool in Arapuca
    Given Arkai-managed Strata is installed
    When I run the tools command with confirmation "yes" for "update strata"
    Then the exit code is 0
    And the output contains "Strata update complete (Strata 0.2.3)"
    And the installer script bootstrap was requested
    And all external tool commands ran inside Arapuca
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
