Feature: CLI argument handling
  The CLI should report invalid command structure with usage information.

  Scenario Outline: A command without a required subcommand shows usage
    When I invoke "arkai --debug <command>"
    Then the exit code is 2
    And the error contains "usage: arkai <command>"
    And the error does not contain "Traceback (most recent call last)"

    Examples:
      | command   |
      | agent     |
      | config    |
      | inference |
      | model     |
      | rag       |
      | sandbox   |
      | vectordb  |
      | wtmcp     |
