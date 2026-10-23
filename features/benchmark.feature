Feature: LLM Model Benchmarking
  As a user
  I want to benchmark LLM models
  So that I can measure their performance characteristics

  Scenario: Display benchmark help
    When I run benchmark command "benchmark --help"
    Then the exit code is 0
    And the output contains "usage: arkai benchmark"
    And the output contains "--model"
    And the output contains "--prompts"
    And the output contains "--iterations"

  Scenario: Benchmark command is available
    When I run benchmark command "--help"
    Then the exit code is 0
    And the output contains "benchmark"
    And the output contains "Benchmark LLM models"

  Scenario: Default prompts are code-review
    When I run benchmark command "benchmark --help"
    Then the exit code is 0
    And the output contains "code-review:all"

  Scenario: Prompt format uses set:size notation
    When I run benchmark command "benchmark --help"
    Then the exit code is 0
    And the output contains "<set>:<size>"
    And the output contains "ai"
    And the output contains "code-review"

  Scenario: Show prompts from ai set
    When I run benchmark command "benchmark --show-prompts ai"
    Then the exit code is 0
    And the output contains "Prompts in set 'ai'"
    And the output contains "SHORT"
    And the output contains "MEDIUM"
    And the output contains "LONG"

  Scenario: Show prompts from code-review set
    When I run benchmark command "benchmark --show-prompts code-review"
    Then the exit code is 0
    And the output contains "Prompts in set 'code-review'"
    And the output contains "SHORT"
    And the output contains "MEDIUM"
    And the output contains "LONG"

  Scenario: Show prompts with invalid set fails
    When I run benchmark command "benchmark --show-prompts invalid"
    Then the exit code is 1
    And the error contains "Unknown prompt set"

  Scenario: Show prompts from coding set
    When I run benchmark command "benchmark --show-prompts coding"
    Then the exit code is 0
    And the output contains "Prompts in set 'coding'"
    And the output contains "SHORT"
    And the output contains "MEDIUM"
    And the output contains "LONG"
    And the output contains "Rust"

  Scenario: No-inference benchmark reuses a borrowed Strata profile
    Given a valid .arkai.yaml file
    When I run benchmark command "-IM --iterations 1 --no-warmup --prompts ai:short" with a reused Strata service
    Then the exit code is 0
    And the benchmark kept using the borrowed Strata service
