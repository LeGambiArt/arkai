Feature: Model Management
  Scenario: List available models
    Given a clean models directory
    When I run "arkai model list"
    Then the exit code is 0
    And the output contains "No models found"

  Scenario: Download a model from HuggingFace
    Given a clean models directory
    When I run "arkai model download ibm-granite/granite-4.1-8b-instruct-GGUF"
    Then the exit code is 0
    And a .gguf file exists in the models directory
    And the output contains "Downloaded"

  Scenario: Remove a model
    Given a model file "test-model.gguf" in the models directory
    When I run "arkai model remove test-model.gguf"
    Then the exit code is 0
    And the file "test-model.gguf" does not exist in the models directory

  Scenario: Remove non-existent model fails
    Given a clean models directory
    When I run "arkai model remove nonexistent.gguf"
    Then the exit code is 1
    And the error contains "Model not found"

  Scenario: List models shows details
    Given a model file "model1.gguf" in the models directory
    And a model file "model2.gguf" in the models directory
    When I run "arkai model list"
    Then the exit code is 0
    And the output contains "model1.gguf"
    And the output contains "model2.gguf"

  Scenario: List HuggingFace cached models with long names
    Given a clean models directory
    When I run "arkai model list" with HuggingFace models
    Then the exit code is 0
    And the output contains "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
    And the output contains "HuggingFace cached models"

  Scenario: Convert command requires model argument
    When I run "arkai model convert" without model argument
    Then the exit code is 2
    And the error contains "required"
