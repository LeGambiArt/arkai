Feature: Inference Engine Management
  Scenario: Stop inference server when not running
    Given a valid .arkai.yaml file
    When I run "arkai inference stop"
    Then the exit code is 0
    And the output contains "not running"

  Scenario: Check engine status when stopped
    Given a valid .arkai.yaml file
    When I run "arkai inference status"
    Then the exit code is 0
    And the output contains "Inference: stopped"

  Scenario: Start fails if port in use
    Given a valid .arkai.yaml file
    And port 8081 is in use
    When I run "arkai inference start"
    Then the exit code is 1
    And the error contains "Port 8081 already in use"

  Scenario: Start fails if custom port in use
    Given a valid .arkai.yaml file
    And port 9090 is in use
    When I run "arkai inference start --port 9090"
    Then the exit code is 1
    And the error contains "Port 9090 already in use"

  Scenario: Start fails if model not found
    Given a valid .arkai.yaml file
    When I run "arkai inference start --model nonexistent.gguf"
    Then the exit code is 1
    And the error contains "Model not found"

  Scenario: Status shows running server
    Given a valid .arkai.yaml file
    And the inference server is running
    When I run "arkai inference status"
    Then the exit code is 0
    And the output contains "Inference: running"

  Scenario: Stop running inference server removes PID file on clean exit
    Given a valid .arkai.yaml file
    And the inference server is running
    When I run "arkai inference stop"
    Then the exit code is 0
    And the inference PID file does not exist

  Scenario: MLX backend is accepted by configuration validation
    Given a valid .arkai.yaml file with inference backend "mlx"
    When I run "arkai config validate"
    Then the exit code is 0

  Scenario: MLX inference fails with an actionable missing dependency error
    Given a valid .arkai.yaml file with inference backend "mlx"
    And MLX-LM is not importable
    When I run "arkai inference start"
    Then the exit code is 1
    And the error contains "MLX-LM is not installed in the active Python environment"
    And the error contains "pip install -e '.[mlx]'"
