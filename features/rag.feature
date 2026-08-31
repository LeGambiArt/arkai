Feature: RAG document operations
  Scenario: Ingest text document
    Given vectordb server is running
    And a test database named "test_db"
    And a text document with sample content
    When I run "arkai rag ingest test_db test_document.txt"
    Then the exit code is 0
    And the output contains "chunks"

  Scenario: Query RAG database
    Given vectordb server is running
    And a test database named "test_db"
    And a document has been ingested
    When I run "arkai rag query test_db sample"
    Then the exit code is 0
    And the output contains "results"
