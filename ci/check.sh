#!/bin/bash

# CI Check Script - Runs formatting, linting, type checking, and tests
# Usage:
#   ./ci/check.sh              # Run all checks
#   ./ci/check.sh format       # Run formatting check
#   ./ci/check.sh lint         # Run linter check
#   ./ci/check.sh type         # Run type checker
#   ./ci/check.sh unit         # Run unit tests
#   ./ci/check.sh bdd          # Run BDD tests
#   ./ci/check.sh coverage     # Generate coverage report

set -e

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Functions for each check
check_format() {
    if [[ $# -gt 0 ]]
    then
        [[ "$1" = "--fix" ]] && shift
        if [[ $# -gt 0 ]]
        then
            echo "Running ruff format with given arguments..."
            ruff format "${@:-}" "${PROJECT_ROOT}"
        else
            echo "Fixing code formatting with ruff..."
            ruff format "${PROJECT_ROOT}"
        fi
    else
        echo "Checking code formatting with ruff..."
        ruff format --check "${PROJECT_ROOT}"
    fi
}

check_lint() {
    echo "Running linter (ruff check)..."
    if [[ $# -gt 0 ]]
    then
        ruff check "${@:-}" "${PROJECT_ROOT}"
    else
        ruff check "${PROJECT_ROOT}"
    fi
}

check_type() {
    echo "Running type checker (ty check)..."
    ty check "${PROJECT_ROOT}"
}

check_unit() {
    echo "Running unit tests with pytest..."
    coverage run --data-file="${PROJECT_ROOT}/.coverage.pytest" -m pytest "${PROJECT_ROOT}"/tests/unit/ -v
}

check_bdd() {
    echo "Running BDD tests with behave..."
    if [[ $# -gt 0 ]]
    then
        coverage run \
            --data-file="${PROJECT_ROOT}/.coverage.behave" \
            -m behave "${PROJECT_ROOT}"/features "$@"
    else
        coverage run \
            --data-file="${PROJECT_ROOT}/.coverage.behave" \
            -m behave "${PROJECT_ROOT}"/features
    fi
}

check_coverage() {
    python -c 'from coverage import __version__ as version; from packaging.version import Version; import sys; sys.exit(0 if Version(version) >= Version("7.14.0") else 1)' && keep="--keep"
    echo "keep: ${keep}"
    [[ "${1:-}" = "-f" ]] && { check_unit; check_bdd; }
    [[ -f "${PROJECT_ROOT}/.coverage.pytest" ]] || check_unit
    [[ -f "${PROJECT_ROOT}/.coverage.behave" ]] || check_bdd
    echo "Generating coverage report for latest test runs..."
    coverage combine ${keep:-} -q "${PROJECT_ROOT}/.coverage.pytest" "${PROJECT_ROOT}/.coverage.behave"
    coverage report ${keep:-} --show-missing --skip-empty --skip-covered
}

run_all() {
    echo "=========================================="
    echo "CI CHECK - arkai"
    echo "=========================================="
    echo ""

    local format_ok=0 lint_ok=0 type_ok=0 unit_ok=0 bdd_ok=0 coverage_ok=0
    local format_msg="" lint_msg="" type_msg="" unit_msg="" bdd_msg="" coverage_msg=""

    echo "1/6: Checking code formatting..."
    if check_format > /tmp/ci_format.log 2>&1; then
        echo "✓ Format check passed"
        format_ok=1
    else
        echo "✗ Format check failed"
        format_msg=$(cat /tmp/ci_format.log)
    fi
    echo ""

    echo "2/6: Running linter..."
    if check_lint > /tmp/ci_lint.log 2>&1; then
        echo "✓ Linting passed"
        lint_ok=1
    else
        echo "✗ Linting failed"
        lint_msg=$(cat /tmp/ci_lint.log)
    fi
    echo ""

    echo "3/6: Running type checker..."
    if check_type > /tmp/ci_type.log 2>&1; then
        echo "✓ Type checking passed"
        type_ok=1
    else
        echo "✗ Type checking failed"
        type_msg=$(cat /tmp/ci_type.log)
    fi
    echo ""

    echo "4/6: Running unit tests..."
    if check_unit > /tmp/ci_unit.log 2>&1; then
        echo "✓ Unit tests passed"
        unit_ok=1
    else
        echo "✗ Unit tests failed"
        unit_msg=$(cat /tmp/ci_unit.log)
    fi
    echo ""

    echo "5/6: Running BDD tests..."
    if check_bdd > /tmp/ci_bdd.log 2>&1; then
        echo "✓ BDD tests passed"
        bdd_ok=1
    else
        echo "✗ BDD tests failed"
        bdd_msg=$(cat /tmp/ci_bdd.log)
    fi
    echo ""

    echo "6/6: Generating coverage report..."
    if check_coverage > /tmp/ci_coverage.log 2>&1; then
        echo "✓ Coverage report generated"
        coverage_ok=1
        coverage_msg=$(tail -20 /tmp/ci_coverage.log | grep -E "TOTAL|%" | tail -1)
    else
        echo "✗ Coverage report generation failed"
        coverage_msg=$(cat /tmp/ci_coverage.log)
    fi
    echo ""

    echo "=========================================="
    echo "SUMMARY"
    echo "=========================================="
    echo "Format check:      $([ $format_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    echo "Linting:           $([ $lint_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    echo "Type checking:     $([ $type_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    echo "Unit tests:        $([ $unit_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    echo "BDD tests:         $([ $bdd_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    echo "Coverage report:   $([ $coverage_ok -eq 1 ] && echo '✓ PASSED' || echo '✗ FAILED')"
    [ -n "$coverage_msg" ] && echo "  $coverage_msg"
    echo ""

    if [ $format_ok -eq 0 ] || [ $lint_ok -eq 0 ] || [ $type_ok -eq 0 ] || \
       [ $unit_ok -eq 0 ] || [ $bdd_ok -eq 0 ] || [ $coverage_ok -eq 0 ]; then
        echo "=========================================="
        echo "FAILURES"
        echo "=========================================="
        [ -n "$format_msg" ] && echo "Format: $format_msg" && echo ""
        [ -n "$lint_msg" ] && echo "Linting: $lint_msg" && echo ""
        [ -n "$type_msg" ] && echo "Type checking: $type_msg" && echo ""
        [ -n "$unit_msg" ] && echo "Unit tests: $unit_msg" && echo ""
        [ -n "$bdd_msg" ] && echo "BDD tests: $bdd_msg" && echo ""
        [ -n "$coverage_msg" ] && [ $coverage_ok -eq 0 ] && echo "Coverage: $coverage_msg" && echo ""
        echo "=========================================="
        return 1
    else
        echo "✓ ALL CHECKS PASSED"
        echo "=========================================="
        return 0
    fi
}

# Main script logic
COMMAND="${1:-all}"

[ $# -gt 0 ] && shift

case "$COMMAND" in
    all)
        run_all
        ;;
    format)
        check_format "$@"
        ;;
    lint)
        check_lint "$@"
        ;;
    type)
        check_type
        ;;
    unit)
        check_unit
        ;;
    bdd)
        check_bdd
        ;;
    coverage)
        [[ "${2:-}" == "-f" ]] && shift && force="-f"
        check_coverage ${force:-}
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all       - Run all checks (default)"
        echo "  format    - Check code formatting"
        echo "  lint      - Run linter"
        echo "  type      - Run type checker"
        echo "  unit      - Run unit tests"
        echo "  bdd       - Run BDD tests"
        echo "  coverage  - Generate coverage report"
        exit 1
        ;;
esac
