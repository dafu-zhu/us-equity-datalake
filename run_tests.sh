#!/bin/bash

# Test runner script for US Equity Data Lake
# Usage: ./run_tests.sh [options]
#
# This script uses uv for package management and test execution.
# Make sure uv is installed: curl -LsSf https://astral.sh/uv/install.sh | sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}US Equity Data Lake Test Suite${NC}"
echo "=================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Warning: uv not found. Install with:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo -e "${YELLOW}Falling back to system pytest...${NC}"
    USE_UV=false
else
    USE_UV=true
fi

# Parse command line arguments
TEST_TYPE="all"
VERBOSE=""
COVERAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--unit)
            TEST_TYPE="unit"
            shift
            ;;
        -i|--integration)
            TEST_TYPE="integration"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=src/quantdl --cov-report=html --cov-report=term"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  -u, --unit         Run unit tests only"
            echo "  -i, --integration  Run integration tests only"
            echo "  -v, --verbose      Verbose output"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                 # Run all tests"
            echo "  ./run_tests.sh -u              # Run unit tests only"
            echo "  ./run_tests.sh -u -v -c        # Run unit tests with coverage"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set PYTHONPATH (for fallback mode)
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Determine pytest command
if [ "$USE_UV" = true ]; then
    PYTEST_CMD="uv run pytest"
else
    PYTEST_CMD="pytest"
fi

# Run tests based on type
case $TEST_TYPE in
    unit)
        echo -e "${YELLOW}Running unit tests...${NC}"
        $PYTEST_CMD -m unit $VERBOSE $COVERAGE tests/unit/
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        $PYTEST_CMD -m integration $VERBOSE $COVERAGE tests/integration/
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        $PYTEST_CMD $VERBOSE $COVERAGE tests/
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"

    if [ -n "$COVERAGE" ]; then
        echo ""
        echo -e "${YELLOW}Coverage report generated: htmlcov/index.html${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
