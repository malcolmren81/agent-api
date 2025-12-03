#!/bin/bash
# Phase 7.1.2 Backend Testing Suite
# Runs all tests: Unit, Integration, and Smoke tests

set -e

echo "========================================="
echo "Phase 7.1.2 Backend Testing Suite"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
UNIT_RESULT=0
INTEGRATION_RESULT=0
SMOKE_RESULT=0

# Change to script directory
cd "$(dirname "$0")"

# 1. Unit Tests
echo -e "${YELLOW}[1/3] Running Unit Tests...${NC}"
echo "-------------------------------------------"
if pytest tests/services/test_product_template_service.py -v --tb=short 2>&1; then
    echo -e "${GREEN}✓ Unit Tests PASSED${NC}"
    UNIT_RESULT=0
else
    echo -e "${RED}✗ Unit Tests FAILED${NC}"
    UNIT_RESULT=1
fi
echo ""

# 2. Integration Tests
echo -e "${YELLOW}[2/3] Running Integration Tests...${NC}"
echo "-------------------------------------------"
if pytest tests/integration/test_template_integration.py -v -m integration --tb=short 2>&1; then
    echo -e "${GREEN}✓ Integration Tests PASSED${NC}"
    INTEGRATION_RESULT=0
else
    echo -e "${RED}✗ Integration Tests FAILED${NC}"
    INTEGRATION_RESULT=1
fi
echo ""

# 3. Smoke Tests
echo -e "${YELLOW}[3/3] Running Smoke Tests (Live API)...${NC}"
echo "-------------------------------------------"
if [ -f "test_list_templates_live.py" ]; then
    if python3 test_list_templates_live.py 2>&1; then
        echo -e "${GREEN}✓ List Templates Test PASSED${NC}"
    else
        echo -e "${RED}✗ List Templates Test FAILED${NC}"
        SMOKE_RESULT=1
    fi

    echo ""

    if python3 test_get_template_live.py 2>&1; then
        echo -e "${GREEN}✓ Get Template Test PASSED${NC}"
    else
        echo -e "${RED}✗ Get Template Test FAILED${NC}"
        SMOKE_RESULT=1
    fi
else
    echo -e "${YELLOW}⚠ Smoke tests not found (OK if not deployed)${NC}"
    SMOKE_RESULT=0
fi
echo ""

# Summary
echo "========================================="
echo "Test Results Summary"
echo "========================================="
printf "Unit Tests:        "
if [ $UNIT_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

printf "Integration Tests: "
if [ $INTEGRATION_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

printf "Smoke Tests:       "
if [ $SMOKE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
fi

echo "========================================="

# Final result
TOTAL_RESULT=$((UNIT_RESULT + INTEGRATION_RESULT + SMOKE_RESULT))

if [ $TOTAL_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED - Ready for deployment${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED - Fix issues before deployment${NC}"
    exit 1
fi
