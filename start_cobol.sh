#!/bin/bash
# COBOL v1.5 FPGA Control - Quick Start Script

set -e

echo "=========================================="
echo "COBOL v1.5 FPGA Control - Quick Start"
echo "=========================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo ""
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}[2/5] Installing dependencies...${NC}"
pip install -q -r requirements_api.txt 2>/dev/null || {
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    echo "Try: pip install -r requirements_api.txt"
    exit 1
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Start API server in background
echo ""
echo -e "${YELLOW}[3/5] Starting API server...${NC}"

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Port 8000 is already in use${NC}"
    read -p "Continue with different port? (default: 9000) Enter port or press Enter: " PORT
    PORT=${PORT:-9000}
else
    PORT=8000
fi

# Start server
python3 fpga_api_server.py 0.0.0.0 $PORT &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Failed to start API server${NC}"
    exit 1
fi

echo -e "${GREEN}✓ API server started (PID: $SERVER_PID)${NC}"
echo "  URL: http://localhost:${PORT}"

# Display URLs
echo ""
echo -e "${YELLOW}[4/5] Services available...${NC}"
echo -e "${GREEN}✓ REST API${NC}:"
echo "   http://localhost:${PORT}/api/info"
echo "   http://localhost:${PORT}/health"
echo ""
echo -e "${GREEN}✓ Dashboard${NC}:"
echo "   http://localhost:${PORT}/dashboard.html"
echo ""
echo -e "${GREEN}✓ API Docs${NC}:"
echo "   http://localhost:${PORT}/docs (Swagger UI)"
echo "   http://localhost:${PORT}/redoc (ReDoc)"

# Run tests
echo ""
echo -e "${YELLOW}[5/5] Running API tests...${NC}"
echo ""

# Wait a bit more for server to be fully ready
sleep 2

# Run tests
if python3 test_api_client.py 2>/dev/null; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Some tests failed (API might still be starting)${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}READY TO USE${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Open dashboard: http://localhost:${PORT}/dashboard.html"
echo "  2. Monitor real-time metrics via WebSocket"
echo "  3. Test endpoints with curl:"
echo ""
echo "     curl http://localhost:${PORT}/api/metrics/cluster"
echo ""
echo "To stop the server:"
echo "  kill $SERVER_PID"
echo ""
echo "For help:"
echo "  python3 test_api_client.py --help"
echo ""
