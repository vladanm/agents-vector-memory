#!/bin/bash
# Simple test to check if MCP server outputs clean JSON without tokenizer pollution

echo "Starting MCP server and checking first 2000 bytes of output..."
echo ""

# Start server and capture first 2000 bytes
timeout 5 python3 main.py --database-path /tmp/test_simple.db 2>&1 | head -c 2000 > /tmp/mcp_output.txt

echo "Output captured:"
echo "----------------------------------------"
cat /tmp/mcp_output.txt
echo ""
echo "----------------------------------------"

# Check for tokenizer config pollution
if grep -q "do_lower_case" /tmp/mcp_output.txt || \
   grep -q "tokenizer_class" /tmp/mcp_output.txt || \
   grep -q "unk_token" /tmp/mcp_output.txt; then
    echo ""
    echo "❌ FAILED: Found tokenizer config pollution in stdout!"
    exit 1
else
    echo ""
    echo "✅ SUCCESS: No tokenizer config pollution detected!"

    # Check if output looks like JSON
    if head -1 /tmp/mcp_output.txt | grep -q '^{'; then
        echo "✅ Output starts with JSON (as expected for JSON-RPC)"
    fi

    exit 0
fi
