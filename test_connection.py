#!/usr/bin/env python3
"""
Test MCP server connection to verify stdout suppression fix.
"""

import subprocess
import json
import time
import sys

def test_mcp_connection():
    """Test that MCP server starts and responds to initialize request."""

    print("Starting MCP server...")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--database-path", "/tmp/test_mcp_connection.db"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait a moment for server to start
    time.sleep(1)

    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    print("Sending initialize request...")
    request_str = json.dumps(init_request) + "\n"
    proc.stdin.write(request_str)
    proc.stdin.flush()

    # Read response (with timeout)
    print("Waiting for response...")
    try:
        response_line = proc.stdout.readline()
        print(f"Raw response: {response_line[:200]}...")

        # Try to parse as JSON
        try:
            response = json.loads(response_line)
            print("\n✓ SUCCESS: Server responded with valid JSON-RPC")
            print(f"  Response ID: {response.get('id')}")
            print(f"  Server name: {response.get('result', {}).get('serverInfo', {}).get('name')}")
            print(f"  Protocol version: {response.get('result', {}).get('protocolVersion')}")

            # Check for tokenizer config pollution
            if "do_lower_case" in response_line or "tokenizer_class" in response_line:
                print("\n✗ FAILED: Response contains tokenizer config pollution!")
                return False

            return True

        except json.JSONDecodeError as e:
            print(f"\n✗ FAILED: Response is not valid JSON: {e}")
            print(f"  First 500 chars: {response_line[:500]}")
            return False

    except Exception as e:
        print(f"\n✗ FAILED: Error reading response: {e}")
        return False

    finally:
        proc.terminate()
        proc.wait(timeout=2)

if __name__ == "__main__":
    success = test_mcp_connection()
    sys.exit(0 if success else 1)
