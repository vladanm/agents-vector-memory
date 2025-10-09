#!/usr/bin/env python3
import json
import subprocess
import sys

# Start the MCP server
cmd = [
    "/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/venv/bin/python",
    "/Users/vladanm/projects/vector-memory-mcp/vector-memory-2-mcp/main.py",
    "--database-path",
    "/Users/vladanm/projects/subagents/simple-agents/memory/memory/agent_session_memory.db"
]

proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0
)

# Send initialize request
init_req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }
}

print(f"Sending: {json.dumps(init_req)}", file=sys.stderr)
proc.stdin.write(json.dumps(init_req) + "\n")
proc.stdin.flush()

# Read response
response = proc.stdout.readline()
print(f"Got init response: {response}", file=sys.stderr)

# Send initialized notification
init_notif = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
}
print(f"Sending: {json.dumps(init_notif)}", file=sys.stderr)
proc.stdin.write(json.dumps(init_notif) + "\n")
proc.stdin.flush()

# Send list_tools request
list_tools_req = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
}

print(f"Sending: {json.dumps(list_tools_req)}", file=sys.stderr)
proc.stdin.write(json.dumps(list_tools_req) + "\n")
proc.stdin.flush()

# Read response with timeout
import select
import time
timeout = 5
start = time.time()
while time.time() - start < timeout:
    ready = select.select([proc.stdout], [], [], 0.1)
    if ready[0]:
        response = proc.stdout.readline()
        if response:
            print(f"Got list_tools response: {response}", file=sys.stderr)
            try:
                data = json.loads(response)
                if "result" in data and "tools" in data["result"]:
                    print(f"SUCCESS: Got {len(data['result']['tools'])} tools", file=sys.stderr)
                break
            except:
                pass
else:
    print(f"TIMEOUT after {timeout}s", file=sys.stderr)

proc.terminate()
proc.wait()
