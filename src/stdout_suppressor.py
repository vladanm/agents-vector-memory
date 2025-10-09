"""
Stdout Suppression Utilities
=============================

Critical for MCP stdio protocol: MUST prevent any library from writing to stdout.
Only JSON-RPC messages are allowed on stdout.
"""

import os
import sys
from contextlib import contextmanager
from typing import IO


@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to completely suppress stdout and stderr.

    CRITICAL for MCP stdio protocol:
    - sentence-transformers prints tokenizer config to stdout during model loading
    - This corrupts JSON-RPC communication
    - We MUST redirect ALL output to /dev/null during model initialization

    Usage:
        with suppress_stdout_stderr():
            model = SentenceTransformer('all-MiniLM-L6-v2')
    """
    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Save original sys.stdout and sys.stderr objects
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Flush before redirecting
    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate the original file descriptors
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    try:
        # Open /dev/null (or NUL on Windows)
        devnull = open(os.devnull, 'w')

        # Redirect stdout and stderr to /dev/null at the file descriptor level
        os.dup2(devnull.fileno(), original_stdout_fd)
        os.dup2(devnull.fileno(), original_stderr_fd)

        # Also redirect Python-level sys.stdout/stderr
        sys.stdout = devnull
        sys.stderr = devnull

        yield

    finally:
        # Flush devnull before closing
        if hasattr(devnull, 'flush'):
            devnull.flush()

        # Restore original file descriptors
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the saved file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

        # Restore Python-level sys.stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close devnull
        devnull.close()


@contextmanager
def suppress_stdout_only():
    """
    Context manager to suppress ONLY stdout (allows stderr for logging).

    Use this when you want to keep error logging but prevent stdout pollution.

    Usage:
        with suppress_stdout_only():
            model = SentenceTransformer('all-MiniLM-L6-v2')
    """
    # Save original file descriptor
    original_stdout_fd = sys.stdout.fileno()

    # Save original sys.stdout object
    original_stdout = sys.stdout

    # Flush before redirecting
    sys.stdout.flush()

    # Duplicate the original file descriptor
    saved_stdout_fd = os.dup(original_stdout_fd)

    try:
        # Open /dev/null (or NUL on Windows)
        devnull = open(os.devnull, 'w')

        # Redirect stdout to /dev/null at the file descriptor level
        os.dup2(devnull.fileno(), original_stdout_fd)

        # Also redirect Python-level sys.stdout
        sys.stdout = devnull

        yield

    finally:
        # Flush devnull before closing
        if hasattr(devnull, 'flush'):
            devnull.flush()

        # Restore original file descriptor
        os.dup2(saved_stdout_fd, original_stdout_fd)

        # Close the saved file descriptor
        os.close(saved_stdout_fd)

        # Restore Python-level sys.stdout
        sys.stdout = original_stdout

        # Close devnull
        devnull.close()
