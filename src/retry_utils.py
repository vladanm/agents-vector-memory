"""Retry utilities for database operations.

Provides exponential backoff retry logic for handling transient failures,
particularly SQLite database locks.
"""

import time
import sqlite3
import logging
from typing import TypeVar, Callable, Any
from functools import wraps

from .exceptions import DatabaseLockError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    max_delay: float = 2.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 0.1)
        backoff_factor: Multiplier for each retry (default: 2.0)
        max_delay: Maximum delay between retries (default: 2.0)

    Returns:
        Decorator function

    Example:
        @exponential_backoff(max_retries=3)
        def my_db_operation():
            # ... database code ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Only retry on database locked errors
                    if "locked" not in error_msg and "busy" not in error_msg:
                        raise

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: "
                            f"Database locked. Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

            # If we get here, all retries failed
            raise DatabaseLockError(
                f"Database operation failed after {max_retries + 1} attempts: {last_exception}"
            )

        return wrapper
    return decorator


def retry_on_lock(
    func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 0.1
) -> T:
    """Direct retry function (non-decorator) for inline use.

    Args:
        func: Function to execute with retry logic
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Result from func

    Raises:
        DatabaseLockError: If all retries fail

    Example:
        result = retry_on_lock(lambda: conn.execute(sql))
    """
    delay = initial_delay
    backoff_factor = 2.0
    max_delay = 2.0
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except sqlite3.OperationalError as e:
            last_exception = e
            error_msg = str(e).lower()

            if "locked" not in error_msg and "busy" not in error_msg:
                raise

            if attempt < max_retries:
                logger.warning(
                    f"Database locked (attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"Database locked after {max_retries + 1} attempts")

    raise DatabaseLockError(
        f"Database operation failed after {max_retries + 1} attempts: {last_exception}"
    )
