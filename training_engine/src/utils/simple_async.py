"""
Simple async runner for the Twi Speech Training Engine

This module provides a simple way to run async functions without
dealing with event loop management issues.
"""

import asyncio
import functools
from typing import TypeVar, Coroutine, Any

T = TypeVar('T')


def run_async_fresh(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine in a fresh event loop

    This function always creates a new event loop, runs the coroutine,
    and then closes the loop. This avoids "Event loop is closed" errors
    when mixing sync and async code.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    # Save the current event loop (if any)
    try:
        old_loop = asyncio.get_event_loop()
    except RuntimeError:
        old_loop = None

    # Create a fresh event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the coroutine
        result = loop.run_until_complete(coro)
        return result
    finally:
        # Always close the loop
        loop.close()

        # Restore the old event loop (if any)
        if old_loop is not None:
            asyncio.set_event_loop(old_loop)
        else:
            # Clear the event loop
            asyncio.set_event_loop(None)


class IsolatedAsyncRunner:
    """
    A runner that ensures each async operation runs in complete isolation
    """

    @staticmethod
    def run(coro: Coroutine[Any, Any, T]) -> T:
        """
        Run a coroutine in an isolated event loop

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        return run_async_fresh(coro)

    @staticmethod
    def run_multiple(*coros):
        """
        Run multiple coroutines in the same isolated event loop

        Args:
            *coros: Variable number of coroutines to run

        Returns:
            List of results from the coroutines
        """
        async def gather_all():
            return await asyncio.gather(*coros)

        return run_async_fresh(gather_all())


def async_to_sync_isolated(async_func):
    """
    Decorator to convert an async function to sync using isolated event loops

    Args:
        async_func: The async function to wrap

    Returns:
        A synchronous version of the function
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        coro = async_func(*args, **kwargs)
        return run_async_fresh(coro)

    return wrapper
