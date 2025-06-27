"""
Async utilities for the Twi Speech Training Engine

This module provides utilities for managing async operations and event loops
to avoid common issues like "Event loop is closed" errors.
"""

import asyncio
import functools
import sys
from typing import Any, Callable, Coroutine, TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncRunner:
    """A class to manage async operations with proper event loop handling"""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get the current event loop or create a new one if needed"""
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, try to get the current loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                return loop
            except RuntimeError:
                # Create a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                return loop

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run an async coroutine with proper event loop handling"""
        try:
            # First, try to use asyncio.run() if we're not in an async context
            loop = asyncio.get_running_loop()
            # If we get here, we're already in an async context
            raise RuntimeError("Already in async context, use await instead")
        except RuntimeError:
            # We're not in an async context, safe to proceed
            try:
                # Try using asyncio.run first (preferred method)
                return asyncio.run(coro)
            except RuntimeError as e:
                if "Event loop is closed" in str(e) or "Cannot run the event loop while another loop is running" in str(e):
                    # Fall back to manual loop management
                    loop = self.get_or_create_loop()
                    try:
                        return loop.run_until_complete(coro)
                    except RuntimeError as inner_e:
                        # If still failing, create a completely new loop
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(coro)
                        finally:
                            # Keep the loop open for future use
                            self._loop = new_loop
                else:
                    raise

    def __del__(self):
        """Clean up the event loop when the runner is destroyed"""
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception:
                pass


# Global instance for convenience
_async_runner = AsyncRunner()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine with proper event loop handling

    This function handles various edge cases:
    - Running from sync context
    - Event loop already closed
    - Multiple event loops

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    return _async_runner.run(coro)


def ensure_async(func: Callable) -> Callable:
    """
    Decorator that ensures a function can be called from both sync and async contexts

    If the decorated function is a coroutine, it will be properly handled
    whether called from sync or async code.
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            coro = func(*args, **kwargs)
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # We're in async context, return the coroutine
                return coro
            except RuntimeError:
                # We're in sync context, run the coroutine
                return run_async(coro)
        return wrapper
    else:
        # Not a coroutine, return as is
        return func


def create_task_safe(coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """
    Create an asyncio task safely, handling the case where no loop is running

    Args:
        coro: The coroutine to create a task from

    Returns:
        The created task
    """
    try:
        # Try to create task in current loop
        return asyncio.create_task(coro)
    except RuntimeError:
        # No running loop, create one and run the coroutine
        loop = _async_runner.get_or_create_loop()
        return loop.create_task(coro)


async def gather_safe(*coros, return_exceptions: bool = False):
    """
    Safely gather multiple coroutines, similar to asyncio.gather

    Args:
        *coros: Coroutines to gather
        return_exceptions: Whether to return exceptions as results

    Returns:
        List of results from the coroutines
    """
    try:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)
    except Exception as e:
        logger.error(f"Error in gather_safe: {e}")
        if return_exceptions:
            return [e for _ in coros]
        raise


def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function in an executor

    This is useful for running blocking I/O operations without blocking the event loop.

    Args:
        func: The function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function
    """
    async def _run():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    return run_async(_run())


class AsyncContextManager:
    """Base class for creating async context managers that work in both sync and async contexts"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self):
        # For sync context, run the async enter
        return run_async(self.__aenter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # For sync context, run the async exit
        run_async(self.__aexit__(exc_type, exc_val, exc_tb))


# Utility functions for common patterns

def sync_to_async(sync_func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a synchronous function to an async function

    This runs the sync function in an executor to avoid blocking.

    Args:
        sync_func: The synchronous function to convert

    Returns:
        An async version of the function
    """
    @functools.wraps(sync_func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(sync_func, *args, **kwargs))

    return async_wrapper


def async_to_sync(async_func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Convert an async function to a synchronous function

    This properly handles event loop creation and management.

    Args:
        async_func: The async function to convert

    Returns:
        A synchronous version of the function
    """
    @functools.wraps(async_func)
    def sync_wrapper(*args, **kwargs):
        return run_async(async_func(*args, **kwargs))

    return sync_wrapper


# Clean up on module exit
import atexit

def _cleanup():
    """Clean up any open event loops"""
    global _async_runner
    if _async_runner:
        del _async_runner

atexit.register(_cleanup)
