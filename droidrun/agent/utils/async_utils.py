import asyncio

def async_to_sync(func):
    """
    Convert an async function to a sync function.

    Args:
        func: Async function to convert

    Returns:
        Callable: Synchronous version of the async function
    """

    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper