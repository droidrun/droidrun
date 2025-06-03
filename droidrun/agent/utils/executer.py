import io
import contextlib
import ast
import traceback
import logging
from typing import Any, Dict
from droidrun.agent.utils.async_utils import async_to_sync
from llama_index.core.workflow import Context
import asyncio
from asyncio import AbstractEventLoop
import threading

logger = logging.getLogger("droidrun")

class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.

    NOTE: not safe for production use! Use with caution.
    """

    def __init__(self, loop: AbstractEventLoop, locals: Dict[str, Any] = {}, globals: Dict[str, Any] = {}, tools = {}, use_same_scope: bool = True):
        """
        Initialize the code executor.

        Args:
            locals: Local variables to use in the execution context
            globals: Global variables to use in the execution context
            tools: List of tools available for execution
        """

        # loop throught tools and add them to globals, but before that check if tool value is async, if so convert it to sync. tools is a dictionary of tool name: function
        # e.g. tools = {'tool_name': tool_function}
        
        # check if tools is a dictionary
        if isinstance(tools, dict):
            for tool_name, tool_function in tools.items():
                if asyncio.iscoroutinefunction(tool_function):
                    # If the function is async, convert it to sync
                    tool_function = async_to_sync(tool_function)
                # Add the tool to globals
                globals[tool_name] = tool_function
        elif isinstance(tools, list):
            # If tools is a list, convert it to a dictionary with tool name as key and function as value
            for tool in tools:
                if asyncio.iscoroutinefunction(tool):
                    # If the function is async, convert it to sync
                    tool = async_to_sync(tool)
                # Add the tool to globals
                globals[tool.__name__] = tool
        else:
            raise ValueError("Tools must be a dictionary or a list of functions.")


        import time
        globals['time'] = time
        
        self.globals = globals
        self.locals = locals
        self.loop = loop
        self.use_same_scope = use_same_scope
        if self.use_same_scope:
            # If using the same scope, set the globals and locals to the same dictionary
            self.globals = self.locals = {**self.locals, **{k: v for k, v in self.globals.items() if k not in self.locals}}

    async def execute(self, ctx: Context, code: str) -> str:
        """
        Execute Python code and capture output and return values.

        Args:
            code: Python code to execute

        Returns:
            str: Output from the execution, including print statements.
        """
        # Update UI elements before execution
        self.globals['ui_elements'] = await ctx.get("ui_state")
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        output = ""
        try:
            # Execute with captured output
            with contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                def execute_code():
                    exec(code, self.globals, self.locals)
                t = threading.Thread(target=execute_code)
                t.start()
                t.join()

            # Get output
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()

        except Exception as e:
            # Capture exception information
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        return output