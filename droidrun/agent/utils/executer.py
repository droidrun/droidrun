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
from droidrun.tools.adb import AdbTools

logger = logging.getLogger("droidrun")


class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.

    NOTE: not safe for production use! Use with caution.
    """

    def __init__(
        self,
        loop: AbstractEventLoop,
        locals: Dict[str, Any] = {},
        globals: Dict[str, Any] = {},
        tools={},
        tools_instance=None,
        use_same_scope: bool = True,
    ):
        """
        Initialize the code executor.

        Args:
            locals: Local variables to use in the execution context
            globals: Global variables to use in the execution context
            tools: List of tools available for execution
            tools_instance: Original tools instance (e.g., AdbTools instance)
        """

        self.tools_instance = tools_instance

        # loop throught tools and add them to globals, but before that check if tool value is async, if so convert it to sync. tools is a dictionary of tool name: function
        # e.g. tools = {'tool_name': tool_function}

        # check if tools is a dictionary
        if isinstance(tools, dict):
            logger.debug(
                f"🔧 Initializing SimpleCodeExecutor with tools: {tools.items()}"
            )
            for tool_name, tool_function in tools.items():
                if asyncio.iscoroutinefunction(tool_function):
                    # If the function is async, convert it to sync
                    tool_function = async_to_sync(tool_function)
                # Add the tool to globals
                globals[tool_name] = tool_function
        elif isinstance(tools, list):
            logger.debug(f"🔧 Initializing SimpleCodeExecutor with tools: {tools}")
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

        globals["time"] = time

        self.globals = globals
        self.locals = locals
        self.loop = loop
        self.use_same_scope = use_same_scope
        self.tools = tools
        if self.use_same_scope:
            # If using the same scope, set the globals and locals to the same dictionary
            self.globals = self.locals = {
                **self.locals,
                **{k: v for k, v in self.globals.items() if k not in self.locals},
            }

    async def execute(self, ctx: Context, code: str) -> str:
        """
        Execute Python code and capture output and return values.

        Args:
            code: Python code to execute

        Returns:
            str: Output from the execution, including print statements.
        """
        # Update UI elements before execution
        self.globals['ui_state'] = await ctx.get("ui_state", None)
        self.globals['step_screenshots'] = []
        self.globals['step_ui_states'] = []
        
        if self.tools_instance and isinstance(self.tools_instance, AdbTools):
            self.tools_instance._set_context(ctx)

        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        output = ""
        try:
            # Execute with captured output
            thread_exception = []
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):

                def execute_code():
                    try:
                        exec(code, self.globals, self.locals)
                    except Exception as e:
                        import traceback

                        thread_exception.append((e, traceback.format_exc()))

                t = threading.Thread(target=execute_code)
                t.start()
                t.join()

            # Get output
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()
            if thread_exception:
                e, tb = thread_exception[0]
                output += f"\nError: {type(e).__name__}: {str(e)}\n{tb}"

        except Exception as e:
            # Capture exception information
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        result = {
            'output': output,
            'screenshots': self.globals['step_screenshots'],
            'ui_states': self.globals['step_ui_states'],
        }
        return result
