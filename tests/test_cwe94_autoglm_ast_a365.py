"""
Test for CWE-94: Insufficient validation in parse_action allows
shell metacharacters in action parameters (e.g., app names for Launch).

Data flow:
  LLM response -> parse_action -> action["app"] -> _handle_launch
  -> device.launch_app -> tools.start_app -> device.shell(f"...{package}")
"""

import sys
import os
import ast
import re
import importlib
import types


def _load_parse_action():
    """
    Import parse_action directly from the module file, bypassing droidrun.__init__
    (which requires the package to be installed).
    """
    project_root = os.path.expanduser(
        "~/projects/audits/droidrun-droidrun-worktrees/cwe94-autoglm-ast-a365"
    )
    module_path = os.path.join(
        project_root, "droidrun", "agent", "external", "autoglm.py"
    )

    # Create minimal stub modules to satisfy imports
    # We only need parse_action which doesn't use any of these at runtime
    for mod_name in [
        "droidrun",
        "droidrun.agent",
        "droidrun.agent.utils",
        "droidrun.agent.utils.chat_utils",
        "droidrun.agent.utils.inference",
        "droidrun.agent.utils.llm_picker",
        "droidrun.agent.external",
        "droidrun.log_handlers",
    ]:
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            # Add stub functions/classes that might be imported
            stub.to_chat_messages = lambda x: x
            stub.acall_with_retries = lambda *a, **kw: None
            stub.load_llm = lambda *a, **kw: None
            sys.modules[mod_name] = stub

    spec = importlib.util.spec_from_file_location(
        "droidrun.agent.external.autoglm", module_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_action


parse_action = _load_parse_action()


class TestParseActionValidation:
    """Tests that parse_action validates action types and sanitizes parameters."""

    def test_normal_launch_action(self):
        """Normal Launch action should parse successfully."""
        response = 'do(action="Launch", app="com.android.settings")'
        action = parse_action(response)
        assert action["_metadata"] == "do"
        assert action["action"] == "Launch"
        assert action["app"] == "com.android.settings"

    def test_normal_tap_action(self):
        """Normal Tap action should parse successfully."""
        response = 'do(action="Tap", element=[500, 500])'
        action = parse_action(response)
        assert action["_metadata"] == "do"
        assert action["action"] == "Tap"
        assert action["element"] == [500, 500]

    def test_normal_type_action(self):
        """Normal Type action should parse successfully."""
        response = 'do(action="Type", text="hello world")'
        action = parse_action(response)
        assert action["_metadata"] == "do"
        assert action["action"] == "Type"
        assert action["text"] == "hello world"

    def test_normal_finish_action(self):
        """Normal finish action should parse successfully."""
        response = 'finish(message="Task completed.")'
        action = parse_action(response)
        assert action["_metadata"] == "finish"

    def test_shell_injection_in_app_name_semicolon(self):
        """App name with shell metacharacter ';' should be rejected."""
        response = 'do(action="Launch", app="com.test; id")'
        try:
            action = parse_action(response)
            assert ";" not in action.get("app", ""), (
                f"Shell metacharacter ';' in app name was not rejected: {action['app']}"
            )
        except ValueError:
            pass  # Rejecting is also acceptable

    def test_shell_injection_in_app_name_backtick(self):
        """App name with shell metacharacter '`' should be rejected."""
        response = 'do(action="Launch", app="com.test`whoami`")'
        try:
            action = parse_action(response)
            assert "`" not in action.get("app", ""), (
                f"Shell metacharacter '`' in app name was not rejected: {action['app']}"
            )
        except ValueError:
            pass

    def test_shell_injection_in_app_name_dollar(self):
        """App name with shell metacharacter '$(' should be rejected."""
        response = 'do(action="Launch", app="com.test$(whoami)")'
        try:
            action = parse_action(response)
            assert "$" not in action.get("app", ""), (
                f"Shell metacharacter '$' in app name was not rejected: {action['app']}"
            )
        except ValueError:
            pass

    def test_shell_injection_in_app_name_pipe(self):
        """App name with shell metacharacter '|' should be rejected."""
        response = 'do(action="Launch", app="com.test|cat /data")'
        try:
            action = parse_action(response)
            assert "|" not in action.get("app", ""), (
                f"Shell metacharacter '|' in app name was not rejected: {action['app']}"
            )
        except ValueError:
            pass

    def test_shell_injection_in_app_name_ampersand(self):
        """App name with shell metacharacter '&&' should be rejected."""
        response = 'do(action="Launch", app="com.test&&echo pwned")'
        try:
            action = parse_action(response)
            assert "&" not in action.get("app", ""), (
                f"Shell metacharacter '&' in app name was not rejected: {action['app']}"
            )
        except ValueError:
            pass

    def test_unknown_action_type_rejected(self):
        """Unknown action types should be rejected."""
        response = 'do(action="ExecuteShell", command="id")'
        try:
            action = parse_action(response)
            assert action.get("action") in {
                "Launch", "Tap", "Type", "Type_Name", "Swipe", "Back", "Home",
                "Double Tap", "Long Press", "Wait", "Take_over", "Note",
                "Call_API", "Interact",
            }, f"Unknown action type was accepted: {action.get('action')}"
        except ValueError:
            pass  # Rejecting is acceptable

    def test_ast_function_name_not_do(self):
        """Function calls other than 'do()' should be rejected in do-branch."""
        # This starts with "do" prefix so it enters the do-branch,
        # but the AST function name should be validated
        response = 'do_evil(action="Tap", element=[500, 500])'
        try:
            action = parse_action(response)
            # Should have been rejected because func name != "do"
            assert False, f"Non-do function call was accepted: {action}"
        except ValueError:
            pass  # Expected

    def test_friendly_app_name_with_spaces(self):
        """Friendly app names with spaces should work."""
        response = 'do(action="Launch", app="Google Chrome")'
        action = parse_action(response)
        assert action["action"] == "Launch"
        assert action["app"] == "Google Chrome"

    def test_app_name_with_dots_underscores(self):
        """Package-style app names with dots and underscores should work."""
        response = 'do(action="Launch", app="com.android.vending")'
        action = parse_action(response)
        assert action["action"] == "Launch"
        assert action["app"] == "com.android.vending"

    def test_swipe_action(self):
        """Normal Swipe action should parse."""
        response = 'do(action="Swipe", start=[100,200], end=[300,400])'
        action = parse_action(response)
        assert action["action"] == "Swipe"
        assert action["start"] == [100, 200]
        assert action["end"] == [300, 400]

    def test_back_action(self):
        """Normal Back action should parse."""
        response = 'do(action="Back")'
        action = parse_action(response)
        assert action["action"] == "Back"


def run_tests():
    """Run all tests and report results."""
    test = TestParseActionValidation()
    methods = [m for m in dir(test) if m.startswith("test_")]
    passed = 0
    failed = 0
    for method_name in sorted(methods):
        method = getattr(test, method_name)
        try:
            method()
            print(f"  PASS: {method_name}")
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {method_name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
