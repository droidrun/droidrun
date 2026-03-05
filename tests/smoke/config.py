"""Smoke test matrix configuration."""

from dataclasses import dataclass, field


@dataclass
class SmokeTestConfig:
    name: str
    reasoning: bool
    vision: bool
    max_steps: int
    task: str
    output_schema: bool = False
    credentials: bool = False
    expected_package: str = ""
    assertions: list[str] = field(default_factory=list)


SMOKE_TESTS: list[SmokeTestConfig] = [
    SmokeTestConfig(
        name="fast-no-vision",
        reasoning=False,
        vision=False,
        max_steps=15,
        task="Go to Settings and find the Android version number",
        expected_package="settings",
        assertions=["result_success", "package_name"],
    ),
    SmokeTestConfig(
        name="fast-vision",
        reasoning=False,
        vision=True,
        max_steps=15,
        task="Go to Settings and find the Android version number",
        output_schema=True,
        expected_package="settings",
        assertions=["result_success", "structured_output", "package_name"],
    ),
    SmokeTestConfig(
        name="reasoning-no-vision",
        reasoning=True,
        vision=False,
        max_steps=30,
        task="Open Chrome, tap the search bar, and use the type_secret tool to type the saved credential into it",
        credentials=True,
        expected_package="chrome",
        assertions=["type_secret_called"],
    ),
    SmokeTestConfig(
        name="reasoning-vision",
        reasoning=True,
        vision=True,
        max_steps=30,
        task="Go to Settings and find the Android version number",
        expected_package="settings",
        assertions=["result_success", "package_name"],
    ),
]
