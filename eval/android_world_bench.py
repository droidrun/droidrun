#!/usr/bin/env python3
"""
Android World Benchmark Runner for DroidRun

This script provides integration with the AndroidWorld benchmark suite,
allowing evaluation of DroidRun against standard tasks.
"""

import os
import sys
import argparse
import logging
import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("android_world_bench")

# Import utility modules
from eval.utils.environment import (
    initialize_environment,
    get_device_serial,
)

from eval.utils.accessibility import enable_accessibility_service
from eval.utils.task_manager import (
    TaskRegistry,
    initialize_task,
    get_task_description,
    check_task_success,
    teardown_task,
)
from eval.utils.results import (
    ResultManager,
    create_task_result,
    update_result_from_agent,
    ProgressTracker,
)
from eval.utils.agent import create_agent, run_agent
from eval.utils.keepalive import OverlayKeepalive, disable_overlay_once


# Now import AndroidWorld modules
try:
    from android_world import registry
except ImportError as e:
    logger.error(f"Failed to import AndroidWorld modules: {e}")
    logger.error("Please make sure AndroidWorld is correctly installed")
    sys.exit(1)


class AndroidWorldBenchmark:
    """Benchmark DroidRun using AndroidWorld tasks."""

    def __init__(
        self,
        task_ids: Optional[List[int]] = None,
        task_names: Optional[List[str]] = None,
        llm_provider: str = "OpenAI",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        adb_path: str = "adb",
        console_port: int = 5554,
        perform_emulator_setup: bool = False,
        random_seed: int = 42,
        results_dir: str = "eval_results",
        max_steps_per_task: int = 50,
        task_family: str = registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        n_task_combinations: int = 1,
        portal_service_name: str = "com.droidrun.portal/com.droidrun.portal.DroidrunPortalService",
        progress_file: str = "task_progress.json",
        use_keepalive: bool = True,
        keepalive_interval: int = 5,
        reasoning: bool = False,
    ):
        """Initialize the benchmark.

        Args:
            task_ids: List of task IDs to run (1-116)
            task_names: List of specific task names to run
            llm_provider: LLM provider to use (OpenAI, Anthropic, Gemini, etc.)
            llm_model: Model name to use
            temperature: Temperature for LLM sampling
            adb_path: Path to ADB executable
            console_port: Emulator console port
            perform_emulator_setup: Whether to perform initial emulator setup
            random_seed: Random seed for reproducibility
            results_dir: Directory to save results
            max_steps_per_task: Maximum steps to allow per task
            task_family: Task family to benchmark
            n_task_combinations: Number of parameter combinations per task
            portal_service_name: Name of the DroidRun accessibility service
            progress_file: Path to progress tracking file
            use_keepalive: Whether to use the keepalive service
            keepalive_interval: Interval in seconds for the keepalive service
        """
        self.task_ids = task_ids
        self.task_names = task_names
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.reasoning = reasoning
        self.adb_path = adb_path
        self.console_port = console_port
        self.perform_emulator_setup = perform_emulator_setup
        self.random_seed = random_seed
        self.results_dir = results_dir
        self.max_steps_per_task = max_steps_per_task
        self.task_family = task_family
        self.n_task_combinations = n_task_combinations
        self.portal_service_name = portal_service_name
        self.progress_file = progress_file
        self.use_keepalive = use_keepalive
        self.keepalive_interval = keepalive_interval

        # Components initialized during setup
        self.env = None
        self.device_serial = None
        self.task_registry = None
        self.result_manager = None
        self.progress_tracker = None
        self.keepalive_service = None

    async def initialize(self):
        """Initialize all components needed for the benchmark."""
        # Initialize environment
        self.env = initialize_environment(
            adb_path=self.adb_path,
            console_port=self.console_port,
            perform_setup=self.perform_emulator_setup,
        )
        if not self.env:
            logger.error("Failed to initialize environment. Exiting.")
            sys.exit(1)

        # Get device serial
        self.device_serial = await get_device_serial()
        if not self.device_serial:
            logger.error("Failed to get device serial. Exiting.")
            sys.exit(1)

        # Enable accessibility service
        accessibility_enabled = await enable_accessibility_service(
            adb_path=self.adb_path,
            device_serial=self.device_serial,
            service_name=self.portal_service_name,
            disable_first=True,
        )
        if not accessibility_enabled:
            logger.error("Failed to enable accessibility service. Exiting.")
            sys.exit(1)

        # Initialize task registry
        self.task_registry = TaskRegistry(task_family=self.task_family)

        # Initialize result manager
        self.result_manager = ResultManager(results_dir=self.results_dir)

        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(progress_file=self.progress_file)

        # Initialize keepalive service
        self.keepalive_service = OverlayKeepalive(
            device_serial=self.device_serial, interval=self.keepalive_interval
        )

        logger.info("Benchmark initialization complete")

    def list_tasks(self):
        """Print the list of available tasks."""
        if not self.task_registry:
            self.task_registry = TaskRegistry(task_family=self.task_family)

        task_ids = self.task_registry.get_task_ids()

        print("\nAvailable AndroidWorld Tasks:")
        print("-----------------------------")
        print(f"{'ID':<4} {'Task Name':<50}")
        print("-" * 55)

        for task_id, task_name in sorted(task_ids.items()):
            print(f"{task_id:<4} {task_name:<50}")

    async def run_task(self, task_name: str, task_instance):
        """Run a single task.

        Args:
            task_name: Name of the task
            task_instance: Task instance to run

        Returns:
            Task result
        """
        logger.info(f"Running task: {task_name}")

        # Get task description
        task_description = get_task_description(task_instance)
        logger.info(f"Task description: {task_description}")

        # Create initial result
        task_result = create_task_result(task_name, task_description)
        task_result["max_steps"] = self.max_steps_per_task

        # Initialize task
        task_initialized = await initialize_task(self.env, task_instance)
        if not task_initialized:
            task_result["error"] = "Failed to initialize task"
            return task_result

        # Enable accessibility service for the task
        await enable_accessibility_service(
            adb_path=self.adb_path,
            device_serial=self.device_serial,
            service_name=self.portal_service_name,
            disable_first=True,
        )

        # Start keepalive service if enabled
        if self.use_keepalive and self.keepalive_service:
            self.keepalive_service.start()

        # Create and run agent
        start_time = time.time()
        agent = None
        tools_instance = None
        try:
            # Create agent
            agent, agent_config = create_agent(
                device_serial=self.device_serial,
                task_description=task_description,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                temperature=self.temperature,
                max_steps=self.max_steps_per_task,
                debug=True,
                reasoning=self.reasoning,
            )

            # Store tools instance for screenshots
            tools_instance = agent.tools_instance

            # Run agent
            agent_result = await run_agent(agent, task_name)

            task_dir = os.path.join(
                self.results_dir, f"task_{task_name.replace(' ', '_')}"
            )

            task_result["screenshot_gif"] = agent.trajectory.create_screenshot_gif(
                output_path=os.path.join(task_dir, f"task_execution")
            )

            # Update result with agent information
            task_result = update_result_from_agent(task_result, agent_result, agent)

            # Add screenshots to result if available
            if hasattr(tools_instance, "screenshots") and tools_instance.screenshots:
                task_result["screenshots"] = tools_instance.screenshots

        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            task_result["error"] = str(e)
        finally:
            # Stop keepalive service if it was started
            if (
                self.use_keepalive
                and self.keepalive_service
                and self.keepalive_service.running
            ):
                self.keepalive_service.stop()

        # Set execution time
        end_time = time.time()
        task_result["execution_time"] = end_time - start_time

        # Check if task was successful
        task_result["success"] = check_task_success(self.env, task_instance)

        return task_result

    async def run_benchmark(self):
        """Run the benchmark on the selected tasks."""
        # Initialize components
        await self.initialize()

        # Create task suite
        task_suite = self.task_registry.create_task_suite(
            task_ids=self.task_ids,
            task_names=self.task_names,
            n_combinations=self.n_task_combinations,
            random_seed=self.random_seed,
        )

        if not task_suite:
            logger.error("No tasks to run")
            return

        # Get the number of completed tasks to skip
        skip_count = self.progress_tracker.get_completed_tasks()
        if skip_count > 0:
            logger.info(f"Skipping {skip_count} completed tasks")
            if skip_count >= len(task_suite):
                logger.info("All tasks already completed")
                return
            task_suite = task_suite[skip_count:]

        logger.info(f"Running benchmark with {len(task_suite)} tasks")
        completed_count = skip_count

        try:
            # Run each task
            for i, (task_name, task_instance) in enumerate(task_suite):
                try:
                    # Create a task-specific directory in results_dir
                    task_dir = os.path.join(
                        self.results_dir, f"task_{task_name.replace(' ', '_')}"
                    )
                    os.makedirs(task_dir, exist_ok=True)

                    # Run the task
                    task_result = await self.run_task(task_name, task_instance)

                    # Save the result JSON
                    json_path = os.path.join(task_dir, "result.json")
                    with open(json_path, "w") as f:
                        json.dump(task_result, f, indent=2)

                    # Save the result in the result manager as well
                    self.result_manager.save_task_result(task_result)

                    # Update progress if task was successful
                    if task_result["success"]:
                        completed_count += 1
                        self.progress_tracker.update_progress(completed_count)

                except Exception as e:
                    logger.error(f"Error running task {task_name}: {e}")
                finally:
                    # Tear down the task
                    teardown_task(self.env, task_instance)

            # Print summary
            self.result_manager.print_summary()
        finally:
            # Make sure keepalive service is stopped
            if self.keepalive_service and self.keepalive_service.running:
                self.keepalive_service.stop()

            # Close environment
            if self.env:
                logger.info("Closing environment")
                self.env.close()


async def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run AndroidWorld benchmark tasks with DroidRun"
    )

    # Task selection arguments
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task-ids", type=int, nargs="+", help="Task IDs to run (1-116)"
    )
    task_group.add_argument(
        "--task-names", type=str, nargs="+", help="Task names to run"
    )
    task_group.add_argument(
        "--list-tasks", action="store_true", help="List available tasks and exit"
    )
    task_group.add_argument(
        "--n-task-combinations",
        type=int,
        default=1,
        help="Number of parameter combinations per task",
    )

    # LLM configuration
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--llm-provider",
        type=str,
        default="OpenAI",
        help="LLM provider (OpenAI, Anthropic, Gemini, etc.)",
    )
    llm_group.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini", help="Model name to use"
    )
    llm_group.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for LLM sampling"
    )
    llm_group.add_argument(
        "--reasoning", type=bool, default=False, help="Enable reasoning for LLM"
    )

    # Environment configuration
    env_group = parser.add_argument_group("Environment Configuration")
    env_group.add_argument(
        "--adb-path", type=str, default="adb", help="Path to ADB executable"
    )
    env_group.add_argument(
        "--console-port", type=int, default=5554, help="Emulator console port"
    )
    env_group.add_argument(
        "--perform-emulator-setup",
        action="store_true",
        help="Perform initial emulator setup (install apps, set permissions)",
    )
    env_group.add_argument(
        "--portal-service",
        type=str,
        default="com.droidrun.portal/com.droidrun.portal.DroidrunPortalService",
        help="Name of the DroidRun accessibility service",
    )
    env_group.add_argument(
        "--no-keepalive",
        action="store_true",
        help="Disable the keepalive service for overlay toggling",
    )
    env_group.add_argument(
        "--keepalive-interval",
        type=int,
        default=5,
        help="Interval in seconds for the keepalive service",
    )

    # Benchmark configuration
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    bench_group.add_argument(
        "--results-dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )
    bench_group.add_argument(
        "--max-steps", type=int, default=50, help="Maximum steps per task"
    )
    bench_group.add_argument(
        "--task-family",
        type=str,
        default=registry.TaskRegistry.ANDROID_WORLD_FAMILY,
        help="Task family to benchmark",
    )
    bench_group.add_argument(
        "--progress-file",
        type=str,
        default="task_progress.json",
        help="File to track task progress",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = AndroidWorldBenchmark(
        task_ids=args.task_ids,
        task_names=args.task_names,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
        adb_path=args.adb_path,
        console_port=args.console_port,
        perform_emulator_setup=args.perform_emulator_setup,
        random_seed=args.random_seed,
        results_dir=args.results_dir,
        max_steps_per_task=args.max_steps,
        task_family=args.task_family,
        n_task_combinations=args.n_task_combinations,
        portal_service_name=args.portal_service,
        progress_file=args.progress_file,
        use_keepalive=not args.no_keepalive,
        keepalive_interval=args.keepalive_interval,
        reasoning=args.reasoning,
    )

    # Just list tasks if requested
    if args.list_tasks:
        benchmark.list_tasks()
        return

    # Run the benchmark
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main())
