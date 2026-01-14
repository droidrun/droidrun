"""
VerificationAgent - Holistic test case verification.

Analyzes all test execution data to determine if a test case truly passed or failed,
accounting for transient UI states, timing issues, and minor observation mismatches
that don't affect the actual test outcome.
"""

import logging
from typing import Dict, List, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field

from droidrun.agent.utils.inference import astructured_predict_with_retries

logger = logging.getLogger("droidrun")


class VerificationResult(BaseModel):
    """Result from the verification agent."""

    status: str = Field(
        description="Final test status: 'passed' or 'failed'"
    )
    reasoning: str = Field(
        description="Brief explanation of why the test passed or failed (2-3 sentences)"
    )
    confidence: float = Field(
        description="Confidence level in the verdict (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    false_negative_detected: bool = Field(
        default=False,
        description="True if a step was marked as failed but the test actually succeeded",
    )
    critical_failures: List[str] = Field(
        default_factory=list,
        description="List of actual critical failures that caused the test to fail",
    )


VERIFICATION_PROMPT = PromptTemplate(
    """You are a QA verification expert. Your job is to determine if a UI/UX test case
truly passed or failed by analyzing all available execution data holistically.

## Important Context
Some test steps may be marked as "FAIL" due to:
- Transient UI states (loading screens that appear/disappear quickly)
- Timing issues (expected screen appeared but observation was delayed)
- Minor text/UI differences that don't affect functionality

These are FALSE NEGATIVES - the test actually succeeded but was incorrectly marked as failed.

**Critical**
If all the steps are passed, then directly return the result as pass with a summarised view of the Test Case.

## Your Task
Analyze the test execution data and determine:
1. Did the test achieve its ACTUAL OBJECTIVE (not just individual step expectations)?
2. Are any "failures" actually just observation timing issues?
3. Is the final state correct (user ended up where they should be)?

## Test Case Information
**Goal/Description:** {goal}

**Step-by-Step Results:**
{step_results}

**Final Agent Reason:** {final_reason}

**Final State Observation:** {final_state}

## Decision Guidelines
- Mark as PASSED if: The actual user journey completed successfully, even if some intermediate
  observations were missed (like a loading screen that was too fast to capture) OR All the steps were marked as passed.
- Mark as FAILED if: There were actual functional failures (button not found, wrong screen
  reached, errors occurred, user cannot proceed)

Analyze the data and provide your verification verdict."""
)


class VerificationAgent(Workflow):
    """
    Agent that performs holistic verification of test case results.

    Instead of relying on strict step-by-step pass/fail, this agent analyzes
    all execution data to determine if the test case truly achieved its objective.
    """

    def __init__(
        self,
        llm: LLM,
        goal: str,
        step_results: List[Dict],
        final_reason: str,
        final_state: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the verification agent.

        Args:
            llm: The LLM to use for verification
            goal: The test case goal/description
            step_results: List of step results with status and observations
            final_reason: The final reason from the test execution
            final_state: Optional final device state description
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.goal = goal
        self.step_results = step_results
        self.final_reason = final_reason
        self.final_state = final_state or "Not available"

    def _format_step_results(self) -> str:
        """Format step results for the prompt."""
        formatted = []
        for i, step in enumerate(self.step_results, 1):
            status = step.get("status", "UNKNOWN")
            expected = step.get("expected", "N/A")
            observed = step.get("observed", "N/A")
            action = step.get("action", "N/A")

            formatted.append(
                f"Step {i}: [{status}]\n"
                f"  Action: {action}\n"
                f"  Expected: {expected}\n"
                f"  Observed: {observed}"
            )
        return "\n\n".join(formatted) if formatted else "No step data available"

    @step
    async def verify_test_result(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """
        Perform holistic verification of test results.
        """
        logger.debug("🔍 Running verification agent...")

        try:
            step_results_formatted = self._format_step_results()

            result = await astructured_predict_with_retries(
                self.llm,
                VerificationResult,
                VERIFICATION_PROMPT,
                goal=self.goal,
                step_results=step_results_formatted,
                final_reason=self.final_reason,
                final_state=self.final_state,
            )

            logger.info(
                f"✅ Verification complete: {result.status.upper()} "
                f"(confidence: {result.confidence:.0%})"
            )

            if result.false_negative_detected:
                logger.info("📝 False negative detected - test was incorrectly marked as failed")

            return StopEvent(
                result={
                    "verification_result": result,
                    "success": True,
                    "error_message": "",
                }
            )

        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")

            # Return a default failed result on error
            return StopEvent(
                result={
                    "verification_result": VerificationResult(
                        status="failed",
                        reasoning=f"Verification agent encountered an error: {str(e)}",
                        confidence=0.0,
                        false_negative_detected=False,
                        critical_failures=[str(e)],
                    ),
                    "success": False,
                    "error_message": str(e),
                }
            )


async def verify_test_case(
    llm: LLM,
    goal: str,
    action_history: List[Dict],
    summary_history: List[str],
    action_outcomes: List[bool],
    final_reason: str,
    final_state: Optional[str] = None,
) -> VerificationResult:
    """
    Convenience function to run verification on test results.

    Args:
        llm: The LLM to use
        goal: Test case goal/description
        action_history: List of actions taken
        summary_history: List of step summaries
        action_outcomes: List of step pass/fail results
        final_reason: Final reason from execution
        final_state: Optional final device state

    Returns:
        VerificationResult with the holistic verdict
    """
    # Build step results from the available data
    step_results = []
    for i, summary in enumerate(summary_history):
        status = "PASS" if i < len(action_outcomes) and action_outcomes[i] else "FAIL"
        action = action_history[i] if i < len(action_history) else {}

        step_results.append({
            "status": status,
            "action": action.get("thought", "") or action.get("code", ""),
            "expected": "See test case definition",
            "observed": summary,
        })

    agent = VerificationAgent(
        llm=llm,
        goal=goal,
        step_results=step_results,
        final_reason=final_reason,
        final_state=final_state,
    )

    result = await agent.run()
    return result["verification_result"]
