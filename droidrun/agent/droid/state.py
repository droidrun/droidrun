from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict


class DroidAgentState(BaseModel):
    """
    State model for DroidAgent workflow - shared across parent and child workflows.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Task context
    instruction: str = ""
    step_number: int = 0
    runtype: str = "developer"
    user_id: str | None = None

    # ========================================================================
    # Device State (current)
    # ========================================================================
    formatted_device_state: str = ""  # Text description for prompts
    focused_text: str = ""  # Text in focused input field
    a11y_tree: List[Dict] = Field(default_factory=list)  # Raw accessibility tree
    phone_state: Dict = Field(default_factory=dict)  # Package, activity, etc.
    screenshot: str | bytes | None = None  # Current screenshot
    width: int = 0
    height: int = 0

    # ========================================================================
    # Device State (previous - for before/after comparison)
    # ========================================================================
    previous_formatted_device_state: str = ""

    # ========================================================================
    # App Tracking
    # ========================================================================
    app_card: str = ""
    current_package_name: str = ""
    current_activity_name: str = ""
    visited_packages: set = Field(default_factory=set)
    visited_activities: set = Field(default_factory=set)

    # ========================================================================
    # Unified Thought/Plan Tracking (used by all agents)
    # ========================================================================
    last_thought: str = ""  # Most recent thought from any agent
    previous_plan: str = ""  # Plan from previous iteration
    progress_summary: str = ""  # Cumulative progress (replaces each turn)

    # ========================================================================
    # Planning State (Manager sets these)
    # ========================================================================
    plan: str = ""  # Current plan
    current_subgoal: str = ""  # Current subgoal for Executor
    manager_answer: str = ""  # Final answer when complete

    # ========================================================================
    # Action Tracking
    # ========================================================================
    action_history: List[Dict] = Field(default_factory=list)
    summary_history: List[str] = Field(default_factory=list)
    action_outcomes: List[bool] = Field(default_factory=list)
    error_descriptions: List[str] = Field(default_factory=list)
    last_action: Dict = Field(default_factory=dict)
    last_summary: str = ""

    # ========================================================================
    # Memory (append-only information storage)
    # ========================================================================
    memory: str = ""

    # ========================================================================
    # Message History (for stateful agents - list of dicts)
    # ========================================================================
    message_history: List[Dict] = Field(default_factory=list)

    # ========================================================================
    # Error Handling
    # ========================================================================
    error_flag_plan: bool = False
    err_to_manager_thresh: int = 2

    # ========================================================================
    # Script Execution Tracking
    # ========================================================================
    scripter_history: List[Dict] = Field(default_factory=list)
    last_scripter_message: str = ""
    last_scripter_success: bool = True

    # ========================================================================
    # Text Manipulation Tracking
    # ========================================================================
    has_text_to_modify: bool = False
    text_manipulation_history: List[Dict] = Field(default_factory=list)
    last_text_manipulation_success: bool = False

    # ========================================================================
    # Custom Variables (user-defined)
    # ========================================================================
    custom_variables: Dict = Field(default_factory=dict)
    output_dir: str = ""

    # ========================================================================
    # GCP Upload Results
    # ========================================================================
    video_url: str | None = None  # GCS URL to trajectory video after upload

    def update_current_app(self, package_name: str, activity_name: str):
        """
        Update package and activity together, capturing telemetry event only once.

        This prevents duplicate PackageVisitEvents when both package and activity change.
        """
        # Check if either changed
        package_changed = package_name != self.current_package_name
        activity_changed = activity_name != self.current_activity_name

        if not (package_changed or activity_changed):
            return  # No change, nothing to do

        # Update tracking sets
        if package_changed and package_name:
            self.visited_packages.add(package_name)
        if activity_changed and activity_name:
            self.visited_activities.add(activity_name)

        # Update values
        self.current_package_name = package_name
        self.current_activity_name = activity_name

        # Capture telemetry event for any change
        # This ensures we track when apps close or transitions to empty state occur
        from droidrun.telemetry import PackageVisitEvent, capture

        capture(
            PackageVisitEvent(
                package_name=package_name or "Unknown",
                activity_name=activity_name or "Unknown",
                step_number=self.step_number,
            ),
            user_id=self.user_id,
        )
