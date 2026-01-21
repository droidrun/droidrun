import asyncio
from droidrun import DroidAgent, DroidrunConfig

WHATSAPP_CALENDAR_GOAL = """
Open the WhatsApp app on the Android device.
Scan recent chat messages to identify messages that contain meeting-related
information such as date, time, or meeting intent.

When such information is detected:
- Extract the relevant details (date, time, title, description)
- Open the Google Calendar app
- Create a new calendar event with the extracted meeting information

After creating the event:
- Return to WhatsApp
- Continue scanning other messages
- Repeat the process until all important messages are covered

If no meeting-related messages are found:
- Wait for a while
- Repeat the scanning process periodically
"""

async def main():
    config = DroidrunConfig()

    agent = DroidAgent(
        goal=WHATSAPP_CALENDAR_GOAL,
        config=config,
    )

    result = await agent.run()

    print("\n===== AUTOMATION RESULT =====")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")
    print(f"Steps taken: {result.steps}")

if __name__ == "__main__":
    asyncio.run(main())
