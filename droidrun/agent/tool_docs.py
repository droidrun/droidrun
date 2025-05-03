# Tool documentation with exact parameter names
tool_docs = {

    "remember": "remember(memory: str) - Store an important fact, insight, or observation in your memory for future reference. ",

    "tap": "tap(index: int, longpress: bool) - Tap on the element with the given index on the device",
    
    "swipe": "swipe(start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 300) - Swipe from (start_x,start_y) to (end_x,end_y) over duration_ms milliseconds",
    
    "input_text": "input_text(text: str) - Input text on the device - this works only if an input is focused. Always make sure that an edit field was tapped before inserting text",
    
    "press_key": "press_key(keycode: int) - Press a key on the device using keycode",
    
    "start_app": "start_app(name: str) - IMPORTANT: ALWAYS use this to Start an app using its name (e.g., 'Google Playstore' or 'Youtube')",
    
    "complete": "complete(result: str) - IMPORTANT: This tool should ONLY be called after you have ACTUALLY completed all necessary actions for the goal. It does not perform any actions itself - it only signals that you have already achieved the goal through other actions. Include a summary of what was accomplished as the result parameter.",
    
}