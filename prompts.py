#prompts.py

INITIAL_RESPONSE = "Welcome to EChoAI👋"


def create_prompt(transcript, lastContent, latest_response_text=""):
    assistant_context = (
        f"\nMy last response:\n[{latest_response_text}]\n"
        if latest_response_text and latest_response_text!="None"
        else "\nNo previous response from me.\n"
    )

    return f"""
    Below is a transcription of the conversation with potential inaccuracies. The records are ordered from the most recent to the oldest:

    {transcript}

    {assistant_context}

    The latest speech from the speaker (may not be completely accurate):
    [{lastContent}]

    Instructions:
    1. The above records are ordered chronologically from newest to oldest.
    2. IMPORTANT: If I have not responded before (no previous response), I should provide a response now.
    3. If the latest speech is semantically similar to the previous records AND my last response already addressed the same type of request (like number confirmation, registration questions, etc.), return 'None'.
    4. Ensure your response maintains the current conversation context.
    5. Follow your role and system rules strictly.
    6. Use the same language as the client.
    7. Frame your response in square brackets [ ].
    
    IMPORTANT CONSIDERATION BEFORE RESPONDING:
    - Compare the semantic meaning of the latest speech with recent records, not just exact matches
    - If I recently asked for clarification about numbers/registration and the user is still discussing numbers, I should return empty string rather than asking for clarification again
    - Only generate a new response if there's a meaningful change in the conversation context

    If this is a semantically similar input AND I have already responded appropriately, return 'None'. Otherwise, provide your response:
    """