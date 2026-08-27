#src/prompts.py

INITIAL_RESPONSE = "Welcome to EChoAI👋"

# How many completed exchanges to carry. Six turns is roughly the span over
# which a speaker keeps referring back ("that number", "as I said") without
# making the prompt so long that the model loses the current question in it.
DEFAULT_HISTORY_TURNS = 6

# Rules appended to the operator's system role.
#
# These used to live in the user message, restated in full on every turn
# alongside a one-line "transcript" that held only the previous question. With
# that little context the model could not tell a genuinely new question from a
# rephrasing, and answered "How may I assist you today?" over and over in a
# live run. Conversation history now goes in the messages array as actual
# turns, which is what the model is trained to read, and the standing rules
# stay in the system message where they are stated once.
RESPONSE_RULES = """
## Responding

You are hearing a live transcript. It is produced by speech recognition and
will contain errors: mis-heard words, missing punctuation, and sentences cut
short. Infer intent rather than reacting to the literal text.

## Who is speaking

Each line is attributed, and the attribution is not part of what was said:

    Speaker 1: ...       one of the other people present
    Speaker 2 (unsure):  the same, where the voice match was uncertain
    Me: ...              the person you are writing for

There may be several other speakers. They are told apart by voice, which is
imperfect: the same person can appear under two numbers, and two people can
share one. Treat the numbers as a hint about who is who, not as fact, and rely
on what is being said when they disagree.

Write what **Me** should say next, in the role the instructions above give you.
Never write a line for anyone else, and never repeat the attribution in your
reply.

1. Answer the most recent thing said, in the context of the whole passage --
   a question is often spread over several turns.
2. Use the same language the others are using.
3. Keep it short. It will be read or spoken aloud while the conversation is
   still going.
4. The conversation so far is given as previous turns. Do not repeat an
   answer you have already given: if nothing has been added that changes what
   you would say, reply with exactly `None`.
5. Put your reply in square brackets: [like this].

`None` is not a failure. Silence is correct when the speaker is still making
the same point, thinking aloud, or saying something that needs no response.
"""


def build_messages(system_role, history, question,
                   max_turns=DEFAULT_HISTORY_TURNS):
    """
    Assemble the chat messages for one question.

    Args:
        system_role: the operator's persona template.
        history: completed (question, answer) pairs, oldest first.
        question: what the speaker just said.
        max_turns: how many past exchanges to include.

    Returns a standard messages list: the dialogue is expressed as alternating
    user/assistant turns rather than being flattened into a single blob, so
    the model can see what it has already said and honour rule 4.
    """
    messages = [{
        "role": "system",
        "content": (system_role or "").rstrip() + "\n" + RESPONSE_RULES,
    }]

    for asked, answered in list(history)[-max_turns:]:
        if not asked:
            continue
        messages.append({"role": "user", "content": asked})
        if answered:
            messages.append({"role": "assistant", "content": "[{}]".format(answered)})

    messages.append({"role": "user", "content": question})
    return messages


def create_prompt(transcript, lastContent, latest_response_text=""):
    """
    Deprecated single-blob prompt builder.

    Superseded by build_messages, which passes the conversation as real turns.
    Kept so that anything still importing it keeps working.
    """
    assistant_context = (
        f"\nMy last response:\n[{latest_response_text}]\n"
        if latest_response_text and latest_response_text != "None"
        else "\nNo previous response from me.\n"
    )
    return f"""
    Below is a transcription of the conversation with potential inaccuracies:

    {transcript}

    {assistant_context}

    The latest speech from the speaker (may not be completely accurate):
    [{lastContent}]

    Reply in square brackets, in the speaker's language, or `None` if you have
    nothing new to add.
    """
