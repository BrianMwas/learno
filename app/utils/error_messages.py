"""
Utility for converting technical errors to learner-friendly messages.
"""


def format_learner_error(error: Exception) -> str:
    """
    Convert technical errors to friendly, encouraging messages for learners.

    Args:
        error: The exception that occurred

    Returns:
        A friendly, learner-appropriate error message
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Map technical errors to friendly messages
    error_map = {
        "KeyError": "Hmm, I lost track of where we are. Let's start fresh! 🔄",
        "OpenAI": "I'm having trouble thinking right now. Please try again in a moment. 🤔",
        "APIError": "I'm having trouble connecting to my brain. Give me a second! 🧠",
        "RateLimitError": "Whoa, I'm thinking too fast! Let's slow down for just a moment. 😅",
        "timeout": "That's taking too long. Can you ask me that in a simpler way? ⏱️",
        "Timeout": "That's taking too long. Can you ask me that in a simpler way? ⏱️",
        "ValidationError": "Hmm, something doesn't look right with that input. Can you try again? ✏️",
        "ValueError": "Oops, that didn't make sense to me. Can you rephrase? 🤷",
        "GraphInterrupt": "I need some information from you before we continue! 📝",
        "ConnectionError": "I'm having trouble connecting. Check your internet? 🌐",
        "JSONDecodeError": "I got confused reading that. Can you try again? 📄",
    }

    # Check error type first
    for key, message in error_map.items():
        if key in error_type:
            return message

    # Then check error message content
    for key, message in error_map.items():
        if key.lower() in error_message.lower():
            return message

    # Generic fallback - still friendly!
    return "Something unexpected happened. Mind trying that again? I'm here to help! 💪"


def get_stage_error_message(stage: str) -> str:
    """
    Get stage-specific friendly error messages.

    Args:
        stage: The current learning stage

    Returns:
        A stage-appropriate error message
    """
    stage_messages = {
        "introduction": "Oops! Let's start over with introductions. Hi, I'm Meemo! 👋",
        "teaching": "Hmm, I got a bit confused while teaching. Let me try explaining that again! 📚",
        "assessment": "I had trouble with that question. Let me ask you something else! ❓",
        "evaluation_complete": "Something went wrong checking your answer. Let's move on! ✅",
        "needs_hint": "I got stuck giving you a hint. Let me try a different approach! 💡",
        "needs_retry": "Oops! Let's try that question again! 🔁",
        "needs_review": "I had trouble reviewing. Let's go over this topic one more time! 📖",
        "question_answering": "I couldn't quite answer that. Can you ask me in a different way? 🤔",
    }

    return stage_messages.get(stage, "Something went wrong, but we can keep learning! 🚀")
