import re


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Add the previous user question when the new question is a follow-up."""
    if not history:
        return question

    previous_questions = [
        message.get("content", "").strip()
        for message in history
        if message.get("role") == "user" and message.get("content", "").strip()
    ]
    if not previous_questions:
        return question

    # Short questions and pronouns often depend on the previous question's topic.
    follow_up_words = r"\b(it|they|them|this|that|these|those|more|why|how)\b"
    is_follow_up = len(question.split()) <= 8 or re.search(follow_up_words, question.lower())
    if not is_follow_up:
        return question

    return f"Previous question: {previous_questions[-1]} Follow-up question: {question}"
