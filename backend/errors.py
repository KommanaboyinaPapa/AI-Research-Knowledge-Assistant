class RAGError(Exception):
    """An expected error that can be shown safely to an API client."""

    def __init__(self, error_type: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.status_code = status_code


FALLBACK_ERROR = "I could not find the answer in the provided documents."
