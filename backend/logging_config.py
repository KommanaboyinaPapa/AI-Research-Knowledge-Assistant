import json
import logging


class SafeJsonFormatter(logging.Formatter):
    """Format useful event fields without including request secrets or content."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for name in ("request_id", "path", "method", "status_code", "latency_ms", "document_id", "document_filename", "count"):
            value = getattr(record, name, None)
            if value is not None:
                event[name] = value
        return json.dumps(event)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("rag")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(SafeJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
