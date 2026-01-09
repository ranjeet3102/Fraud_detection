import logging
import os

def get_logger(name="pipeline", log_to_file=False, log_dir="logs"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        ))
        logger.addHandler(console_handler)

        # Optional: File handler
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
            ))
            logger.addHandler(file_handler)

    return logger
