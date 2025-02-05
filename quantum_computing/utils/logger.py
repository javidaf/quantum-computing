import logging
import sys
from pathlib import Path

_logger = None

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with console and optional file output
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "quantum_computing") -> logging.Logger:
    """
    Get or create a logger instance. If the logger hasn't been set up yet,
    it will create a default logger with console output.
    
    Args:
        name: Name of the logger (default: "quantum_computing")
    
    Returns:
        logging.Logger: Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logger(name)
    return _logger
