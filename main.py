
import logging

from utils.logger import setup_logger, get_logger

# Setup once at the start of your application
setup_logger(
    name="quantum_computing",
    log_file="logs/quantum_computing.log",
    level=logging.DEBUG
)

# Then use get_logger() everywhere else
logger = get_logger()
logger.info("Application started")