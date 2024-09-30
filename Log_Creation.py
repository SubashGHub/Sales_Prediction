import logging


class Logger:
    def __init__(self, log_file='output.log'):
        # Configure the logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',  # Append mode
            format='%(asctime)s - %(levelname)s \n %(message)s',  # Log format with timestamps
            level=logging.INFO  # Log level set to INFO
        )

    def create_log(self, log_message):
        """Logs the provided message to the log file."""
        try:
            logging.info(log_message)  # Log the message with an INFO level
        except Exception as e:
            logging.error(f"Failed to log message: {e}")
