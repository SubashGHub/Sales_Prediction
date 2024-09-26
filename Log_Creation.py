import sys
from datetime import datetime


class Logger:
    current_time = datetime.now().replace(microsecond=0)

    def create_log(self, value_to_print):

        # Redirect standard output to a file
        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open('output.txt', 'a') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(value_to_print)
            sys.stdout = original_stdout  # Reset the standard output to its original value
