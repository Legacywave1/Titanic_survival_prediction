# src/exception.py
import sys
import traceback

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail if error_detail else sys.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in {file_name} at line {line_number}: {error}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        self.error_message = error_message_detail(error_message, error_detail)
        super().__init__(self.error_message)
