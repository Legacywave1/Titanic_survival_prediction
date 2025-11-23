import os
import sys
from typing import Any

def error_message_detail(error: Any, error_detail: Any = None) -> str:
    """
    Works whether you pass sys, None, or nothing at all.
    Handles the case when sys.exc_info() is broken during import (Docker/Uvicorn).
    """
    # If caller passed sys (your old code) → use it safely
    if error_detail is not None and error_detail is sys:
        exc_info = sys.exc_info()
    else:
        exc_info = error_detail or sys.exc_info()

    # exc_info can be (type, value, traceback) or broken during import → guard everything
    if exc_info and len(exc_info) == 3:
        exc_type, exc_value, exc_tb = exc_info
        if exc_tb:
            filename = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
            lineno = exc_tb.tb_lineno
            return f"Error in {filename} at line {lineno}: {error}"


    return f"Error occurred: {error}"

class CustomException(Exception):
    def __init__(self, error_message: Any, error_detail: Any = None):
        self.error_message = error_message_detail(error_message, error_detail)
        super().__init__(self.error_message)
