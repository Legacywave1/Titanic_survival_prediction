import sys

def error_message_detail(error, error_details):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error occured in python script {file_name} at line [{exc_tb.tb_lineno}] error message [{str(error)}]'
    return error_message

class CustomException(Exception):
    def __init__(self, error_message:str, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

        def __str__(self):
            return self.error_message
