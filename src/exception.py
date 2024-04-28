"""
Custom Exception Handling
"""

import sys
from src.logger import logging

# function to load the execution info and return the error message
def error_message_detail(error, error_detail: sys):
    #exc_info returns the info about error
    _,_,exc_tb = error_detail.exc_info()
    
    # get file name from the exc_info() function
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # construct the error message to return
    error_message = "Error occured in Python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    
    return error_message

# inheriting the exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
    
    
        