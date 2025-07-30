import traceback
import sys
from logger.custom_logger import CustomLogger
logger=CustomLogger().get_logger(__file__)

class DocumentPortalException(Exception):
    """Custom exception for Document Portal"""
    def __init__(self, error_message, error_details=None):
        _, _, exc_tb = sys.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
        self.lineno = exc_tb.tb_lineno if exc_tb else "Unknown"
        self.error_message = str(error_message)
        if error_details is not None and hasattr(error_details, "__traceback__"):
            self.traceback_str = ''.join(traceback.format_exception(type(error_details), error_details, error_details.__traceback__))
        else:
            self.traceback_str = "No traceback available."
        
    def __str__(self):
       return f"""
        Error in [{self.file_name}] at line [{self.lineno}]
        Message: {self.error_message}
        Traceback:
        {self.traceback_str}
        """
    
if __name__ == "__main__":
    try:
        # Simulate an error
        a = 1 / 0
        print(a)
    except Exception as e:
        app_exc=DocumentPortalException(e)
        logger.error(app_exc)
        raise app_exc