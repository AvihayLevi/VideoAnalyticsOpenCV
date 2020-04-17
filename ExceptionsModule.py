import sys, traceback

VACOVE_UNKNOWN_EXIT_CODE=50
VACOVE_TOO_MANY_EXCEPTIONS_EXIT_CODE=99
VACOVE_NO_MONITOR=7
VACOVE_HTTP_STATUS_CODE_EXIT_CODE=3

class VAOCVError(Exception):
    """
    This is a base class Error for the VideoAnalyticsOpenCV (don't use this)  
    """
    pass


class UnknownVAOCVError(VAOCVError):
    """
    This is a general Error, if you don't to know which Error to throw, 
    """
    def __init__(self, message="Unknown Error"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.message = message


class OCRSocketVAOCVError(VAOCVError):
    """
    This error is sent when trying to post to a socket for 5 time is failed 
    """
    def __init__(self, message="Failed to send data via OCR"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.message = message


class NoMonitorIDVAOCVError(VAOCVError):
    """
    This error is sent when trying to post to a socket for 5 time is failed 
    """
    def __init__(self, message="no monidtor id"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.message = message


class TooManyExceptionsVAOCVError(VAOCVError):
    """
    when too many exceptions get to main this exception shold be thrown
    """
    def __init__(self, message="no monidtor id"):
        self.expression=message
def handleVAOCVE(e,tb):
    """ 
    param: tb, a traceback of the stack
    param: e, an VACVOError
    return an exit code or -1 if should not exit
    """
    if not isinstance(e,VAOCVError):
        print(tb)
        print("tried to handle un exception which is not VACOVEerror")
        return(VACOVE_UNKNOWN_EXIT_CODE)
    elif isinstance(e,TooManyExceptionsVAOCVError):
        print(tb)
    elif isinstance (e,NoMonitorIDVAOCVError):
        return 5
    elif isinstance(e,OCRSocketVAOCVError):
        print(tb)
        return -1
    elif isinstance(e,TooManyExceptionsVAOCVError):
        print(tb)
        return(VACOVE_TOO_MANY_EXCEPTIONS_EXIT_CODE)    
    else:
        return(VACOVE_UNKNOWN_EXIT_CODE)

#####################################################
#      used for checking mathods not important      #
#####################################################


"""
def f():
    raise OCRSocketVAOCVError
def k():
    print("kkkkkkkkkkkk")
try:
    f()
except Exception as e:
    tb=traceback.format_exc()
    if isinstance(e,OCRSocketVAOCVError):
        print("OCR")
        print(tb)
        x=k
    elif isinstance(e,NoMonitorIDVAOCVError):
        print("ID")
        print(traceback)

x()

exit(6)
"""