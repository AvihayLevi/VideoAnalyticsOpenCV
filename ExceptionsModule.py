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