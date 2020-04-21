import sys, traceback
import time
import os
from socketIO_client_nexus import SocketIO, BaseNamespace
from SocketsModule import SocketNamespace
import requests
import json

VACOVE_UNKNOWN_EXIT_CODE=50
VACOVE_TOO_MANY_EXCEPTIONS_EXIT_CODE=99
VACOVE_NO_MONITOR=7
VACOVE_HTTP_STATUS_CODE_EXIT_CODE=3

def getCVMNGADDRESS():
    api_url_unclean = os.getenv('API_URL')
    api_url=api_url_unclean.split("/api")[0]+"/api/cv_mng"
    return api_url

CV_MNG_ADDRESS=getCVMNGADDRESS()


class VAOCVError(Exception):
    """
    This is a base class Error for the VideoAnalyticsOpenCV (don't use this)  
    """
    def __init__(self):
        self.send_socket = True
        self.send_api = True
        self.error_string = "UNKNOWN_ERROR" 
        self.exit_code = VACOVE_UNKNOWN_EXIT_CODE


class UnknownVAOCVError(VAOCVError):
    """
    This is a general Error, if you don't to know which Error to throw, 
    """
    def __init__(self, message="Unknown Error"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = VACOVE_UNKNOWN_EXIT_CODE
        self.error_string = "UNKNOWN_ERROR"


class MSOCRServiceVAOCVError(VAOCVError):
    """
    This error is sent when can't get an answear from MS OCR Cognitive Service engine 
    """
    def __init__(self, message="Failed to get MSOCR results"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = 20
        self.error_string = "NO_MSOCR_CONNECTION"


class VideoStreamProblemVAOCVError(VAOCVError):
    """
    This error is sent when the video stream cannot be opened 
    """
    def __init__(self, message="Failed to initialize video stream"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = 1
        self.error_string = "NO_INIT_STREAM"


class StreamStoppedVAOCVError(VAOCVError):
    def __init__(self, message="Stream stopped"):
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = 2
        self.error_string = "STREAM_STOPPED"
   
        
class NoInitStreamVAOCVError(VAOCVError):
    def __init__(self, message="Stream stopped"):
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = 1
        self.error_string = "NO_INIT_STREAM"


class OCRSocketVAOCVError(VAOCVError):
    """
    This error is sent when trying to post to a socket for 5 time is failed 
    """
    def __init__(self, message="Failed to send data via OCR"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = False
        self.send_api = True
        self.exit_code = 3
        self.error_string = "NO_UI_CONNECTION"


class APIMESSetupVAOCVError(VAOCVError):
    """
    This error is sent when trying to post to a socket for 5 time is failed 
    """
    def __init__(self, message="Failed to send MES Results via API"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = True
        self.send_api = True
        self.exit_code = 8
        self.error_string = "API_MES_SETUP"


class EnvVarsVAOCVError(VAOCVError):
    """
    This error is sent when trying to read env therefor it will not send data
    """
    def __init__(self, message="Failed to Load Enviroment Variables"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = False 
        self.send_api = False
        self.exit_code = 9
        self.error_string = "ENV_ERROR"


class NoMonitorIDVAOCVError(VAOCVError):
    """
    This error is sent when trying to post to a socket for 5 time is failed 
    """
    def __init__(self, message="no monidtor id"):
        """
        :param object message: if you would like to add some info it can be passed with this string 
        as "Unkown Error" by default. Optional.
        """
        self.expression = message
        self.send_socket = False 
        self.send_api = False
        self.exit_code = 5
        self.error_string = "ENV_ERROR"


class TooManyExceptionsVAOCVError(VAOCVError):
    """
    when too many exceptions get to main this exception shold be thrown
    """
    def __init__(self, message="too many errors"):
        self.expression = message
        self.send_socket = False 
        self.send_api = False
        self.exit_code = 9
        self.error_string = "ENV_ERROR"


class HttpFailedToSendErrorVAOCVError(VAOCVError):
    """
        Not sure if needed
    """
    def __init__(self, message="too many expression"):
        self.expression = message
        self.send_socket = False 
        self.send_api = False
        self.exit_code = 9
        self.error_string = "HTTP_FAILED_SEND"


class HttpBoundriesVAOCVError(VAOCVError):
    def __init__(self, message="can't get boundires"):
        self.expression = message
        self.send_socket = True 
        self.send_api = True
        self.exit_code = 9
        self.error_string = "HTTP_BOUNDRIES"


class OCRBusyVAOCVError(VAOCVError):
     def __init__(self, message="ocr is busy"):
        self.expression = message
        self.send_socket = True 
        self.send_api = True
        self.exit_code = 9
        self.error_string = "OCR_BUSY"
    
    
class OCRFrameVAOCVError(VAOCVError):
    def __init__(self, message="ocr frame proccessing problem"):
        self.expression = message
        self.send_socket = True 
        self.send_api = True
        self.exit_code = 9
        self.error_string = "OCR_FRAME_ERROR"


class StamVAOCVError(VAOCVError):
    def __init__(self, message="ocr frame proccessing problem"):
        self.expression = message
        self.send_socket = True 
        self.send_api = False
        self.exit_code = 9
        self.error_string = "STAM_ERROR"


def sendFailureSocket(equipment_id,error,ocrSocket):
    try:
        json_dict = {}
        json_dict["JsonData"] = None
        json_dict["DeviceID"] = str(equipment_id)
        json_dict["deviceType"] = None
        json_dict["error"] = str(error)
        ocrSocket.emit('data',json.dumps(json_dict))
        return True
    except:
        return False 


def cvMngPost(equipment_id,error,url=CV_MNG_ADDRESS):
    str_eid=str(equipment_id)
    str_error=str(error)
    for _ in range(4):
        try:
            r=requests.post(url,json={"med_eq_id": str_eid, "error_code":str_error })
            r.raise_for_status()
            return True                       
        except:
            time.sleep(0.2)
    return False


def handleVAOCVE(e,tb,device_id,ocrSocket):
    """ 
    param: tb, a traceback of the stack
    param: e, an VACVOError
    param: 
    return an exit code or -1 if should not exit
    """
    print(tb)
    if not isinstance(e,VAOCVError):
        print("tried to handle un exception which is not VACOVEerror")
        return(VACOVE_UNKNOWN_EXIT_CODE)
    else:
        if e.send_socket:
            print("send via Sucsses:" +str(sendFailureSocket(device_id,e.error_string,ocrSocket)))
        if e.send_api:
            cvMngPost(device_id,e.error_string)
        return e.exit_code


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