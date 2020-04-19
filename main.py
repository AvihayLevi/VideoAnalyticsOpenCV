# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import os
import random
import sys
import time

import CameraCapture
from CameraCapture import CameraCapture
from ExceptionsModule import *


# global counters
SEND_CALLBACKS = 0
UNKNOWN_ERROR_CODE = 100
TOO_MANY_EXCEPTIONS_EXIT_CODE=99

def main(
        videoPath,
        onboardingMode,
        imageProcessingEndpoint="",
        imageProcessingParams="",
        showVideo=False,
        verbose=False,
        loopVideo=True,
        convertToGray=False,
        resizeWidth=0,
        resizeHeight=0,
        annotate=False,
        cognitiveServiceKey="",
        modelId="",
        max_exceptions=-1,
        num_exceptions=0
):
    '''
    Capture a camera feed, send it to processing and forward outputs to EdgeHub

    :param int videoPath: camera device path such as /dev/video0 or a test video file such as /TestAssets/myvideo.avi. Mandatory.
    :param bool onboardingMode: is onBoarding mode or live-stream mode
    :param str imageProcessingEndpoint: service endpoint to send the frames to for processing. Example: "http://face-detect-service:8080". Leave empty when no external processing is needed (Default). Optional.
    :param str imageProcessingParams: query parameters to send to the processing service. Example: "'returnLabels': 'true'". Empty by default. Optional.
    :param bool showVideo: show the video in a windows. False by default. Optional.
    :param bool verbose: show detailed logs and perf timers. False by default. Optional.
    :param bool loopVideo: when reading from a video file, it will loop this video. True by default. Optional.
    :param bool convertToGray: convert to gray before sending to external service for processing. False by default. Optional.
    :param int resizeWidth: resize frame width before sending to external service for processing. Does not resize by default (0). Optional.
    :param int resizeHeight: resize frame width before sending to external service for processing. Does not resize by default (0). Optional.ion(
    :param bool annotate: when showing the video in a window, it will annotate the frames with rectangles given by the image processing service. False by default. Optional. Rectangles should be passed in a json blob with a key containing the string rectangle, and a top left corner + bottom right corner or top left corner with width and height.
    '''
    try:
        print("\nPython %s\n" % sys.version)
        print("RStream Video Analyzer. Press Ctrl-C to exit.")
        with CameraCapture(videoPath, onboardingMode, imageProcessingEndpoint, imageProcessingParams, showVideo, verbose, loopVideo, convertToGray, resizeWidth, resizeHeight, annotate, cognitiveServiceKey, modelId) as cameraCapture:
            cameraCapture.start()
    
    except Exception as e: 
        print(e)
        tb = traceback.format_exc()
        if isinstance(e,KeyboardInterrupt):
            print("Camera capture module stopped")
            sys.exit(0)
        else:
            exitCode = handleException(e,tb)
            # Check if exception is from the kind that needs to restart the process localy and make main run again:
            if (exitCode == -1):
                if (num_exceptions < max_exceptions or max_exceptions == -1):
                    print("restatrting CameraCapture after dealing with exception error " + str(e.__class__))
                    tb = None
                    num_exceptions += 1
                    main(videoPath, onboardingMode, imageProcessingEndpoint, imageProcessingEndpoint, showVideo,
                        verbose, loopVideo, convertToGray, resizeWidth, resizeHeight, annotate,cognitiveServiceKey, modelId, max_exceptions, 
                        num_exceptions)
                else:
                    raise TooManyExceptionsVAOCVError
            else:
                sys.exit(exitCode)
    

def handleException(e,tb):
    """
    this function should hanndle any exception but KeyboardInterrupt that happens in CameraCapture,
    in case there is a known Error a VACOVError should be thrown  
    """
    if isinstance(e,VAOCVError):
        exitCode = handleVAOCVE(e,tb)
    else: 
        exitCode = handleUnknownError(e,tb)
    return exitCode
    

def handleUnknownError(e,tb):
    return UNKNOWN_ERROR_CODE


def __convertStringToBool(env):
    if env in ['True', 'TRUE', '1', 'y', 'YES', 'Y', 'Yes', 'True']:
        return True
    elif env in ['False', 'FALSE', '0', 'n', 'NO', 'N', 'No']:
        return False
    else:
        raise ValueError('Could not convert string to bool.')


if __name__ == '__main__':
    # Check if all Enviroment Variables passed:
    try:
        API_URL = os.getenv('API_URL', "")
        SOCKET_URL = os.getenv('SOCKET_URL', "")
        CV_MODEL = os.getenv('CV_MODEL', "")
        FRAME_DELAY = os.getenv('FRAME_DELAY', "")
        VIDEO_PATH = os.getenv('VIDEO_PATH', "")
        ONBOARDING_MODE = os.getenv('ONBOARDING_MODE', "")
        COMPUTER_VISION_SUBSCRIPTION_KEY = os.getenv('COMPUTER_VISION_SUBSCRIPTION_KEY', "")
        COMPUTER_VISION_ENDPOINT = os.getenv('COMPUTER_VISION_ENDPOINT', "")
        DEVICE_ID = os.getenv('DEVICE_ID', "")

        env_vars_list = [API_URL, SOCKET_URL, CV_MODEL, FRAME_DELAY, VIDEO_PATH, ONBOARDING_MODE, COMPUTER_VISION_SUBSCRIPTION_KEY, COMPUTER_VISION_ENDPOINT, DEVICE_ID]
        if "" in env_vars_list:
            print("Missing Enviroment Variable in [API_URL, SOCKET_URL, CV_MODEL, FRAME_DELAY, VIDEO_PATH, ONBOARDING_MODE, COMPUTER_VISION_SUBSCRIPTION_KEY, COMPUTER_VISION_ENDPOINT, DEVICE_ID]: \n", \
                 [API_URL, SOCKET_URL, CV_MODEL, FRAME_DELAY, VIDEO_PATH, ONBOARDING_MODE, "SECRET-Not-Printing", COMPUTER_VISION_ENDPOINT, DEVICE_ID])
            raise EnvVarsVAOCVError("Not all Enviroment Variables passed succesfuly")
    except EnvVarsVAOCVError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print("UNKNOWN ERROR: ", e)
        sys.exit(1)
    
    try:
        VIDEO_PATH = os.environ['VIDEO_PATH']
        print("Vid Path", VIDEO_PATH)
        ONBOARDING_MODE = __convertStringToBool(os.getenv('ONBOARDING_MODE', 'True'))
        IMAGE_PROCESSING_ENDPOINT = os.getenv('IMAGE_PROCESSING_ENDPOINT', "bla")
        IMAGE_PROCESSING_PARAMS = os.getenv('IMAGE_PROCESSING_PARAMS', "")
        SHOW_VIDEO = __convertStringToBool(os.getenv('SHOW_VIDEO', 'False'))
        VERBOSE = __convertStringToBool(os.getenv('VERBOSE', 'False'))
        LOOP_VIDEO = __convertStringToBool(os.getenv('LOOP_VIDEO', 'True'))
        CONVERT_TO_GRAY = __convertStringToBool(os.getenv('CONVERT_TO_GRAY', 'False'))
        RESIZE_WIDTH = int(os.getenv('RESIZE_WIDTH', 0))
        RESIZE_HEIGHT = int(os.getenv('RESIZE_HEIGHT', 0))
        ANNOTATE = __convertStringToBool(os.getenv('ANNOTATE', 'False'))
        COGNITIVE_SERVICE_KEY = os.getenv('COGNITIVE_SERVICE_KEY', "")
        MODEL_ID = os.getenv('MODEL_ID', "")
        MAX_EXCEPTIONS = int(os.getenv('MAX_EXCEPTIONS',-1)) # set the maximum of accepted excpetions handled befor crashing when -1 (default) handle all exceptions 
    except ValueError as error:
        print(error)
        sys.exit(1)

    main(VIDEO_PATH, ONBOARDING_MODE, IMAGE_PROCESSING_ENDPOINT, IMAGE_PROCESSING_PARAMS, SHOW_VIDEO,
         VERBOSE, LOOP_VIDEO, CONVERT_TO_GRAY, RESIZE_WIDTH, RESIZE_HEIGHT, ANNOTATE, COGNITIVE_SERVICE_KEY, MODEL_ID, MAX_EXCEPTIONS)
