#Imports
import sys
import cv2
# pylint: disable=E1101
# pylint: disable=E0401
# Disabling linting that is not supported by Pylint for C extensions such as OpenCV. See issue https://github.com/PyCQA/pylint/issues/1955 
import numpy
import requests
import json
import time
import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from socketIO_client_nexus import SocketIO, BaseNamespace

import VideoStream
from VideoStream import VideoStream
import AnalyzeFrame
import AnnotationParser
from AnnotationParser import AnnotationParser
# import ImageServer
# from ImageServer import ImageServer
import AnalyzeMeasures
# import AnalyzeMeasures2S
# import AnalyzeFrame2
from SocketsModule import SocketNamespace
from ExceptionsModule import *

class CameraCapture(object):

    def __IsInt(self,string):
        try: 
            int(string)
            return True
        except ValueError:
            return False
    

    def __init__(
            self,
            videoPath,
            onboardingMode,
            imageProcessingEndpoint = "",
            imageProcessingParams = "", 
            showVideo = False, 
            verbose = False,
            loopVideo = False,
            convertToGray = False,
            resizeWidth = 0,
            resizeHeight = 0,
            annotate = False,
            cognitiveServiceKey="",
            modelId=""):
        self.videoPath = videoPath
        self.onboardingMode = onboardingMode
        # Avihay's bug fix:
        # TODO: add argument to choose which kind of processing - file or stream
        if not self.__IsInt(videoPath):
            # case of a stream
            self.isWebcam = True
        else:
            # case of a video file
            self.isWebcam = False
        
        # TODO: remove all commands related to imageProcessingEndpoint. It's irelevant
        self.imageProcessingEndpoint = imageProcessingEndpoint
        if imageProcessingParams == "":
            self.imageProcessingParams = "" 
        else:
            self.imageProcessingParams = json.loads(imageProcessingParams)
        self.showVideo = showVideo
        self.verbose = verbose
        self.loopVideo = loopVideo
        self.convertToGray = convertToGray
        self.resizeWidth = resizeWidth
        self.resizeHeight = resizeHeight
        self.annotate = (self.imageProcessingEndpoint != "") and self.showVideo & annotate
        self.nbOfPreprocessingSteps = 0
        self.autoRotate = False
        self.vs = None
        # TODO: wrap in try and add default value
        self.monitor_id = os.getenv("DEVICE_ID")
        self.results_list = []

        if not self.onboardingMode: # live-stream mode, will use known boundries
            self.__get_boundries()
            # connect to server
            SOCKET_URL = os.getenv("SOCKET_URL")
            try:
                socketIO = SocketIO(SOCKET_URL, 443, BaseNamespace, False)
                time.sleep(3)
                self.ocrSocket = socketIO.define(SocketNamespace, '/ocr')
            except:
                print("Failed to open socket!")
                raise SocketInitVAOCVError("Can't establish a connection to the socket")
        else:
            self.__get_device_type_for_onboarding()
        
        if self.convertToGray:
            self.nbOfPreprocessingSteps +=1
        if self.resizeWidth != 0 or self.resizeHeight != 0:
            self.nbOfPreprocessingSteps +=1
        
        self.cognitiveServiceKey = cognitiveServiceKey
        self.modelId = modelId

        if self.verbose:
            print("Container vesrion: --> v1.1")
            print("Initialising the camera capture with the following parameters: ")
            print("   - Video path: " + str(self.videoPath))
            print("   - OnBoarding mode: " + str(self.onboardingMode))
            print("   - Device ID: " + str(self.monitor_id))
            print("   - Device type: " + str(self.device_type))
            print("   - Computer vision model: " + str(os.getenv('CV_MODEL', "")))
            # print("   - Image processing endpoint: " + self.imageProcessingEndpoint)
            # print("   - Image processing params: " + json.dumps(self.imageProcessingParams))
            # print("   - Show video: " + str(self.showVideo))
            # print("   - Loop video: " + str(self.loopVideo))
            # print("   - Convert to gray: " + str(self.convertToGray))
            # print("   - Resize width: " + str(self.resizeWidth))
            # print("   - Resize height: " + str(self.resizeHeight))
            # print("   - Annotate: " + str(self.annotate))
            print()
        
        self.displayFrame = None
        # if self.showVideo:
        #     self.imageServer = ImageServer(5012, self)
        #     self.imageServer.start()
        #     # self.imageServer.run()
        
        COMPUTER_VISION_ENDPOINT = os.environ["COMPUTER_VISION_ENDPOINT"]
        COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]
        self.computervision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))

    def __get_boundries(self):
        API_URL = os.getenv("API_URL")
        url = API_URL + "/" + self.monitor_id + "?image=false"
        response = None
        for trail in range(4):
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    raise(Exception("Bad API Status Code: " + str(response.status_code)))
            except Exception as e:
                print("Not getting MES Setup results via API.")
                if trail == 3:
                    if response is not None:
                        if response.status_code != 200:
                            raise HttpResultsWrongCodeVAOCVError("Bad API response trying to Get, status code: " + str(response.status_code)) 
                    else:
                        raise HttpCantGetResultsVAOCVError("Can't Get Setup Results via API!: "+ str(e))
                time.sleep(1)
                continue
            break

        json_response = response.text
        dict_response = json.loads(json_response)
        # check API results:
        important_keys = ['boundries', 'areas', 'type', 'corners'] 
        try:
            result = all((k in dict_response.keys()) and (dict_response[k]) for k in important_keys)   
            if not result:
                raise HttpResultsAreEmptyOrMissingVAOCVError("One or More of the next Fields are Empty\Missing: 'boundries', 'areas', 'type', 'corners' in API Get Result") 
        except HttpResultsAreEmptyOrMissingVAOCVError as e:
            raise e
        
        boundries_list = dict_response["boundries"]
        
        # TODO: Change back to type:value in production 
        # self.boundries = {item['id']: item['point'] for item in boundries_list}
        self.boundries = {item['type']: item['point'] for item in boundries_list}
        
        areas_list = dict_response["areas"]
        self.areas_of_interes = {item['id']: item['point'] for item in areas_list}
        self.device_type = dict_response['type']
        os.environ["DEVICE_TYPE"] = self.device_type
        self.setupMarkersCorners = [(d['point'][0], d['point'][1]) for d in dict_response["corners"]]
        return


    def __get_device_type_for_onboarding(self):
        API_URL = os.getenv("API_URL")
        url = API_URL + "/" + self.monitor_id + "?image=false"
        response = None
        for trail in range(4):
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    raise(Exception("Bad API Status Code: " + str(response.status_code)))
            except Exception as e:
                print("Not getting device data via API.")
                if trail == 3:
                    if response is not None:
                        if response.status_code != 200:
                            raise HttpResultsWrongCodeVAOCVError("Bad API response while trying to Get, status code: " + str(response.status_code)) 
                    else:
                        raise HttpCantGetResultsVAOCVError("Can't Get device type via API!: "+ str(e))
                time.sleep(1)
                continue
            break
        json_response = response.text
        dict_response = json.loads(json_response)
        # check API results:
        important_keys = ['type'] 
        try:
            result = all((k in dict_response.keys()) and (dict_response[k]) for k in important_keys)   
            if not result:
                raise HttpResultsAreEmptyOrMissingVAOCVError("One or More of the next Fields are Empty\Missing: ['type'],  in API Get Result") 
        except HttpResultsAreEmptyOrMissingVAOCVError as e:
            raise e
        
        self.device_type = dict_response['type']
        os.environ["DEVICE_TYPE"] = self.device_type
        print(self.device_type)
        return


    def __annotate(self, frame, response):
        AnnotationParserInstance = AnnotationParser()
        #TODO: Make the choice of the service configurable
        listOfRectanglesToDisplay = AnnotationParserInstance.getCV2RectanglesFromProcessingService1(response)
        for rectangle in listOfRectanglesToDisplay:
            cv2.rectangle(frame, (rectangle(0), rectangle(1)), (rectangle(2), rectangle(3)), (0,0,255),4)
        return

    
    def __sendFrameForProcessing(self, frame):
        # TODO: try-except-throw - by what Lior wants for the wrapper
        if self.onboardingMode:
            corners_flag = AnalyzeMeasures.AnalyzeMeasures(frame, self.computervision_client)
            # reutrn True if 4 corners detected, else - False:
            return corners_flag
            # AnalyzeMeasures2.AnalyzeFrame(frame, self.computervision_client)
        else:
            # new_old_corners = AnalyzeFrame.AnalyzeFrame(frame, self.computervision_client, self.boundries, self.areas_of_interes, self.ocrSocket, self.setupMarkersCorners)
            new_old_corners, new_results_list = AnalyzeFrame.AnalyzeFrame(frame, self.computervision_client, self.boundries, self.areas_of_interes, self.ocrSocket, self.setupMarkersCorners, self.results_list)
            # TODO: try-except
            self.setupMarkersCorners = new_old_corners
            self.results_list = new_results_list
            # AnalyzeFrame2.AnalyzeFrame(frame, self.computervision_client, self.boundries)
        return True

    
    def __displayTimeDifferenceInMs(self, endTime, startTime):
        return str(int((endTime-startTime) * 1000)) + " ms"

    
    def __enter__(self):
        try:
            for i in range(4):
                cap = cv2.VideoCapture(self.videoPath)
                if cap.isOpened()== False:
                    if i == 3: #we tryed 4 times
                        raise VideoStreamInitProblemVAOCVError('Video Path Is Not Open.' + str(self.videoPath))
                    else:
                        time.sleep(2)
                        continue
                else:
                    # print("releasing")
                    cap.release()
                    break #link is good
        except VideoStreamInitProblemVAOCVError as e:
            raise(e)
        except Exception as e:
            print('UNKNOWN Exception while Opening Stream')
            raise e
        if self.isWebcam:
            #The VideoStream class always gives us the latest frame from the webcam. It uses another thread to read the frames.
            # self.vs = VideoStream(int(self.videoPath)).start()
            self.vs = VideoStream(self.videoPath).start()
            time.sleep(1.0)#needed to load at least one frame into the VideoStream class
            #self.capture = cv2.VideoCapture(int(self.videoPath))
        else:
            #In the case of a video file, we want to analyze all the frames of the video thus are not using VideoStream class
            self.capture = cv2.VideoCapture(self.videoPath)
        return self

    
    def get_display_frame(self):
        return self.displayFrame

    
    def start(self):
        frameCounter = 0
        perfForOneFrameInMs = None
        while True:
            if self.showVideo or self.verbose:
                startOverall = time.time()
            if self.verbose:
                startCapture = time.time()

            frameCounter +=1
            if self.isWebcam:
                try:
                    if self.vs.stopped:
                        print('stopping')
                        raise VideoStreamReadProblemVAOCVError('Video stream Stopped!')
                    frame = self.vs.read()
                except VideoStreamReadProblemVAOCVError as e:
                    raise(e)
                except Exception as e:
                    print('UNKNOWN Exception in Middle of Reading Stream')
                    raise e
            else:
                frame = self.capture.read()[1]
                if frameCounter == 1:
                    if self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) < self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):
                        self.autoRotate = True
                if self.autoRotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) #The counterclockwise is random...It coudl well be clockwise. Is there a way to auto detect it?
            if self.verbose:
                if frameCounter == 1:
                    if not self.isWebcam:
                        print("Original frame size: " + str(int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))) + "x" + str(int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                        print("Frame rate (FPS): " + str(int(self.capture.get(cv2.CAP_PROP_FPS))))
                print("Frame number: " + str(frameCounter))
                print("Time to capture (+ straighten up) a frame: " + self.__displayTimeDifferenceInMs(time.time(), startCapture))
                startPreProcessing = time.time()
            
            #Loop video
            if not self.isWebcam:             
                if frameCounter == self.capture.get(cv2.CAP_PROP_FRAME_COUNT):
                    if self.loopVideo: 
                        frameCounter = 0
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        break

            #Pre-process locally
            if self.nbOfPreprocessingSteps == 1 and self.convertToGray:
                preprocessedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.nbOfPreprocessingSteps == 1 and (self.resizeWidth != 0 or self.resizeHeight != 0):
                preprocessedFrame = cv2.resize(frame, (self.resizeWidth, self.resizeHeight))

            if self.nbOfPreprocessingSteps > 1:
                preprocessedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                preprocessedFrame = cv2.resize(preprocessedFrame, (self.resizeWidth,self.resizeHeight))
            
            if self.verbose:
                print("Time to pre-process a frame: " + self.__displayTimeDifferenceInMs(time.time(), startPreProcessing))
                startEncodingForProcessing = time.time()

            #Process externally
            if self.imageProcessingEndpoint != "":

                #Encode frame - not in use for now
                if self.nbOfPreprocessingSteps == 0:
                    encodedFrame = cv2.imencode(".jpg", frame)[1].tostring()
                else:
                    encodedFrame = cv2.imencode(".jpg", preprocessedFrame)[1].tostring()

                if self.verbose:
                    print("Time to encode a frame for processing: " + self.__displayTimeDifferenceInMs(time.time(), startEncodingForProcessing))
                    startProcessingExternally = time.time()

                #Send for processing
                if self.onboardingMode:
                    print('Onboarding mode, will stop stream after 1 frame')
                    try:
                        response = self.__sendFrameForProcessing(encodedFrame)
                    except Exception as e:
                        self.vs.stopped = True
                        time.sleep(2)
                        self.vs.stream.release()
                        raise(e)
                    if response:
                        # if found 4 corners and there's a good mapping - stop and return
                        self.vs.stopped = True
                        time.sleep(2)
                        self.vs.stream.release()
                        break
                else:
                    try:
                        response = self.__sendFrameForProcessing(encodedFrame)
                    except Exception as e:
                        self.vs.stopped = True
                        time.sleep(2)
                        self.vs.stream.release()
                        raise(e)

                
                if self.verbose:
                    print("Time to process frame externally: " + self.__displayTimeDifferenceInMs(time.time(), startProcessingExternally))
                    startSendingToEdgeHub = time.time()

            #Display frames
            if self.showVideo:
                try:
                    if self.nbOfPreprocessingSteps == 0:
                        if self.verbose and (perfForOneFrameInMs is not None):
                            cv2.putText(frame, "FPS " + str(round(1000/perfForOneFrameInMs, 2)),(10, 35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255), 2)
                        if self.annotate:
                            #TODO: fix bug with annotate function
                            self.__annotate(frame, response)
                        self.displayFrame = cv2.imencode('.jpg', frame)[1].tobytes()
                    else:
                        if self.verbose and (perfForOneFrameInMs is not None):
                            cv2.putText(preprocessedFrame, "FPS " + str(round(1000/perfForOneFrameInMs, 2)),(10, 35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255), 2)
                        if self.annotate:
                            #TODO: fix bug with annotate function
                            self.__annotate(preprocessedFrame, response)
                        self.displayFrame = cv2.imencode('.jpg', preprocessedFrame)[1].tobytes()
                except Exception as e:
                    print("Could not display the video to a web browser.") 
                    print('Excpetion -' + str(e))
                if self.verbose:
                    if 'startDisplaying' in locals():
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startDisplaying))
                    elif 'startSendingToEdgeHub' in locals():
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startSendingToEdgeHub))
                    else:
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startEncodingForProcessing))
                perfForOneFrameInMs = int((time.time()-startOverall) * 1000)
                if not self.isWebcam:
                    waitTimeBetweenFrames = max(int(1000 / self.capture.get(cv2.CAP_PROP_FPS))-perfForOneFrameInMs, 1)
                    print("Wait time between frames :" + str(waitTimeBetweenFrames))
                    if cv2.waitKey(waitTimeBetweenFrames) & 0xFF == ord('q'):
                        break

            if self.verbose:
                perfForOneFrameInMs = int((time.time()-startOverall) * 1000)
                print("Total time for one frame: " + self.__displayTimeDifferenceInMs(time.time(), startOverall))

    def __exit__(self, exception_type, exception_value, traceback):
        if not self.isWebcam:
            self.capture.release()
        if self.showVideo:
            self.imageServer.close()
            cv2.destroyAllWindows()